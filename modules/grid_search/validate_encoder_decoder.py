'''
EncoderDecoderValidation: drives the real OFDM hardware chain (ModulateDataOFDM ->
ApplyEncoder -> Send -> Measure -> Resample -> ApplyDecoder -> DemodulateDataOFDM)
with frozen encoder/decoder pairs and writes a grid-search-style run folder.

Each validated E/D model gets runs/<run_id>/ with metrics.json (BER, rRMSE%, EVM%),
constellation.png and waveform.png. A single validation.zarr at the experiment root
stores the sent/received time-domain frames grouped by E/D model id.

Goal: run at the end of the test grid search on the live channel.
'''
import json
from pathlib import Path

import numpy as np
import torch
import zarr
from matplotlib.figure import Figure

from pyflux.core.block import Signal
from pyflux.core.chain import Chain
from modules.experimental_blocks import ApplyEncoder, ApplyDecoder
from modules.grid_search.base import GridSearchBase
from modules.grid_search.encoder_decoder import ARCH_KEYS
from modules.equalization_comparison import (
    DEFAULT_FONTS as EQUALIZATION_COMPARISON_FONTS,
    plot_equalization_comparison,
    save_equalization_constellations,
)
from modules.models import TCN
from modules.utils import calculate_BER, evm_pct

N_CONSTELLATION_TRIALS = 4  # trials shown in the equalization-comparison figure


class EncoderDecoderValidation(GridSearchBase):
    def __init__(self, ed_models, ofdm_blocks, num_trials, constellation, clip_value,
                 noise_floor_points=0, higher_order_context=None, device="cpu", seed=0,
                 experiments_dir=None, experiment_name="ed_validation", run_prefix=None,
                 debug=False):
        # (modulate, send, measure, resample, demod) or with an optional FractionalSync
        # block inserted just before demod
        self.ofdm_blocks = ofdm_blocks
        self.num_trials = num_trials
        self.constellation = constellation
        self.clip_value = clip_value
        self.noise_floor_points = int(noise_floor_points)

        # when set (dict with f_min, f_max, symbol_period_s), a higher-order-test
        # constellation diagram is produced that also reports the spectral efficiency
        self.higher_order_context = higher_order_context

        points = [{"model": m["run_id"],
                   "params": {k: m["arch"][k] for k in ARCH_KEYS},
                   "channel_form": m["channel_form"],
                   "channel_run_id": m.get("channel_run_id"),
                   "channel_receptive_field": m.get("channel_receptive_field"),
                   "channel_distribution": m.get("channel_distribution", "none"),
                   "checkpoint": str(m["checkpoint"])}
                  for m in ed_models]
        super().__init__(points, {"ed_models": [m["run_id"] for m in ed_models]},
                         {"num_trials": num_trials}, experiments_dir, device, seed,
                         experiment_name, run_prefix=run_prefix,
                         extra_manifest={"num_trials": num_trials})
        self.rank_by = "evm_pct"
        self.debug = debug

    def _prepare(self):
        self.val_group = zarr.open_group(self.exp_dir / "validation.zarr", mode="a")
        return None

    @staticmethod
    def _bits(source_bits):
        return np.array([int(b) for b in source_bits], dtype="uint8")

    def _load_pair(self, arch, checkpoint):
        encoder, decoder = TCN(**arch).to(self.device), TCN(**arch).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        for model in (encoder, decoder):
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
        return encoder, decoder

    def _run_point(self, point, run_dir, context) -> dict:
        modulate, send, measure, resample, *sync_blocks, demod = self.ofdm_blocks
        encoder, decoder = self._load_pair(point["params"], point["checkpoint"])
        chain = Chain([modulate,
                       ApplyEncoder(encoder, modulate, self.clip_value, self.device),
                       send, measure, resample, *sync_blocks,
                       ApplyDecoder(decoder, demod, self.device, debug=self.debug),
                       demod])

        sent_syms, recv_syms, true_bits, sent_time, recv_time = [], [], [], [], []
        frac_delays, encoder_powers, encoder_drives = [], [], []
        fft_length = demod.ofdm_symbol_length_with_cp - demod.cyclic_prefix_length
        freqs, example = None, None
        # per-stage constellations for the first few trials (equalization-comparison figure)
        ed_constellations = {"Sent": [], "Encoded": [], "Received": [], "Decoded": []}
        sync_outliers_skipped = 0
        max_sync_retries = 3
        for i in range(self.num_trials):
            for attempt in range(max_sync_retries + 1):
                x = chain.run(Signal(data=np.zeros(1), sampling_rate=modulate.fs_out))
                art = x.artifact_container
                if not art.get("sync_outlier"):
                    break
                sync_outliers_skipped += 1
                residual = art.get("residual_delay_samples", float("nan"))
                retries_left = max_sync_retries - attempt
                if retries_left > 0:
                    print(f"  sync outlier on trial {i + 1} (residual {residual:+.3f} samples), "
                          f"retrying ({retries_left} left)")

            if self.debug:
                print(f"trial {i + 1}/{self.num_trials}  sent={art['sent_symbols'].shape} "
                      f"recv={art['received_symbols'].shape} "
                      f"sent_time={art['sent_baseband'].shape} "
                      f"recv_time={x.data.shape}")
                
            sent_syms.append(np.asarray(art["sent_symbols"]))
            recv_syms.append(np.asarray(art["received_symbols"]))
            true_bits.append(self._bits(art["source_bits"]))
            sent_time.append(np.asarray(art["sent_baseband"], dtype="float32"))
            recv_time.append(np.asarray(x.data, dtype="float32"))
            if "fractional_delay_samples" in art:
                frac_delays.append(float(art["fractional_delay_samples"]))
            # symbol power in the same units as POWER_MIN/POWER_MAX (mean-square of the
            # payload, what ModulateDataOFDM's random scaling targets)
            encoder_output = np.asarray(art["encoder_output"], dtype=np.float64)
            encoder_powers.append(float(np.mean(encoder_output[-fft_length:] ** 2)))
            encoder_drives.append(encoder_output.astype("float32"))
            freqs = np.asarray(art["subcarrier_freqs_hz"])

            # equalization-comparison stages: the constellation as it passes encoder ->
            # channel -> decoder. Encoded/Received are FFT'd from the time-domain symbols;
            # Sent/Decoded are the demodulated frequency-domain symbols.
            if len(ed_constellations["Sent"]) < N_CONSTELLATION_TRIALS:
                ed_constellations["Sent"].append(np.asarray(art["sent_symbols"]))
                ed_constellations["Encoded"].append(self._symbol_constellation(art["encoder_output"], demod))
                ed_constellations["Received"].append(self._symbol_constellation(art["decoder_input"], demod))
                ed_constellations["Decoded"].append(np.asarray(art["received_symbols"]))

            if i == 0:
                example = {k: np.asarray(art[k]) for k in
                           ("encoder_input", "encoder_output", "decoder_input", "decoder_output")}
                example["decoder_window"] = art.get("decoder_window")

        example["residual"] = sent_time[0] - recv_time[0]  # in-band sent vs recovered symbol

        sent_syms, recv_syms = np.stack(sent_syms), np.stack(recv_syms)
        sent_t = torch.tensor(sent_syms)
        recv_t = torch.tensor(recv_syms)
        metrics = {
            "evm_pct": evm_pct(sent_t, recv_t).item(),
            "ber": calculate_BER(recv_t.flatten(), np.concatenate(true_bits), self.constellation),
            "num_params": encoder.get_num_params() + decoder.get_num_params(),
            "channel_form": point["channel_form"],
            "channel_run_id": point.get("channel_run_id"),
            "channel_receptive_field": point.get("channel_receptive_field"),
            "channel_distribution": point.get("channel_distribution", "none"),
        }
        if frac_delays:
            metrics["frac_delay_std_samples"] = float(np.std(frac_delays))
        metrics["encoder_power_mean"] = float(np.mean(encoder_powers))
        metrics["encoder_power_std"] = float(np.std(encoder_powers))
        metrics["sync_outliers_skipped"] = sync_outliers_skipped

        self._store_waveforms(run_dir.name, np.stack(sent_time), np.stack(recv_time), point["channel_form"],
                              frac_delays=np.asarray(frac_delays, dtype="float32") if frac_delays else None,
                              encoder_drive=np.stack(encoder_drives))
        label = f"{run_dir.name} | channel: {point['channel_form']}"
        self._plot_constellation(run_dir, sent_syms, recv_syms, freqs,
                                 ed_run_id=run_dir.name,
                                 channel_form=point["channel_form"],
                                 channel_run_id=point.get("channel_run_id"),
                                 evm=metrics["evm_pct"])
        self._plot_waveform(run_dir, example, label)
        self._plot_encoder_power(run_dir, encoder_powers)

        # final step: a few trials of raw OFDM with no encoder/decoder in the loop, so the
        # equalization-comparison figure can contrast the E/D against no ML equalization
        no_ed_chain = Chain([modulate, send, measure, resample, *sync_blocks, demod])
        no_ed_sent, no_ed_received = [], []
        for _ in range(N_CONSTELLATION_TRIALS):
            x = no_ed_chain.run(Signal(data=np.zeros(1), sampling_rate=modulate.fs_out))
            no_ed_sent.append(np.asarray(x.artifact_container["sent_symbols"]))
            no_ed_received.append(np.asarray(x.artifact_container["received_symbols"]))
        self._plot_equalization_comparison(run_dir, point["channel_form"],
                                           ed_constellations, no_ed_sent, no_ed_received, freqs)

        if self.noise_floor_points > 0:
            metrics["noise_floor_evm_pct"] = self._plot_noise_floor(
                run_dir, encoder, decoder, sent_syms, recv_syms, freqs)

        if self.higher_order_context is not None:
            self._plot_higher_order_constellation(run_dir, sent_syms, recv_syms, freqs,
                                                  evm=metrics["evm_pct"], ber=metrics["ber"])
        return metrics

    def _plot_higher_order_constellation(self, run_dir, sent_syms, recv_syms, freqs, evm, ber):
        '''Received equalized constellation for a higher-order APSK test, styled like the
        equalization-comparison panels (unit-power scatter coloured by carrier frequency,
        ideal reference overlaid in red). The received EVM and BER measured over every trial
        and the nominal spectral efficiency at this modulation order go in the title, so they
        do not overlap the constellation points.'''
        context = self.higher_order_context
        sent_syms = np.asarray(sent_syms)
        recv_syms = np.asarray(recv_syms)

        bits_per_carrier = self.constellation.bits_per_symbol
        modulation_order = 2 ** bits_per_carrier
        num_carriers = sent_syms.shape[1]
        bandwidth_hz = float(context["f_max"] - context["f_min"])
        symbol_period_s = float(context["symbol_period_s"])
        spectral_efficiency = num_carriers * bits_per_carrier / (symbol_period_s * bandwidth_hz)

        scale = np.sqrt(np.mean(np.abs(sent_syms) ** 2)) + 1e-12
        received = recv_syms.ravel() / scale
        reference = sent_syms.ravel() / scale
        carrier_color = np.tile(np.asarray(freqs), sent_syms.shape[0])

        fig = Figure(figsize=(7.5, 6.5))
        ax = fig.subplots()
        scatter = ax.scatter(received.real, received.imag, s=8, c=carrier_color, cmap="viridis")
        ax.scatter(reference.real, reference.imag, s=45, marker="x", c="red",
                   linewidth=1.3, zorder=5)

        ax.set_xlabel("In-Phase")
        ax.set_ylabel("Quadrature")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", "box")
        fig.colorbar(scatter, ax=ax, label="Carrier Frequency (Hz)", fraction=0.046, pad=0.04)

        title = (f"Received Equalized {modulation_order} APSK Constellation Diagram\n"
                 "(Trained on Best Channel Model)\n"
                 f"Received EVM = {evm:.1f}% | Received BER = {ber:.2e} | "
                 f"Spectral Efficiency (Nominal) = {spectral_efficiency:.2f} bit/s/Hz")
        fig.suptitle(title, fontsize=11)
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
        fig.savefig(run_dir / "plots" / "higher_order_constellation.png", dpi=130,
                    bbox_inches="tight")

    def _plot_noise_floor(self, run_dir, encoder, decoder, sent_syms, recv_syms, freqs):
        '''Contrast the empirical end-to-end residual EVM (from the validation trials)
        with the additive noise floor. The noise floor is estimated by encoding one fixed
        symbol and replaying its AWG waveform through channel -> decoder N times: with the
        transmitted waveform identical, the per-carrier variance about the mean is pure
        additive noise, measured at the same point as the end-to-end EVM.'''
        if self.noise_floor_points < 2:
            raise ValueError("the noise floor is a variance over replays, so it needs at least "
                             f"2 points; NOISE_FLOOR_POINTS is {self.noise_floor_points}")
        modulate, send, measure, resample, *sync_blocks, demod = self.ofdm_blocks

        residual_power = np.mean(np.abs(sent_syms - recv_syms) ** 2, axis=0)
        signal_power = np.mean(np.abs(sent_syms) ** 2, axis=0)
        end_to_end_evm = np.sqrt(residual_power / (signal_power + 1e-12)) * 100

        encode_chain = Chain([modulate, ApplyEncoder(encoder, modulate, self.clip_value, self.device)])
        fixed = encode_chain.run(Signal(data=np.zeros(1), sampling_rate=modulate.fs_out))
        fixed_awg = np.asarray(fixed.artifact_container["awg_waveform"])
        fixed_preamble = np.asarray(fixed.artifact_container["preamble"])
        fixed_sent = np.asarray(fixed.artifact_container["sent_symbols"])

        replay_chain = Chain([send, measure, resample, *sync_blocks,
                              ApplyDecoder(decoder, demod, self.device), demod])
        repeated_received = []
        for _ in range(self.noise_floor_points):
            signal = Signal(data=np.zeros(1), sampling_rate=modulate.fs_out)
            signal.artifact_container["awg_waveform"] = fixed_awg
            signal.artifact_container["preamble"] = fixed_preamble
            out = replay_chain.run(signal)
            repeated_received.append(np.asarray(out.artifact_container["received_symbols"]))
        repeated_received = np.stack(repeated_received)
        self._store_noise_floor(run_dir.name, repeated_received, fixed_sent)

        noise_power = (np.abs(repeated_received - repeated_received.mean(axis=0)) ** 2
                       ).sum(axis=0) / (self.noise_floor_points - 1) # -1 for Bessel's correction
        reference_power = np.abs(fixed_sent) ** 2
        noise_floor_evm = np.sqrt(noise_power / (reference_power + 1e-12)) * 100

        frequencies_mhz = np.asarray(freqs) / 1e6
        end_to_end_mean = float(np.sqrt(np.mean(residual_power) / (np.mean(signal_power) + 1e-12)) * 100)
        noise_floor_mean = float(np.sqrt(np.mean(noise_power) / (np.mean(reference_power) + 1e-12)) * 100)

        fig = Figure(figsize=(6, 4))
        ax = fig.subplots()
        ax.plot(frequencies_mhz, end_to_end_evm, color="#0072B2", marker="o", markersize=3,
                linewidth=1.2, label=f"End-to-end residual (mean {end_to_end_mean:.1f}%)")
        ax.plot(frequencies_mhz, noise_floor_evm, color="#D55E00", marker="^", markersize=3,
                linewidth=1.2, linestyle="--",
                label=f"Estimated noise floor (mean {noise_floor_mean:.1f}%)")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("EVM (%)")
        ax.set_title("End-to-End Residual EVM% vs. Estimated EVM% Noise Floor")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=False)

        fig.tight_layout()
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
        fig.savefig(run_dir / "plots" / "noise_floor_evm.png", dpi=130, bbox_inches="tight")
        return noise_floor_mean

    def _symbol_constellation(self, time_symbol_with_cp, demod):
        '''FFT a [CP | payload] time-domain symbol down to the active-carrier constellation,
        matching how DemodulateDataOFDM recovers received_symbols.'''
        payload = np.asarray(time_symbol_with_cp)[demod.cyclic_prefix_length:]
        return np.fft.fft(payload, norm="ortho")[demod.subcarrier_indicies]

    def _plot_equalization_comparison(self, run_dir: Path, channel_form: str, ed_stages: dict[str, list[np.ndarray]],
                                      no_ed_sent: list[np.ndarray], no_ed_received: list[np.ndarray],
                                      freqs: np.ndarray) -> None:
        '''Saves the constellations to plots/equalization_constellations.npz and draws the
        figure from that file, so it can be restyled later with
        experiments/plot_equalization_comparison.py instead of rerunning the hardware.'''
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        constellation_path = plots_dir / "equalization_constellations.npz"
        save_equalization_constellations(constellation_path, channel_form, ed_stages,
                                         no_ed_sent, no_ed_received, freqs)
        plot_equalization_comparison(constellation_path, plots_dir / "equalization_comparison",
                                     EQUALIZATION_COMPARISON_FONTS)

    # ------------------------------------------------------------- artifacts
    def _store_waveforms(self, model_id, sent, received, channel_form, frac_delays=None,
                         encoder_drive=None):
        g = self.val_group[model_id] if model_id in self.val_group else self.val_group.create_group(model_id)
        arrays = [("sent_time", sent), ("received_time", received)]
        if frac_delays is not None:
            arrays.append(("fractional_delay_samples", frac_delays))
        if encoder_drive is not None:
            arrays.append(("encoder_drive", encoder_drive))
        for name, arr in arrays:
            za = g.create_array(name, shape=arr.shape, chunks=arr.shape, dtype=arr.dtype, overwrite=True)
            za[:] = arr
        g.attrs["channel_form"] = channel_form

    def _store_noise_floor(self, model_id, replays, reference):
        g = self.val_group[model_id]
        for name, arr in (("noise_floor_replays", replays), ("noise_floor_reference", reference)):
            za = g.create_array(name, shape=arr.shape, chunks=arr.shape, dtype=arr.dtype, overwrite=True)
            za[:] = arr

    def _plot_constellation(self, run_dir, sent, received, freqs, ed_run_id=None,
                            channel_form=None, channel_run_id=None, evm=None):
        sent = np.asarray(sent)
        received = np.asarray(received)
        freqs = np.asarray(freqs)
        n_rows = sent.shape[0] if sent.ndim > 1 else 1
        c = np.tile(freqs, n_rows)
        sent_flat, recv_flat = sent.ravel(), received.ravel()

        fig = Figure(figsize=(11, 5))
        ax_sent, ax_recv = fig.subplots(1, 2)
        ax_sent.scatter(sent_flat.real, sent_flat.imag, s=10, c=c, cmap="viridis")
        ax_sent.set_title("Sent")
        sc = ax_recv.scatter(recv_flat.real, recv_flat.imag, s=10, c=c, cmap="viridis")
        # overlay reference constellation symbols as red X's
        ax_recv.scatter(sent_flat.real, sent_flat.imag, s=30, marker="x", c="red", linewidth=1.5, alpha=0.7, label="Reference")
        ax_recv.set_title("Received" if evm is None else f"Received (EVM={evm:.2f}%)")
        ax_recv.legend(fontsize=8, loc="upper right")

        for ax in (ax_sent, ax_recv):
            ax.set_xlabel("In-Phase")
            ax.set_ylabel("Quadrature")
            ax.grid(True)
            ax.set_aspect("equal", "box")
        fig.colorbar(sc, ax=[ax_sent, ax_recv], label="Carrier Frequency (Hz)")
        channel = " ".join(p for p in (channel_form, channel_run_id) if p)
        suptitle = " | ".join(p for p in (ed_run_id, f"channel: {channel}" if channel else "") if p)
        if suptitle:
            fig.suptitle(suptitle)
        fig.savefig(run_dir / "plots" / "constellation.png", dpi=120)

    def _plot_encoder_power(self, run_dir, powers):
        '''Histogram of per-trial encoder output symbol power (same units as the
        data-collection POWER_MIN/POWER_MAX), to check the E/D operating point
        against the channel-model training power range.'''
        powers = np.asarray(powers)
        fig = Figure(figsize=(6, 4))
        ax = fig.subplots()
        ax.hist(powers, bins="auto", edgecolor="black", alpha=0.8)
        ax.set_xlabel("Encoder output symbol power (RMS², config POWER units)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{run_dir.name} - encoder output power\n"
                     f"mean power = {powers.mean():.3f} (std = {powers.std():.3f}, "
                     f"n = {len(powers)})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
        fig.savefig(run_dir / "plots" / "encoder_power.png", dpi=120)

    def _plot_waveform(self, run_dir, example, label):
        # The transmit-domain rows (encoder in/out, decoder output) share a fixed
        # +-clip_value scale (AWG units) so drive amplitude and reconstruction quality
        # are directly comparable: the decoder output should look like the encoder input.
        # The decoder input is the received scope waveform (a different, smaller scale) and
        # the residual is a small error signal, so both autoscale on their own axes.
        #
        # The red line marks the CP -> FFT-window boundary on the sliced [CP | payload]
        # decoder rows: samples left of it are the discarded cyclic prefix, samples right
        # of it are the payload that demod actually FFTs.
        TRANSMIT_DOMAIN = "fixed"
        OWN_SCALE = "auto"
        rows = [
            ("encoder input", example["encoder_input"], False, TRANSMIT_DOMAIN),
            ("encoder output", example["encoder_output"], False, TRANSMIT_DOMAIN),
            ("decoder input (received)", example["decoder_input"], True, OWN_SCALE),
            ("decoder output (recovered)", example["decoder_output"], True, TRANSMIT_DOMAIN),
            ("residual (sent - recovered)", example["residual"], False, OWN_SCALE),
        ]
        window = example.get("decoder_window")

        fig = Figure(figsize=(9, 11))
        axes = fig.subplots(len(rows), 1)
        for ax, (title, signal, mark, scale) in zip(axes, rows):
            ax.plot(signal, lw=1)

            if mark and window is not None:
                for bound in window:
                    ax.axvline(bound, color="red", lw=1.0, label="CP -> FFT window")
                ax.legend(fontsize=7, loc="upper right")

            if scale == TRANSMIT_DOMAIN:
                ax.set_ylim(-1.1 * self.clip_value, 1.1 * self.clip_value)
                ax.axhline(self.clip_value, color="grey", lw=0.8, ls=":")
                ax.axhline(-self.clip_value, color="grey", lw=0.8, ls=":")

            ax.set_ylabel(title, fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Sample index")
        fig.suptitle(label)
        fig.savefig(run_dir / "plots" / "waveform.png", dpi=120)


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def derive_channel_form(record) -> str:
    '''Human-readable channel model family for plot labels, from a channel-grid
    runs.jsonl record: "gmp", "prob TCN"/"nonprob TCN", "prob LRU"/"nonprob LRU".'''
    rec = record or {}
    model = rec.get("model")
    if model == "gmp":
        return "gmp"
    if model in ("tcn", "lru"):
        dist = rec.get("distribution", "none")
        prefix = "prob" if dist not in ("none", None) else "nonprob"
        return f"{prefix} {model.upper()}"
    return model or "unknown"


def select_encoder_decoders(ed_exp_dir, channel_exp_dir=None, run_ids=None):
    '''Build the EncoderDecoderValidation input from a finished E/D grid search.

    run_ids: manual list of E/D run_ids to validate; None selects every run.
    channel_exp_dir: the channel grid search dir, used to label each E/D model with
        its originating channel form (skipped -> "unknown").'''
    ed_exp_dir = Path(ed_exp_dir)
    rows = _read_jsonl(ed_exp_dir / "runs.jsonl")
    if run_ids:
        by_id = {r["run_id"]: r for r in rows}
        missing = [rid for rid in run_ids if rid not in by_id]
        if missing:
            raise ValueError(f"E/D run_ids not found in {ed_exp_dir}: {missing}")
        rows = [by_id[rid] for rid in run_ids]

    channel_map = {}
    if channel_exp_dir is not None:
        channel_map = {r["run_id"]: r for r in _read_jsonl(Path(channel_exp_dir) / "runs.jsonl")}

    selected = []
    for row in rows:
        rid = row["run_id"]
        ch_meta = channel_map.get(row.get("channel_run_id"), {})
        selected.append({
            "run_id": rid,
            "arch": {k: row[k] for k in ARCH_KEYS},
            "checkpoint": ed_exp_dir / "runs" / rid / "model.pt",
            "channel_run_id": row.get("channel_run_id"),
            "channel_form": derive_channel_form(ch_meta),
            "channel_receptive_field": ch_meta.get("receptive_field"),
            "channel_distribution": ch_meta.get("distribution", "none"),
        })
    return selected
