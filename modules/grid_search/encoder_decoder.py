'''
EncoderDecoderGridSearch: trains TCN encoder/decoder pairs end-to-end against a set
of frozen, already-trained channel models (see channel_model.select_channel_models)
and writes a resumable run folder in the same layout as ChannelModelGridSearch.

Each encoder/decoder grid point is paired with every supplied channel model, so
run_id encodes both the architecture params and the channel model's run_id.
'''
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import yaml
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from modules.constellation_diagram import get_constellation
from modules.experimental_blocks import band_limited_zc_preamble
from modules.grid_search.adapters import MODEL_REGISTRY
from modules.grid_search.base import GridSearchBase
from modules.grid_search.grid import expand_grid, resolve_runtime
from modules.models import TCN, QxxTCN, AdderTCN#, ShiftTCN
from modules.utils import (calculate_BER, calculate_per_burst_rrmse_pct_loss, evm_pct, in_band_time_loss,
                           load_ofdm_dataset, symbols_to_time)

ARCH_KEYS = ("nlayers", "dilation_base", "kernel_size", "hidden_channels", "activation")

ED_MODELS = {
    "tcn_ae": TCN,
    "Qxx_tcn": QxxTCN,
    "adder_tcn": AdderTCN,
    #"shift_tcn": ShiftTCN,
}


class EncoderDecoderGridSearch(GridSearchBase):
    def __init__(
                 self,
                 grid_config: dict,
                 channel_models: list,
                 dataset_path,
                 experiments_dir=None,
                 device="cpu",
                 seed=0,
                 experiment_name="encoder_decoder",
                 preamble_length=256,
                 clip_threshold=3.0,
                 run_prefix=None,
                 ):

        self.dataset_path = Path(dataset_path)
        self.channel_models = {cm["run_id"]: cm for cm in channel_models}
        self.constellation = get_constellation(grid_config["constellation"])
        self.preamble_amplitude = float(grid_config["preamble_amplitude"])
        self.clip_threshold = float(clip_threshold)
        self.preamble_length = preamble_length
        self.mix = grid_config["Mix-Match_Archs"]

        ed_points = expand_grid(grid_config["models"])
        points = [{**p, "channel_run_id": run_id} for p in ed_points for run_id in self.channel_models] if not self.mix else [{**p, "decoder": {**d},  "channel_run_id": run_id} for p in ed_points for d in ed_points for run_id in self.channel_models]

        shared_params = {k: v for k, v in grid_config.items() if k != "params"}
        super().__init__(points, grid_config, shared_params, experiments_dir, device, seed,
                          experiment_name, run_prefix=run_prefix, extra_manifest={
                              "dataset_path": str(self.dataset_path),
                              "channel_models": list(self.channel_models),
                          })
        self.rank_by = "evm_pct"

    @classmethod
    def from_experiment_config(cls,
                               config_path,
                               channel_models,
                               device=None,
                               seed=None,
                               experiment_name="encoder_decoder",
                               experiments_dir=None,
                               dataset_path=None,
                               run_prefix=None,
                               ):
        '''
        Build the E/D grid from the ENCODER_DECODER section of the config,
        paired against an already-selected list of channel_models (see
        channel_model.select_channel_models).

        dataset_path overrides DATA_COLLECTION.DATASET_PATH from the config file;
        use this when the dataset was just created and the YAML still shows null.
        '''
        with open(Path(config_path), encoding="utf-8") as f:
            full = yaml.safe_load(f)
        grid_config = {k.lower(): v for k, v in full["ENCODER_DECODER"].items()}
        if dataset_path is None:
            dataset_path = full["DATA_COLLECTION"]["DATASET_PATH"]
        device, seed = resolve_runtime(full, device, seed)
        return cls(grid_config, channel_models=channel_models, dataset_path=dataset_path,
                   device=device, seed=seed, experiment_name=experiment_name,
                   experiments_dir=experiments_dir,
                   preamble_length=int(full["DATA_COLLECTION"]["PREAMBLE_LENGTH"]),
                   clip_threshold=float(full["DATA_COLLECTION"]["CLIP_THRESHOLD"]),
                   run_prefix=run_prefix)

    def _prepare(self, ofdm_config=None):
        if ofdm_config is None:
            _, _, ofdm_config = load_ofdm_dataset(str(self.dataset_path), self.device)
        # band-limited ZC preamble (identical to ModulateDataOFDM) prepended to every
        # training burst so the encoder/decoder learn to preserve it for hardware sync
        fs = ofdm_config.baseband_fft_length * ofdm_config.subcarrier_spacing
        freqs = ofdm_config.subcarrier_freqs_hz
        preamble = band_limited_zc_preamble(self.preamble_length, fs,
                                            float(freqs.min()), float(freqs.max()), self.preamble_amplitude)
        self.preamble = torch.tensor(preamble, dtype=torch.float32, device=self.device).unsqueeze(0)
        loaded = {}
        for run_id, cm in self.channel_models.items():
            model = MODEL_REGISTRY[cm["model"]].load(cm["params"], cm["checkpoint"], self.device).model
            for p in model.parameters():
                p.requires_grad_(False)
            loaded[run_id] = model
        return ofdm_config, loaded

    def _sample_batch(self, batch_size, num_bits, ofdm_config, preamble_dne=None):
        true_bits = np.random.randint(0, 2, size=(batch_size, num_bits))
        symbols = [self.constellation.bits_to_symbols("".join(map(str, bits))) for bits in true_bits]
        true_frame = torch.tensor(np.stack(symbols), dtype=torch.complex64, device=self.device)
        sent_time = symbols_to_time(true_frame, ofdm_config.num_leading_zeros, ofdm_config.num_trailing_zeros,
                                    negative_rail=-self.clip_threshold, positive_rail=self.clip_threshold)
        sent_time = torch.hstack((sent_time[:, -ofdm_config.cyclic_prefix_length:], sent_time))
        if preamble_dne:
            fs = ofdm_config.baseband_fft_length * ofdm_config.subcarrier_spacing
            freqs = ofdm_config.subcarrier_freqs_hz
            preamble = band_limited_zc_preamble(self.preamble_length, fs,
                                                float(freqs.min()), float(freqs.max()), self.preamble_amplitude)
            self.preamble = torch.tensor(preamble, dtype=torch.float32, device=self.device).unsqueeze(0)
        sent_time = torch.hstack((self.preamble.expand(batch_size, -1), sent_time))  # [preamble | CP | symbol]
        return torch.tensor(true_bits, device=self.device), sent_time

    def _forward(self, encoder, decoder, channel_model, sent_time, noise_scale=1.0, return_encoded=False):
        # encode, pass through the channel, and decode only the OFDM symbol (CP + payload);
        # the preamble is never processed
        pre = self.preamble_length
        preamble, symbol = sent_time[:, :pre], sent_time[:, pre:]
        encoded_symbol = encoder(symbol).clamp(-self.clip_threshold, self.clip_threshold)
        channel_out = channel_model(encoded_symbol)
        if isinstance(channel_out, tuple):
            # probabilistic channel: scale the sampled noise realization around the mean
            # so training noise can be annealed (noise_scale 1 = full noise, 0 = mean only)
            noisy, mean = channel_out[0], channel_out[1]
            received_symbol = mean + noise_scale * (noisy - mean)
        else:
            received_symbol = channel_out
        decoded_symbol = decoder(received_symbol)
        decoded_time = torch.hstack((preamble, decoded_symbol))
        if return_encoded:
            return decoded_time, encoded_symbol
        return decoded_time

    def _frame_to_freq(self, time_frame, ofdm_config):
        '''
        Strip the preamble + CP, then FFT the OFDM symbol down to the symbols carried
        on the active subcarriers
        '''
        start = self.preamble_length + ofdm_config.cyclic_prefix_length
        symbol = time_frame[:, start:start + ofdm_config.baseband_fft_length]
        return torch.fft.fft(symbol, norm="ortho", dim=-1)[:, ofdm_config.active_carrier_indices]

    def _decode_freq(self, encoder, decoder, channel_model, sent_time, ofdm_config):
        '''
        Run a frame through encoder->channel->decoder and recover the
        frequency-domain symbols on the active carriers.
        '''
        return self._frame_to_freq(self._forward(encoder, decoder, channel_model, sent_time), ofdm_config)

    def _test_ber(self, encoder, decoder, channel_model, ofdm_config, true_bits, sent_time) -> float:
        '''Bit-error rate of the encoder/decoder pair on a fixed evaluation batch.'''
        was_training = encoder.training
        encoder.eval(); decoder.eval()
        with torch.no_grad():
            decoded_freq = self._decode_freq(encoder, decoder, channel_model, sent_time, ofdm_config)
        ber = calculate_BER(decoded_freq.flatten(), true_bits.flatten(), constellation=self.constellation)
        if was_training:
            encoder.train(); decoder.train()
        return ber

    def _evaluate(self, encoder, decoder, channel_model, ofdm_config, num_bits, batch_size) -> dict:
        '''
        BER and symbol rRMSE on a fresh held-out batch (both on the same data).
        '''
        true_bits, sent_time = self._sample_batch(batch_size, num_bits, ofdm_config)
        was_training = encoder.training
        encoder.eval(); decoder.eval()
        with torch.no_grad():
            sent_freq = self._frame_to_freq(sent_time, ofdm_config)
            recv_freq = self._decode_freq(encoder, decoder, channel_model, sent_time, ofdm_config)
        if was_training:
            encoder.train(); decoder.train()
        ber = calculate_BER(recv_freq.flatten(), true_bits.flatten(), constellation=self.constellation)
        return {"ber": ber, "rrmse_pct": calculate_per_burst_rrmse_pct_loss(sent_freq, recv_freq)}

    def _run_point(self, point, run_dir, context) -> dict:
        ofdm_config, channel_models = context
        channel_model = channel_models[point["channel_run_id"]]
        p = point["params"]

        # an optional per-point seed lets a grid train several E/Ds that differ only by
        # initialization/sampling (e.g. the controlled-CDF experiment sweeps seeds)
        if "seed" in p:
            seed = int(p["seed"])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        #arch = {k: p[k] for k in ARCH_KEYS}
        encoder = ED_MODELS[point["model"]](**p).to(self.device)
        decoder = ED_MODELS[point["model"]](**p).to(self.device) if not self.mix else ED_MODELS[point["model"]](**point["decoder"]["params"]).to(self.device)
        optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()),
                                 lr=float(p["lr"]),
                                 weight_decay=float(p.get("weight_decay", 0.0)))

        scheduler = None
        if "patience" in p:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min",
                factor=float(p.get("factor", 0.5)),
                patience=int(p["patience"]),
                min_lr=float(p.get("min_lr", 1e-6)),
            )
        elif "cosineAnnealing_min_lr" in p:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(p["epochs"]),
                eta_min=float(p.get("cosineAnnealing_min_lr", 1e-6)),
            )

        num_bits = len(ofdm_config.active_carrier_indices) * self.constellation.bits_per_symbol
        batch_size = p["batch_size"]

        # channel-noise annealing: full noise until noise_anneal_start * epochs, then a
        # linear decay reaching 0 at the final epoch. 
        # 
        # 1.0 disables annealing and reproduces prior behavior.
        epochs = int(p["epochs"])
        noise_anneal_start = float(p.get("noise_anneal_start", 1.0))
        anneal_start_epoch = int(round(noise_anneal_start * epochs))
        annealing_active = noise_anneal_start < 1.0
        deterministic_channel = bool(p.get("deterministic_channel", False))

        # plain drive power
        drive_mean_power_weight = float(p.get("drive_mean_power_weight", 0.0))
        if drive_mean_power_weight < 0.0:
            raise ValueError(f"drive_mean_power_weight must be >= 0, got {drive_mean_power_weight}")
        use_drive_mean_power = drive_mean_power_weight > 0.0

        kurtosis_weight = float(p.get("kurtosis_weight", 0.0))
        if kurtosis_weight < 0.0:
            raise ValueError(f"kurtosis_weight must be >= 0, got {kurtosis_weight}")
        use_drive_kurtosis = kurtosis_weight > 0.0
        if use_drive_kurtosis:
            kurtosis_target = float(p["kurtosis_target"])

        # fixed held-out batch so the BER-vs-epoch curve and the final
        # constellation plot are measured on consistent data across epochs
        eval_bits, eval_sent_time = self._sample_batch(batch_size, num_bits, ofdm_config)

        encoder.train()
        decoder.train()
        history = {"loss": [], "ber": [], "lr": [], "noise_scale": [], "drive_rms": []}
        if use_drive_mean_power:
            history["drive_power"] = []
        if use_drive_kurtosis:
            history["drive_kurtosis"] = []
        for epoch in range(epochs):
            if deterministic_channel:
                noise_scale = 0.0
            elif epoch < anneal_start_epoch:
                noise_scale = 1.0
            else:
                noise_scale = 1.0 - (epoch - anneal_start_epoch) / max(epochs - 1 - anneal_start_epoch, 1)
                noise_scale = max(0.0, noise_scale)

            _, sent_time = self._sample_batch(batch_size, num_bits, ofdm_config)
            decoded_time, encoded_symbol = self._forward(encoder, decoder, channel_model, sent_time,
                                                         noise_scale=noise_scale, return_encoded=True)
            # loss on the OFDM symbol only; the preamble is never encoded/decoded
            offset = self.preamble_length
            loss = in_band_time_loss(sent_time[:, offset:], decoded_time[:, offset:],
                                     ofdm_config.active_carrier_indices, ofdm_config.baseband_fft_length)

            if use_drive_mean_power:
                drive_power = encoded_symbol.pow(2).mean()
                loss = loss + drive_mean_power_weight * drive_power

            if use_drive_kurtosis:
                drive_kurtosis = encoded_symbol.pow(4).mean() / encoded_symbol.pow(2).mean() ** 2
                loss = loss + kurtosis_weight * (drive_kurtosis - kurtosis_target) ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None and (not annealing_active or epoch >= anneal_start_epoch):
                if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.step()
                else:
                    scheduler.step(loss.item())
            history["loss"].append(loss.item())
            history["ber"].append(self._test_ber(encoder, decoder, channel_model, ofdm_config, eval_bits, eval_sent_time))
            history["lr"].append(optimizer.param_groups[0]["lr"])
            history["noise_scale"].append(noise_scale)
            history["drive_rms"].append(encoded_symbol.detach().pow(2).mean().sqrt().item())
            if use_drive_mean_power:
                history["drive_power"].append(drive_power.item())
            if use_drive_kurtosis:
                history["drive_kurtosis"].append(drive_kurtosis.item())

        metrics = self._evaluate(encoder, decoder, channel_model, ofdm_config, num_bits, batch_size)
        metrics["num_params"] = encoder.get_num_params() + decoder.get_num_params()
        metrics["channel_run_id"] = point["channel_run_id"]
        metrics["name"] = point["model"]
        if self.mix:
            metrics["encoder"] = p
            metrics["decoder"] = point["decoder"]["params"]

        if point["model"] != "tcn_ae":
            metrics["bit_width"] = p.get("quantization").get("data_width")
            metrics["energy"] = encoder.energy + decoder.energy
        # In the future, add: Power/Energy metric, RM/BOP/NABS (From adderCNN paper)

        # propagate channel model metadata
        ch_meta = self.channel_models[point["channel_run_id"]]
        metrics["channel_receptive_field"] = ch_meta.get("receptive_field")
        metrics["channel_distribution"] = ch_meta.get("distribution", "none")

        with torch.no_grad():
            _, final_drive = self._forward(encoder.eval(), decoder.eval(), channel_model,
                                           eval_sent_time, noise_scale=0.0, return_encoded=True)
        drive_power = final_drive.pow(2).mean()
        metrics["drive_rms"] = drive_power.sqrt().item()
        metrics["drive_kurtosis"] = (final_drive.pow(4).mean() / drive_power ** 2).item()
        metrics["drive_papr"] = (final_drive.abs().max() ** 2 / drive_power).item()

        torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, run_dir / "model.pt")
        self._write_history(run_dir, history)
        self._plot_ber(run_dir, history["ber"])
        with torch.no_grad():
            sent_freq = self._frame_to_freq(eval_sent_time, ofdm_config)
            encoded = encoder(eval_sent_time)
            channel_out = channel_model(encoded)
            recv_freq = self._decode_freq(encoder.eval(), decoder.eval(), channel_model, eval_sent_time, ofdm_config)
            evm = evm_pct(sent_freq, recv_freq).item()
        ch_model_type = f"{ch_meta.get('model', 'channel').upper()} {ch_meta.get('distribution', 'none')}"
        self._plot_constellation(run_dir, sent_freq, recv_freq, ofdm_config.subcarrier_freqs_hz,
                                 channel_id=point["channel_run_id"], channel_type=ch_model_type, evm=evm)
        #self._plot_constellation(run_dir, sent_freq, self._frame_to_freq(encoder(eval_sent_time), ofdm_config=ofdm_config), ofdm_config.subcarrier_freqs_hz, rec_title="encoded")
        if isinstance(channel_out, tuple):
            # probabilistic channel: scale the sampled noise realization around the mean
            # so training noise can be annealed (noise_scale 1 = full noise, 0 = mean only)
            noisy, mean = channel_out[0], channel_out[1]
            received_symbol = mean + noise_scale * (noisy - mean)
        sv = [sent_freq, self._frame_to_freq(encoded, ofdm_config=ofdm_config), self._frame_to_freq(received_symbol, ofdm_config=ofdm_config), recv_freq]
        self._plot_constellation_enhanced_for_sv(run_dir, sv=sv,py=sv, freqs=ofdm_config.subcarrier_freqs_hz, evm_py=evm)
        return metrics

    def run(self, **prepare_kwargs):
        super().run(**prepare_kwargs)
        self._plot_evm_vs_bitwidth(self.summary_dir)
        self._plot_evm_vs_energy(self.summary_dir)
        return self.exp_dir

    # ------------------------------------------------------------------- plots
    def _plot_ber(self, run_dir, ber_curve):
        '''BER measured on the fixed held-out batch after each training epoch.'''
        fig = Figure(figsize=(7, 4))
        ax = fig.subplots()
        ax.plot(range(len(ber_curve)), ber_curve, marker=".", ms=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("BER")
        ax.set_title(f"{run_dir.name} - BER vs epoch")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
        fig.savefig(run_dir / "plots" / "ber.png", dpi=120)

    def _plot_constellation(self, run_dir, sent, received, freqs, channel_id=None, channel_type=None, evm=None, sent_title="Sent", rec_title="Recieved"):
        '''Sent vs received QPSK symbols on the active carriers, coloured by
        carrier frequency. Received plot overlays sent symbols as red X markers for reference.'''
        sent_np = sent.detach().cpu().numpy()
        recv_np = received.detach().cpu().numpy()
        # one frequency value per symbol, tiled across the batch to match ravel order
        c = np.tile(freqs.detach().cpu().numpy(), sent_np.shape[0])
        sent_np_flat = sent_np.ravel()
        recv_np_flat = recv_np.ravel()

        fig = Figure(figsize=(11, 5))
        ax_sent, ax_recv = fig.subplots(1, 2)
        ax_sent.scatter(sent_np_flat.real, sent_np_flat.imag, s=10, c=c, cmap="viridis")
        ax_sent.set_title(sent_title)
        sc = ax_recv.scatter(recv_np_flat.real, recv_np_flat.imag, s=10, c=c, cmap="viridis")
        # overlay reference constellation symbols as red X's
        ax_recv.scatter(sent_np_flat.real, sent_np_flat.imag, s=30, marker="x", c="red", linewidth=1.5, alpha=0.7, label="Reference")

        title = rec_title
        if channel_id or channel_type or evm is not None:
            parts = []
            if channel_id:
                parts.append(channel_id)
            if channel_type:
                parts.append(channel_type)
            if evm is not None:
                parts.append(f"EVM={evm:.2f}%")
            title = f"{rec_title} (" + " | ".join(parts) + ")"
        ax_recv.set_title(title)
        ax_recv.legend(fontsize=8, loc="upper right")

        for ax in (ax_sent, ax_recv):
            ax.set_xlabel("In-Phase")
            ax.set_ylabel("Quadrature")
            ax.grid(True)
            ax.set_aspect("equal", "box")
        fig.colorbar(sc, ax=[ax_sent, ax_recv], label="Carrier Frequency (Hz)")
        fig.suptitle(run_dir.name)
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ", "").replace(":", "-")
        fig.savefig(run_dir / "plots" / f"constellation_{sent_title}_{rec_title}_{current_time}.png", dpi=120)

    def _plot_constellation_enhanced_for_sv(self, run_dir, sv, py, freqs, channel_id=None, channel_type=None, evm_sv=None, evm_py=None):
        '''Sent vs received QPSK symbols on the active carriers, coloured by
        carrier frequency. Received plot overlays sent symbols as red X markers for reference.'''
        if len(sv) < 4 or len(py) < 4:
            raise ValueError("sv and py must each contain at least 4 frequency tensors: sent, encoded, channel output, decoded")

        fig, axes = plt.subplots(2, 8, figsize=(44, 10))
       # axes = fig.subplots(2, 8)

        def plot_stage(ax, main, freqs, title, reference=None):
            main_np = main.detach().cpu().numpy()
            c = np.tile(freqs.detach().cpu().numpy(), main_np.shape[0])
            main_flat = main_np.ravel()
            ax.scatter(main_flat.real, main_flat.imag, s=10, c=c, cmap="viridis")
            if reference != None:
                ref_np = reference.detach().cpu().numpy()
                ref_flat = ref_np.ravel()
                ax.scatter(ref_flat.real, ref_flat.imag, s=30, marker="x", c="red", linewidth=1.5, alpha=0.7, label="Reference")
                ax.legend(fontsize=8, loc="upper right")
            ax.set_title(title)
            ax.set_xlabel("In-Phase")
            ax.set_ylabel("Quadrature")
            ax.grid(True)
            ax.set_aspect("equal", "box")

        pairs = [
            ("Sent", "Encoded", 0, 1),
            ("Encoded", "Recieved", 1, 2),
            ("Recieved", "Decoded", 2, 3),
            ("Sent", "Decoded", 0, 3),
        ]
        rows = [(sv, "SystemVerilog", evm_sv), (py, "Pytorch", evm_py)]

        for row_idx, (tensor_list, label, row_evm) in enumerate(rows):
            for pair_idx, (left_name, right_name, left_idx, right_idx) in enumerate(pairs):
                left_ax = axes[row_idx, pair_idx * 2]
                right_ax = axes[row_idx, pair_idx * 2 + 1]
                plot_stage(left_ax, tensor_list[left_idx], freqs,
                           f"{left_name} ({label})")
                plot_stage(right_ax, tensor_list[right_idx], freqs,
                           f"{right_name} ({label}) ref: {left_name}", reference=tensor_list[left_idx])

            # Last plot in the row is the "Decoded" plot (pair_idx=3, right_ax) —
            # annotate its EVM just to the right of it.
            if row_evm is not None:
                last_ax = axes[row_idx, -1]
                last_ax.text(1.15, 0.5, f"EVM={row_evm:.2f}%",
                              transform=last_ax.transAxes,
                              fontsize=11, fontweight="bold",
                              ha="left", va="center",
                              rotation=0)

        fig.colorbar(axes[0, 0].collections[0], ax=axes.flatten().tolist(), label="Carrier Frequency (Hz)")

        title_parts = []
        if channel_id:
            title_parts.append(channel_id)
        if channel_type:
            title_parts.append(channel_type)
        if title_parts:
            fig.suptitle(f"{run_dir.name} — " + " | ".join(title_parts))
        else:
            fig.suptitle(run_dir.name)

        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ", "").replace(":", "-")
        fig.savefig(run_dir / "plots" / f"constellation_{current_time}.png", dpi=120)
        return fig

    # def _plot_evm_vs_bitwidth(self, run_dir, show_best_line=False):
    #     data = self.all_metrics
    #     bit_widths_all = [d.get("bit_width") for d in data if "bit_width" in d]
    #     rrmse_all = [d.get("rrmse_pct") for d in data if "bit_width" in d]

    #     if len(bit_widths_all) != len(rrmse_all):
    #         raise ValueError("bit_width and rrmse_pct must be present for every point")
    #     if len(bit_widths_all) == 0:
    #         return

    #     dot_color = "#2E86AB"
    #     line_color = "#E4572E"

    #     fig = Figure(figsize=(9, 6), dpi=150)
    #     ax = fig.add_subplot(111)
    #     ax.invert_xaxis()

    #     # Scatter every point (duplicates per bit width included)
    #     ax.scatter(bit_widths_all, rrmse_all, s=70, color=dot_color,
    #             edgecolor="white", linewidth=1.0, zorder=3,
    #             label="RRMSE (%)")

    #     # Compute min rrmse per bit width, for optional line + annotations
    #     best_by_bw = {}
    #     for bw, r in zip(bit_widths_all, rrmse_all):
    #         if bw not in best_by_bw or r < best_by_bw[bw]:
    #             best_by_bw[bw] = r

    #     sorted_bws = sorted(best_by_bw.keys(), reverse=True)
    #     best_rrmse = [best_by_bw[bw] for bw in sorted_bws]

    #     if show_best_line:
    #         ax.plot(sorted_bws, best_rrmse, linestyle="--", linewidth=1.5,
    #                 color=line_color, zorder=2, label="Best RRMSE per bit width")

    #     # Annotate only the min-per-bitwidth points
    #     for bw, r in zip(sorted_bws, best_rrmse):
    #         ax.annotate(f"{r:.2f}%", (bw, r), textcoords="offset points",
    #                     xytext=(0, 10), ha="center", fontsize=8.5, color=line_color)

    #     ax.set_xlabel("Bit Width", fontsize=12, fontweight="bold")
    #     ax.set_ylabel("RRMSE (%)", fontsize=12, fontweight="bold")
    #     ax.set_title("RRMSE vs. Bit Width", fontsize=14, fontweight="bold", pad=15)

    #     ax.set_xticks(sorted(set(bit_widths_all), reverse=True))
    #     ax.margins(y=0.15)
    #     ax.grid(True, alpha=0.3)
    #     ax.legend(fontsize=10, frameon=True, framealpha=0.9, loc="upper right")
    #     ax.tick_params(labelsize=10)

    #     for spine in ("top", "right"):
    #         ax.spines[spine].set_visible(False)

    #     fig.tight_layout()

    #     plots_dir = run_dir / "plots"
    #     plots_dir.mkdir(parents=True, exist_ok=True)
    #     out_path = plots_dir / "rrmse_vs_bitwidth.png"
    #     fig.savefig(out_path, dpi=120)
    #     #print(f"Saved plot to {out_path}")

    #     return fig, ax
    # def _plot_evm_vs_energy(self, run_dir, show_best_line=False):
    #     data = self.all_metrics
    #     energy_all = [d.get("energy") for d in data if "energy" in d]
    #     rrmse_all = [d.get("rrmse_pct") for d in data if "energy" in d]

    #     if len(energy_all) != len(rrmse_all):
    #         raise ValueError("bit_width and energy must be present for every point")
    #     if len(energy_all) == 0:
    #         return

    #     dot_color = "#2E86AB"
    #     line_color = "#E4572E"

    #     fig = Figure(figsize=(9, 6), dpi=150)
    #     ax = fig.add_subplot(111)
    #     ax.invert_xaxis()

    #     # Scatter every point (duplicates per bit width included)
    #     ax.scatter(energy_all, rrmse_all, s=70, color=dot_color,
    #             marker="x", linewidth=1.0, zorder=3,
    #             label="RRMSE (%)")

    #     # Compute min rrmse per bit width, for optional line + annotations
    #     best_by_bw = {}
    #     for bw, r in zip(energy_all, rrmse_all):
    #         if bw not in best_by_bw or r < best_by_bw[bw]:
    #             best_by_bw[bw] = r

    #     sorted_bws = sorted(best_by_bw.keys(), reverse=True)
    #     best_rrmse = [best_by_bw[bw] for bw in sorted_bws]

    #     if show_best_line:
    #         ax.plot(sorted_bws, best_rrmse, linestyle="--", linewidth=1.5,
    #                 color=line_color, zorder=2, label="Best RRMSE per energy")

    #     # Annotate only the min-per-bitwidth points
    #     for bw, r in zip(sorted_bws, best_rrmse):
    #         ax.annotate(f"{r:.2f}%", (bw, r), textcoords="offset points",
    #                     xytext=(0, 10), ha="center", fontsize=8.5, color=line_color)

    #     ax.set_xlabel("Energy (J)", fontsize=12, fontweight="bold")
    #     ax.set_ylabel("RRMSE (%)", fontsize=12, fontweight="bold")
    #     ax.set_title("RRMSE vs. Energy (J)", fontsize=14, fontweight="bold", pad=15)

    #     ax.set_xticks(sorted(set(energy_all), reverse=True))
    #     ax.margins(y=0.15)
    #     ax.grid(True, alpha=0.3)
    #     ax.legend(fontsize=10, frameon=True, framealpha=0.9, loc="upper right")
    #     ax.tick_params(labelsize=10)

    #     for spine in ("top", "right"):
    #         ax.spines[spine].set_visible(False)

    #     fig.tight_layout()

    #     plots_dir = run_dir / "plots"
    #     plots_dir.mkdir(parents=True, exist_ok=True)
    #     out_path = plots_dir / "rrmse_vs_energy.png"
    #     fig.savefig(out_path, dpi=120)
    #     #print(f"Saved plot to {out_path}")

    #     return fig, ax
    def _get_name_color_map(self, names):
        unique_names = sorted(list(set(names)))
        num_names = len(unique_names)
        if num_names <= 10:
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i) for i in range(num_names)]
        else:
            cmap = plt.get_cmap("gist_rainbow")
            colors = [cmap(i / num_names) for i in range(num_names)]
        return dict(zip(unique_names, colors))
    
    def _plot_evm_vs_bitwidth(self, run_dir, show_best_line=False):
        data = [d for d in self.all_metrics if "bit_width" in d and "rrmse_pct" in d]
        if not data:
            return None

        line_color = "#E4572E"
        names = [d.get("name", "Unknown") for d in data]
        color_map = self._get_name_color_map(names)

        fig = Figure(figsize=(9, 6), dpi=150)
        ax = fig.add_subplot(111)

        # Plot each point grouped by metric 'name' to generate distinct colors & legend entries
        plotted_names = set()
        for d in data:
            bw = d["bit_width"]
            rrmse = d["rrmse_pct"]
            name = d.get("name", "Unknown")
            lbl = name if name not in plotted_names else None
            plotted_names.add(name)

            ax.scatter(bw, rrmse, s=70, color=color_map[name],
                       marker="x", linewidth=1.0, zorder=3, label=lbl)

        # Compute min rrmse per bit width
        best_by_bw = {}
        for d in data:
            bw = d["bit_width"]
            r = d["rrmse_pct"]
            if bw not in best_by_bw or r < best_by_bw[bw]:
                best_by_bw[bw] = r

        sorted_bws = sorted(best_by_bw.keys(), reverse=True)
        best_rrmse = [best_by_bw[bw] for bw in sorted_bws]

        if show_best_line:
            ax.plot(sorted_bws, best_rrmse, linestyle="--", linewidth=1.5,
                    color=line_color, zorder=2, label="Best RRMSE per bit width")

        # Annotate min-per-bitwidth points
        for bw, r in zip(sorted_bws, best_rrmse):
            ax.annotate(f"{r:.2f}%", (bw, r), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8.5, color=line_color)

        ax.set_xlabel("Bit Width", fontsize=12, fontweight="bold")
        ax.set_ylabel("RRMSE (%)", fontsize=12, fontweight="bold")
        ax.set_title("RRMSE vs. Bit Width", fontsize=14, fontweight="bold", pad=15)

        all_bws = [d["bit_width"] for d in data]
        ax.set_xticks(sorted(set(all_bws), reverse=True))
        ax.invert_xaxis()
        ax.margins(y=0.15)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, frameon=True, framealpha=0.9, loc="upper right")
        ax.tick_params(labelsize=10)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        fig.tight_layout()

        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_path = plots_dir / "rrmse_vs_bitwidth.png"
        fig.savefig(out_path, dpi=120)

        return fig, ax

    def _plot_evm_vs_energy(self, run_dir, show_best_line=False):
        data = [d for d in self.all_metrics if "energy" in d and "rrmse_pct" in d]
        if not data:
            return None

        line_color = "#E4572E"
        names = [d.get("name", "Unknown") for d in data]
        color_map = self._get_name_color_map(names)

        fig = Figure(figsize=(9, 6), dpi=150)
        ax = fig.add_subplot(111)

        # Plot each point grouped by metric 'name'
        plotted_names = set()
        for d in data:
            eng = d["energy"]
            rrmse = d["rrmse_pct"]
            name = d.get("name", "Unknown")
            lbl = name if name not in plotted_names else None
            plotted_names.add(name)

            ax.scatter(eng, rrmse, s=70, color=color_map[name],
                       marker="x", linewidth=1.5, zorder=3, label=lbl)

        # Compute min rrmse per energy level
        best_by_energy = {}
        for d in data:
            e = d["energy"]
            r = d["rrmse_pct"]
            if e not in best_by_energy or r < best_by_energy[e]:
                best_by_energy[e] = r

        sorted_energy = sorted(best_by_energy.keys())
        best_rrmse = [best_by_energy[e] for e in sorted_energy]

        if show_best_line:
            ax.plot(sorted_energy, best_rrmse, linestyle="--", linewidth=1.5,
                    color=line_color, zorder=2, label="Best RRMSE per energy")

        # Annotate min-per-energy points
        for e, r in zip(sorted_energy, best_rrmse):
            ax.annotate(f"{r:.2f}%", (e, r), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8.5, color=line_color)

        ax.set_xlabel("Energy (pJ per clk cycle)", fontsize=12, fontweight="bold")
        ax.set_ylabel("RRMSE (%)", fontsize=12, fontweight="bold")
        ax.set_title("RRMSE vs. Energy (pJ per clk cycle)", fontsize=14, fontweight="bold", pad=15)

        ax.margins(y=0.15)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, frameon=True, framealpha=0.9, loc="upper right")
        ax.tick_params(labelsize=10)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        fig.tight_layout()

        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_path = plots_dir / "rrmse_vs_energy.png"
        fig.savefig(out_path, dpi=120)

        return fig, ax