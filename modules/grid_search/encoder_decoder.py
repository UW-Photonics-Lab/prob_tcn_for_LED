'''
EncoderDecoderGridSearch: trains TCN encoder/decoder pairs end-to-end against a set
of frozen, already-trained channel models (see channel_model.select_channel_models)
and writes a resumable run folder in the same layout as ChannelModelGridSearch.

Each encoder/decoder grid point is paired with every supplied channel model, so
run_id encodes both the architecture params and the channel model's run_id.
'''
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from matplotlib.figure import Figure

from modules.constellation_diagram import get_constellation
from modules.experimental_blocks import band_limited_zc_preamble
from modules.grid_search.adapters import MODEL_REGISTRY
from modules.grid_search.base import GridSearchBase
from modules.grid_search.grid import expand_grid, resolve_runtime
from modules.models import TCN
from modules.utils import (calculate_BER, evm_pct, in_band_time_loss,
                           load_ofdm_dataset, symbols_to_time)

ARCH_KEYS = ("nlayers", "dilation_base", "kernel_size", "hidden_channels", "activation")


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
        self.eval_chunk_size = int(grid_config["eval_chunk_size"])
        self.clip_threshold = float(clip_threshold)
        self.preamble_length = preamble_length

        ed_points = expand_grid([{"model": "tcn_ae", "params": grid_config["params"]}])
        runs_per_arch = int(grid_config.get("runs_per_arch", 1))
        if runs_per_arch > 1 and not any("seed" in point["params"] for point in ed_points):
            ed_points = [{**point, "params": {**point["params"], "seed": seed}}
                         for point in ed_points for seed in range(runs_per_arch)]
        points = [{**p, "channel_run_id": run_id} for p in ed_points for run_id in self.channel_models]
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
            model = MODEL_REGISTRY[cm["model"]].load(
                cm["params"], cm["checkpoint"], self.device,
                shared={"EVAL_CHUNK_SIZE": self.eval_chunk_size}).model
            for p in model.parameters():
                p.requires_grad_(False)
            loaded[run_id] = model
        return ofdm_config, loaded

    def _sample_batch(self, batch_size, num_bits, ofdm_config):
        true_bits = np.random.randint(0, 2, size=(batch_size, num_bits))
        symbols = [self.constellation.bits_to_symbols("".join(map(str, bits))) for bits in true_bits]
        true_frame = torch.tensor(np.stack(symbols), dtype=torch.complex64, device=self.device)
        sent_time = symbols_to_time(true_frame, ofdm_config.num_leading_zeros, ofdm_config.num_trailing_zeros,
                                    negative_rail=-self.clip_threshold, positive_rail=self.clip_threshold)
        sent_time = torch.hstack((sent_time[:, -ofdm_config.cyclic_prefix_length:], sent_time))
        sent_time = torch.hstack((self.preamble.expand(batch_size, -1), sent_time))  # [preamble | CP | symbol]
        return torch.tensor(true_bits, device=self.device), sent_time

    def _forward(self, encoder, decoder, channel_model, sent_time, noise_scale=1.0, return_encoded=False):
        # encode, pass through the channel, and decode only the OFDM symbol (CP + payload);
        # the preamble is never processed
        pre = self.preamble_length
        preamble, symbol = sent_time[:, :pre], sent_time[:, pre:]
        encoded_symbol = encoder(symbol).clamp(-self.clip_threshold, self.clip_threshold)

        with torch.backends.cudnn.flags(enabled=False):
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
        return {"ber": ber, "evm_pct": evm_pct(sent_freq, recv_freq).item()}

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

        arch = {k: p[k] for k in ARCH_KEYS}
        encoder = TCN(**arch).to(self.device)
        decoder = TCN(**arch).to(self.device)
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
            recv_freq = self._decode_freq(encoder.eval(), decoder.eval(), channel_model, eval_sent_time, ofdm_config)
            evm = evm_pct(sent_freq, recv_freq).item()
        ch_model_type = f"{ch_meta.get('model', 'channel').upper()} {ch_meta.get('distribution', 'none')}"
        self._plot_constellation(run_dir, sent_freq, recv_freq, ofdm_config.subcarrier_freqs_hz,
                                 channel_id=point["channel_run_id"], channel_type=ch_model_type, evm=evm)
        return metrics

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

    def _plot_constellation(self, run_dir, sent, received, freqs, channel_id=None, channel_type=None, evm=None):
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
        ax_sent.set_title("Sent")
        sc = ax_recv.scatter(recv_np_flat.real, recv_np_flat.imag, s=10, c=c, cmap="viridis")
        # overlay reference constellation symbols as red X's
        ax_recv.scatter(sent_np_flat.real, sent_np_flat.imag, s=30, marker="x", c="red", linewidth=1.5, alpha=0.7, label="Reference")

        title = "Received"
        if channel_id or channel_type or evm is not None:
            parts = []
            if channel_id:
                parts.append(channel_id)
            if channel_type:
                parts.append(channel_type)
            if evm is not None:
                parts.append(f"EVM={evm:.2f}%")
            title = "Received (" + " | ".join(parts) + ")"
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
        fig.savefig(run_dir / "plots" / "constellation.png", dpi=120)
