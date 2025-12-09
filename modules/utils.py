import torch
import os
import zarr
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Optional
import matplotlib.pyplot as plt
import wandb
import torch.optim as optim
import torch.nn.functional as F

@dataclass
class OFDM_channel:
    FREQUENCIES: Optional[torch.Tensor] = None
    NUM_POINTS_SYMBOL: Optional[int] = None
    CP_LENGTH: Optional[int] = None
    DELTA_K: Optional[int] = None
    KS: Optional[torch.Tensor] = None
    K_MIN: Optional[int] = None
    K_MAX: Optional[int] = None
    LEFT_PADDING_ZEROS: Optional[int] = None
    CP_RATIO: Optional[float] = None
    NUM_POINTS_FRAME: Optional[int] = None
    NUM_POS_FREQS: Optional[int] = None
    RIGHT_PADDING_ZEROS: Optional[int] = None
    sent_frames_freq: Optional[torch.Tensor] = None
    received_frames_freq: Optional[torch.Tensor] = None
    sent_frames_time: Optional[torch.Tensor] = None
    received_frames_time: Optional[torch.Tensor] = None

def symbols_to_time(X,
                    num_left_padding_zeros: int,
                    num_right_padding_zeros: int,
                    negative_rail=-3.0,
                    positive_rail=3.0):
        'Convert OFDM symbols to real valued signal'
        # Make hermetian symmetric
        Nt, Nf = X.shape
        device = X.device
        num_right_padding_zeros = torch.zeros(Nt, num_right_padding_zeros)
        num_left_padding_zeros = torch.zeros(Nt, num_left_padding_zeros)
        X = torch.cat([num_left_padding_zeros, X, num_right_padding_zeros], dim=-1)
        DC_Nyquist = torch.zeros((X.shape[0], 1))
        X_hermitian = torch.flip(X, dims=[1]).conj()
        X_full = torch.hstack([DC_Nyquist, X, DC_Nyquist, X_hermitian])

        # Convert to time domain
        x_time = torch.fft.ifft(X_full, dim=-1, norm="ortho").real
        x_time = torch.clip(x_time, min=negative_rail, max=positive_rail)
        return x_time.to(device)

def extract_zarr_data(file_path, device, delay=None):
    o_sets = OFDM_channel()
    cache_path = file_path.replace(".zarr", "_cached.pt").replace(".h5", "_cached.pt")
    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location=device)
        o_sets.sent_frames_time = data["sent_frames_time"].to(device)
        o_sets.received_frames_time = data["received_frames_time"].to(device)
        o_sets.FREQUENCIES = data["frequencies"].to(device)
        o_sets.NUM_POINTS_SYMBOL = data["NUM_POINTS_SYMBOL"]
        o_sets.CP_LENGTH = data["CP_LENGTH"]
        o_sets.DELTA_K = o_sets.FREQUENCIES[1] - o_sets.FREQUENCIES[0]
        o_sets.KS = (o_sets.FREQUENCIES / o_sets.DELTA_K).to(torch.int)
        o_sets.K_MIN = o_sets.KS[0]
        o_sets.K_MAX = o_sets.KS[-1]
        o_sets.LEFT_PADDING_ZEROS = o_sets.K_MIN - 1 # don't include DC freq
        o_sets.NUM_POS_FREQS = o_sets.K_MAX + 1
        o_sets.NUM_POINTS_FRAME = o_sets.NUM_POINTS_SYMBOL - o_sets.CP_LENGTH
        o_sets.RIGHT_PADDING_ZEROS = (o_sets.NUM_POINTS_FRAME  - 2 * o_sets.NUM_POS_FREQS) // 2
        print("Loaded from cache!")

    else:
        print("No cache found — loading original dataset...")
        root = zarr.open(file_path, mode="r")
        sent, received, received_time = [], [], []

        # Loop through frames
        num_skipped = 0
        for frame_key in root.group_keys():
            try:
                frame = root[frame_key]
                if o_sets.FREQUENCIES is None:
                    o_sets.FREQUENCIES = torch.tensor(frame["freqs"][:], dtype=torch.int).real
                    o_sets.NUM_POINTS_SYMBOL = int(frame.attrs["num_points_symbol"])
                    o_sets.CP_LENGTH = int(frame.attrs["cp_length"])
                else:
                    pass

                sent.append(torch.tensor(frame["sent"][:], dtype=torch.complex64))
                received.append(torch.tensor(frame["received"][:], dtype=torch.complex64))
                if "received_time" in frame:
                    received_time.append(torch.tensor(frame["received_time"][:], dtype=torch.float32))
            except:
                num_skipped += 1
                pass # skip corrupted frames
        print(f"Skipped {num_skipped} corrupted frames")

        o_sets.sent_frames_freq = torch.stack(sent).squeeze(1)
        o_sets.received_frames_freq = torch.stack(received).squeeze(1)
        o_sets.DELTA_K = o_sets.FREQUENCIES[1] - o_sets.FREQUENCIES[0]
        o_sets.KS = (o_sets.FREQUENCIES / o_sets.DELTA_K).to(torch.int)
        o_sets.K_MIN = o_sets.KS[0]
        o_sets.K_MAX = o_sets.KS[-1]
        o_sets.LEFT_PADDING_ZEROS = o_sets.K_MIN - 1 # don't include DC freq
        o_sets.NUM_POS_FREQS = o_sets.K_MAX + 1
        o_sets.NUM_POINTS_FRAME = o_sets.NUM_POINTS_SYMBOL - o_sets.CP_LENGTH
        o_sets.RIGHT_PADDING_ZEROS = (o_sets.NUM_POINTS_FRAME  - 2 * o_sets.NUM_POS_FREQS) // 2

        # Handle received time symbols; perform some cleaning if necessary
        N_shortest = min(t.size(-1) for t in received_time)
        good_indices = [i for i, x in enumerate(received_time) if x.size(-1) == N_shortest]
        received_frames_time = torch.stack([t for t in received_time if t.size(-1) == N_shortest],dim=0).real
        sent_frames_freq = o_sets.sent_frames_freq[good_indices]
        o_sets.received_frames_time = received_frames_time.squeeze(1)
        sent_frames_time = symbols_to_time(sent_frames_freq,
                                           o_sets.LEFT_PADDING_ZEROS,
                                           o_sets.RIGHT_PADDING_ZEROS)
        sent_frames_time = torch.hstack((sent_frames_time[:, -o_sets.CP_LENGTH:], sent_frames_time))
        o_sets.sent_frames_time = sent_frames_time

        cache_path = file_path.replace(".zarr", "_cached.pt").replace(".h5", "_cached.pt")
        torch.save({
            "sent_frames_time": o_sets.sent_frames_time.cpu(),
            "received_frames_time": o_sets.received_frames_time.cpu(),
            "frequencies": o_sets.FREQUENCIES.cpu(),
            "NUM_POINTS_SYMBOL": o_sets.NUM_POINTS_SYMBOL,
            "CP_LENGTH": o_sets.CP_LENGTH
        }, cache_path)

    return o_sets

class ChannelData(Dataset):
    def __init__(self,
                sent_frames,
                received_frames,
                frequencies,
                transform=None,
                target_transform=None):

        self.sent_frames = sent_frames
        self.received_frames = received_frames
        assert len(sent_frames) == len(received_frames)

    def __len__(self):
        return len(self.sent_frames)

    def __getitem__(self, idx):
        return self.sent_frames[idx], self.received_frames[idx]

def log_snr_plots(y_preds, noisy_y_preds, ofdm_settings):
    '''
    If model predicts mean of y and noisy y simultaneously,
    this logs signal and noise power plots
    '''
    noise_pred = noisy_y_preds - y_preds
    noise_power_pred_k = torch.fft.fft(noise_pred[:, ofdm_settings.CP_LENGTH:], norm='ortho', dim=-1).abs().square().mean(dim=0)
    signal_power_model = torch.fft.fft(y_preds[:, ofdm_settings.CP_LENGTH:], norm='ortho', dim=-1).abs().square().mean(dim=0)
    snr_k_model = (signal_power_model / (noise_power_pred_k + 1e-8))
    sample_rate = ofdm_settings.DELTA_K * ofdm_settings.NUM_POINTS_FRAME
    snr_mag_model = 10 * torch.log10(torch.abs(snr_k_model) + 1e-8)
    freqs = torch.fft.fftfreq(len(snr_mag_model), d=1/sample_rate)
    half = len(freqs)//2
    freqs = freqs[:half]
    snr_mag_model = snr_mag_model[:half]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs, snr_mag_model.cpu(), lw=1.5, color="orange")
    ax.set_title("SNR vs Frequency (Model)", fontsize=11)
    ax.set_xlabel("Frequency", fontsize=9)
    ax.set_ylabel("SNR Magnitude (dB)", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)
    wandb.log({"SNR_Frequency": wandb.Image(fig)})
    plt.close(fig)


def make_optimizer(mode, channel_model, noise_model, config):
    if mode == "channel_only":
        return optim.AdamW(
            list(channel_model.parameters()),
            lr=float(config.lr_channel),
            weight_decay=float(config.wd_channel)
        )

    elif mode == "noise_only":
        return optim.AdamW(
            list(noise_model.parameters()),
            lr=float(config.lr_noise),
            weight_decay=float(config.wd_noise)
        )

    elif mode == "joint":
        return optim.AdamW(
            list(channel_model.parameters()) +
            list(noise_model.parameters()),
            lr=float(config.lr_joint),
            weight_decay=float(config.wd_joint)
        )
    else:
        raise ValueError("Unknown mode")


def make_time_validate_plots(enc_in, enc_out, dec_in, dec_out,
                             frame_BER, run_model, step=0, zoom_samples=200):

    # Convert to numpy
    enc_in = enc_in.detach().cpu().numpy().flatten()
    enc_out = enc_out.detach().cpu().numpy().flatten()
    dec_in = dec_in.detach().cpu().numpy().flatten()
    dec_out = dec_out.detach().cpu().numpy().flatten()

    # Power and scaling
    enc_power_in = np.mean(enc_in**2)
    enc_power_out = np.mean(enc_out**2)
    enc_scale = enc_power_out / (enc_power_in + 1e-12)

    dec_power_in = np.mean(dec_in**2)
    dec_power_out = np.mean(dec_out**2)
    dec_scale = dec_power_out / (dec_power_in + 1e-12)

    # MSEs
    mse_encoder = np.mean((enc_in - enc_out) ** 2)
    mse_decoder = np.mean((dec_in - dec_out) ** 2)
    mse_total = np.mean((enc_in - dec_out) ** 2)

    # Log scalars
    prefix = "time_"
    wandb.log({f"{prefix}mse_loss": mse_total}, step=step)
    wandb.log({f"{prefix}frame_BER": frame_BER}, step=step)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    time_points = np.arange(zoom_samples)

    axes[0].plot(time_points, enc_in[:zoom_samples], 'r', alpha=0.5, label='Encoder Input')
    axes[0].plot(time_points, enc_out[:zoom_samples], 'b', alpha=0.8, label='Encoder Output')
    axes[0].set_title(
        f"Encoder Comparison (MSE: {mse_encoder:.2e}) | "
        f"In {enc_power_in:.3f} | Out {enc_power_out:.3f} | Scale {enc_scale:.3f}"
    )
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(time_points, dec_in[:zoom_samples], 'r', alpha=0.5, label='Decoder Input')
    axes[1].plot(time_points, dec_out[:zoom_samples], 'b', alpha=0.8, label='Decoder Output')
    axes[1].set_title(
        f"Decoder Comparison (MSE: {mse_decoder:.2e}) | "
        f"In {dec_power_in:.3f} | Out {dec_power_out:.3f} | Scale {dec_scale:.3f}"
    )
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(time_points, enc_in[:zoom_samples], 'r', alpha=0.5, label='Original Input')
    axes[2].plot(time_points, dec_out[:zoom_samples], 'b', alpha=0.8, label='Final Output')
    axes[2].set_title(
        f"End-to-End Comparison ({'Trained' if run_model else 'Untrained'})\n"
        f"MSE: {mse_total:.2e}, BER: {frame_BER:.2f}"
    )
    axes[2].legend(); axes[2].grid(True)

    fig.tight_layout()
    wandb.log({f"{prefix}time_signals": wandb.Image(fig)}, step=step)
    plt.close(fig)


def in_band_filter(x, ks_indices, nfft):
    mask = torch.zeros(nfft, device=x.device)
    neg_ks_indices = nfft - ks_indices
    mask[ks_indices] = 1.0
    mask[neg_ks_indices] = 1.0

    impulse_response = torch.fft.ifftshift(torch.fft.ifft(mask).real)
    h = impulse_response.view(1, 1, -1)
    filtered_x = F.conv1d(x.unsqueeze(1), h, padding='same').squeeze(1)
    return filtered_x


def in_band_time_loss(sent_time, decoded_time, ks_indices, n_fft, num_taps):
    """Compute in-band loss directly in time domain using filtering"""
    # Create frequency mask
    mask = torch.zeros(n_fft, DEVICE=sent_time.DEVICE)
    neg_ks_indices = n_fft - ks_indices
    mask[ks_indices] = 1.0
    mask[neg_ks_indices] = 1.0

    # Convert to time-domain filter (this is differentiable)
    impulse_response = torch.fft.ifftshift(torch.fft.ifft(mask).real)
    h = impulse_response.view(1, 1, -1)

    # Filter both signals
    sent_filtered = F.conv1d(sent_time.unsqueeze(1), h, padding='same').squeeze(1)
    decoded_filtered = F.conv1d(decoded_time.unsqueeze(1), h, padding='same').squeeze(1)

    # Compute MSE on filtered signals (equivalent to in-band frequency loss)
    loss = torch.mean((sent_filtered[:, num_taps:] - decoded_filtered[:, num_taps:]).pow(2))
    return loss

def calculate_BER(received_symbols, true_bits, constellation):
    # Demap symbols to bits
    constellation_symbols = torch.tensor(
        list(constellation._symbols_to_bits_map.keys()),
        dtype=received_symbols.dtype,
        DEVICE=received_symbols.DEVICE
    )
    distances = abs(received_symbols.reshape(-1, 1) - constellation_symbols.reshape(1, -1))

    closest_idx = distances.argmin(axis=1)
    constellation_symbols_list = list(constellation._symbols_to_bits_map.keys())
    decided_bits = [constellation._symbols_to_bits_map[constellation_symbols_list[idx]] for idx in closest_idx.cpu().numpy()]

    # Flatten decided bits into a 1D array
    decided_bits_flat = [int(bit) for symbol_bits in decided_bits for bit in symbol_bits]

    # Convert to NumPy arrays for comparison
    true_bits_array = np.array(true_bits)
    decided_bits_flat_array = np.array(decided_bits_flat)

    # Take minimum length to avoid shape mismatch
    min_len = min(len(true_bits_array), len(decided_bits_flat_array))
    true_bits_array = true_bits_array[:min_len]
    decided_bits_flat_array = decided_bits_flat_array[:min_len]

    # Calculate BER
    BER = float(np.sum(true_bits_array != decided_bits_flat_array) / len(true_bits_array))
    return BER

def evm_loss(true_symbols, predicted_symbols):
    return torch.mean(torch.abs(true_symbols - predicted_symbols) ** 2)

