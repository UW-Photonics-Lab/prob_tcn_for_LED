import torch
import os
import zarr
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import wandb
import torch.optim as optim
import torch.nn.functional as F
import sys
import time
import json
from modules.models import TCN_channel, TCN
import math

@dataclass
class OFDMConfig:
    '''OFDM frame geometry, named to match the PyFlux ModulateDataOFDM block.'''
    subcarrier_spacing: float
    baseband_fft_length: int
    cyclic_prefix_length: int
    active_carrier_indices: torch.Tensor
    subcarrier_freqs_hz: torch.Tensor
    num_leading_zeros: int
    num_trailing_zeros: int

    @property
    def baseband_ofdm_symbol_length(self) -> int:
        return self.baseband_fft_length + self.cyclic_prefix_length

    @classmethod
    def from_modulator(cls, mod) -> "OFDMConfig":
        '''Build directly from a live ModulateDataOFDM block.'''
        indices = torch.as_tensor(mod.subcarrier_indicies, dtype=torch.int)
        num_leading = len(mod.leading_zero_subcarriers)
        num_trailing = mod.baseband_fft_length // 2 - 1 - num_leading - mod.num_carriers
        return cls(
            subcarrier_spacing=mod.baseband_sampling_rate / mod.baseband_fft_length,
            baseband_fft_length=mod.baseband_fft_length,
            cyclic_prefix_length=mod.cyclic_prefix_length,
            active_carrier_indices=indices,
            subcarrier_freqs_hz=torch.as_tensor(mod.subcarrier_freqs_hz, dtype=torch.float32),
            num_leading_zeros=num_leading,
            num_trailing_zeros=num_trailing,
        )

    def to(self, device) -> "OFDMConfig":
        self.active_carrier_indices = self.active_carrier_indices.to(device)
        self.subcarrier_freqs_hz = self.subcarrier_freqs_hz.to(device)
        return self

    @property
    def baseband_sampling_rate(self) -> float:
        return self.baseband_fft_length * self.subcarrier_spacing

def symbols_to_time(X,
                    num_left_padding_zeros: int,
                    num_right_padding_zeros: int,
                    negative_rail=-3.0,
                    positive_rail=3.0):
        'Convert OFDM symbols to real valued signal'
        # Make hermetian symmetric
        Nt, Nf = X.shape
        device = X.device
        num_right_padding_zeros = torch.zeros(Nt, num_right_padding_zeros, device=device)
        num_left_padding_zeros = torch.zeros(Nt, num_left_padding_zeros, device=device)
        X = torch.cat([num_left_padding_zeros, X, num_right_padding_zeros], dim=-1)
        DC_Nyquist = torch.zeros((X.shape[0], 1), device=device)
        X_hermitian = torch.flip(X, dims=[1]).conj()
        X_full = torch.hstack([DC_Nyquist, X, DC_Nyquist, X_hermitian])

        # Convert to time domain
        x_time = torch.fft.ifft(X_full, dim=-1, norm="ortho").real
        x_time = torch.clip(x_time, min=negative_rail, max=positive_rail)
        return x_time.to(device)

def calculate_per_burst_rrmse_pct_loss(y, y_pred):
    '''each burst's error is normalized by that burst's own
    power before averaging, so every drive power counts equally'''
    r = y - y_pred
    per_burst = torch.mean(torch.abs(r) ** 2, dim=-1) / torch.mean(torch.abs(y) ** 2, dim=-1)
    return (torch.sqrt(per_burst).mean() * 100).item()

def load_ofdm_dataset(file_path, device):
    '''Read an AppendToDataset zarr group into (sent, received, OFDMConfig).'''

    root = zarr.open_group(file_path, mode="r")

    if "sent_burst" in root:
        sent = torch.tensor(root["sent_burst"][:], dtype=torch.float32, device=device)
        received = torch.tensor(root["received_burst"][:], dtype=torch.float32, device=device)
        preamble_length = int(root.attrs["preamble_length"])
        cp_length = int(root.attrs["cyclic_prefix_length"])
        fft_length = sent.shape[1] - preamble_length - cp_length
    else:
        print("[load_ofdm_dataset] WARNING: loading legacy format (symbol only, no preamble/CP)")
        sent = torch.tensor(root["sent_baseband"][:], dtype=torch.float32, device=device)
        received = torch.tensor(root["received_baseband"][:], dtype=torch.float32, device=device)
        fft_length = sent.shape[1]
        cp_length = int(root.attrs["cyclic_prefix_length"])

    indices = torch.tensor(root.attrs["active_carrier_indices"], dtype=torch.int)
    num_carriers = len(indices)
    num_leading = int(indices[0]) - 1  # leading zero subcarriers, DC excluded
    num_trailing = fft_length // 2 - 1 - num_leading - num_carriers
    spacing = root.attrs["f_min_hz"] / int(indices[0])

    config = OFDMConfig(
        subcarrier_spacing=spacing,
        baseband_fft_length=fft_length,
        cyclic_prefix_length=cp_length,
        active_carrier_indices=indices.to(device),
        subcarrier_freqs_hz=(indices.to(torch.float32) * spacing).to(device),
        num_leading_zeros=num_leading,
        num_trailing_zeros=num_trailing,
    )
    return sent, received, config

def in_band_time_loss(sent_time, decoded_time, ks_indices, n_fft):
    """In-band (per-carrier) loss, computed on the OFDM symbol only.
    """
    sent_fft = torch.fft.fft(sent_time[..., -n_fft:], norm="ortho", dim=-1)
    decoded_fft = torch.fft.fft(decoded_time[..., -n_fft:], norm="ortho", dim=-1)
    active = ks_indices.long()
    err = sent_fft[:, active] - decoded_fft[:, active]
    return torch.mean(err.real ** 2 + err.imag ** 2)

def calculate_BER(received_symbols, true_bits, constellation, return_decided_bits=False):
    # Demap symbols to bits
    constellation_symbols = torch.tensor(
        list(constellation._symbols_to_bits_map.keys()),
        dtype=received_symbols.dtype,
        device=received_symbols.device
    )
    distances = abs(received_symbols.reshape(-1, 1) - constellation_symbols.reshape(1, -1))

    closest_idx = distances.argmin(axis=1)
    constellation_symbols_list = list(constellation._symbols_to_bits_map.keys())
    decided_bits = [constellation._symbols_to_bits_map[constellation_symbols_list[idx]] for idx in closest_idx.cpu().numpy()]

    # Flatten decided bits into a 1D array
    decided_bits_flat = [int(bit) for symbol_bits in decided_bits for bit in symbol_bits]

    if torch.is_tensor(true_bits):
        true_bits = true_bits.detach().cpu()
    true_bits_array = np.array(true_bits)
    decided_bits_flat_array = np.array(decided_bits_flat)

    # Take minimum length to avoid shape mismatch
    min_len = min(len(true_bits_array), len(decided_bits_flat_array))
    true_bits_array = true_bits_array[:min_len]
    decided_bits_flat_array = decided_bits_flat_array[:min_len]

    # Calculate BER
    BER = float(np.sum(true_bits_array != decided_bits_flat_array) / len(true_bits_array))

    if return_decided_bits:
        return BER, decided_bits_flat
    return BER

def evm_pct(reference, measured):
    """Error Vector Magnitude (%) between reference and measured symbols.

    Array-generic: works for both NumPy arrays and torch tensors, and returns
    the same type as its inputs
    """
    signal_power = (abs(reference) ** 2).mean()
    residual_power = (abs(reference - measured) ** 2).mean()
    return (residual_power / (signal_power + 1e-12)) ** 0.5 * 100

def float_to_q88_int(x: float, data_width) -> int:
    """Convert float → 16-bit Q8.8 signed integer."""
    scale_factor = data_width // 2
    maximum = pow(2, data_width-1) - 1
    minimum = -1 * pow(2, data_width-1)
    scaled = int(round(x * pow(2, scale_factor)))
    return max(minimum, min(maximum, scaled))

def q88_int_to_hex(v: int, data_width) -> str:
    """Format a signed Q8.8 integer as 4-digit hex (two's complement), variable for other notations"""
    mask = (1 << data_width) - 1
    hex_digits = math.ceil(data_width / 4)
    return f"{v & mask:0{hex_digits}x}"

def q88_hex_to_float(hexa: str, data_width):
    sign_bit = 1 << (data_width - 1)
    v = int(hexa, 16)
    if v & sign_bit:  # If the sign bit is set (negative)
        v -= 1 << data_width
    return v / pow(2, data_width // 2)

def save_mem_file(path: str, values: list, comment: str = "", printDebug=True):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w") as f:
        if comment:
            f.write(f"// {comment}\n")
        for v in values:
            f.write(v + "\n")

    if printDebug:
        print(f"  Saved {len(values)} values → {path}")
