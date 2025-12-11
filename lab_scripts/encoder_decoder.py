
import sys
project_root = r"C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from training_state import STATE
from noisy_state import NOISY_STATE
# from lab_scripts.training_state import STATE
# from lab_scripts.noisy_state import NOISY_STATE
import h5py
import zarr
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import time
import pprint
from transformers import get_linear_schedule_with_warmup
import traceback
from lab_scripts.constellation_diagram import QPSK_Constellation, get_constellation, RingShapedConstellation
from modules.models import TCN
from modules.utils import evm_loss
import torch.nn.functional as F
from scipy.signal import resample_poly
import random
script_dir = os.path.dirname(os.path.abspath(__file__))
from lab_scripts.logging_code import *
decode_logger = setup_logger(log_file=r"C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled\debug_logs\test3.txt")

STATE['validate_model'] = False # Variable
STATE['normalize_power'] = False

load_model = False # Variable
LOAD_DIR = ""
if load_model:
    model_name = "cerulean-frog-8021" # Variable
    base_dir = r"C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled\models\pickled_models"
    LOAD_DIR = os.path.join(base_dir, model_name)
    with open(os.path.join(LOAD_DIR, "config.json"), "r") as f:
        hyperparams = json.load(f)

    print(f"WandB run info:")
    print(f"  Name: {wandb.run.name}")
    print(f"  ID: {wandb.run.id}")
    print(f"  URL: {wandb.run.url}")
    print("Chosen hyperparameters for this session:")

    # Start Weights and Biases session
    wandb.init(project="mldrivenpeled",
            config=hyperparams)
    config = wandb.config

if load_model and STATE['validate_model']:
    wandb.run.tags = list(wandb.run.tags) + ["validate"]
    if wandb.run.notes == None:
        wandb.run.notes = ""
    wandb.run.notes += wandb.run.notes + f"\n Validate model {model_name}"


# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps") # for M chip Macs
else:
    device = torch.device("cpu")


STATE['loss_accumulator'] = []
STATE['predicted_received_symbols'] = []

STATE['encoder_out_buffer'] = []
STATE['decoder_in_buffer'] = []

if load_model:
    with open(os.path.join(LOAD_DIR, "config.json"), "r") as f:
        remote_config = json.load(f)
        encoder = TCN(
            nlayers=remote_config['nlayers'],
            dilation_base=remote_config['dilation_base'],
            num_taps=remote_config['num_taps'],
            hidden_channels=remote_config['hidden_channels']
        )
        encoder.load_state_dict(torch.load(os.path.join(LOAD_DIR, "encoder_weights.pth")))
        encoder.eval()

        decoder = TCN(
            nlayers=remote_config['nlayers'],
            dilation_base=remote_config['dilation_base'],
            num_taps=remote_config['num_taps'],
            hidden_channels=remote_config['hidden_channels'],
        )
        decoder.load_state_dict(torch.load(os.path.join(LOAD_DIR, "decoder_weights.pth")))
        decoder.eval()

        # Add these models to STATE to pass information among scripts
        STATE['encoder'] = encoder
        STATE['decoder'] = decoder

def add_noise(signal, SNR):
    signal_power = signal.abs().pow(2).mean()
    noise_power = signal_power / SNR
    noise_std = (noise_power / 2) ** 0.5 # real and complex
    noise = noise_std * torch.randn_like(signal) + noise_std * 1j * torch.randn_like(signal)
    signal += noise
    return signal

NUM_ITERS = 100
SNR_MIN = 10
SNR_MAX = 1e4

STATE['evms'] = []
STATE['residuals'] = []
STATE['snr_idx'] = 0
SNRS = torch.linspace(SNR_MIN, SNR_MAX, NUM_ITERS)
def characterize_channel_Tx(real, imag, f_low, f_high, subcarrier_spacing):
    Nt = STATE['Nt']
    Nf = STATE['Nf']
    constellation = get_constellation(mode="m7_apsk_constellation")
    bits = torch.randint(0, 2, size=(constellation.modulation_order * Nf * Nt, )).numpy()
    bit_strings = [str(x) for x in bits]
    bits = "".join(bit_strings)
    out = torch.tensor(constellation.bits_to_symbols(bits)).reshape(Nt, Nf)
    out = add_noise(out, SNRS[STATE['snr_idx']])
    STATE['last_sent'] = out
    num_zeros = STATE['num_zeros']
    zeros = torch.zeros((out.shape[0], num_zeros), dtype=out.dtype, device=out.device)
    out_full = torch.cat([zeros, out], dim=1)
    return out_full.real.detach().cpu().numpy().tolist(), out_full.imag.detach().cpu().numpy().tolist()

def characterize_channel_Ry(real, imag):
    real = torch.tensor(real, dtype=torch.float32, device=device).reshape(STATE['Nt'], STATE['Nf'])
    imag = torch.tensor(imag, dtype=torch.float32, device=device).reshape(STATE['Nt'], STATE['Nf'])
    out = real + 1j * imag
    # Save to h5 file
    append_symbol_frame(STATE['last_sent'],
                        STATE['last_freq_symbol_received'],
                        STATE['last_time_symbol_received'],
                        STATE['frequencies'],
                        metadata={"snr_idx": STATE['snr_idx']},
                        h5_path= r"C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled\data\channel_measurements\characterize_channel_3.5V.h5")
    STATE['snr_idx'] +=1
    STATE['snr_idx'] %= SNRS.size(0)
    out = out.flatten() # Must flatten along batch dimension to output from labview
    return out.real.detach().cpu().contiguous().numpy().tolist(), out.imag.detach().cpu().contiguous().numpy().tolist()


STATE['run_model'] = True
def validate_time_encoder(x_t):
    if STATE['run_model']:
        with torch.no_grad():
            x_t = torch.tensor(x_t, dtype=torch.float32)
            # decode_logger.debug(f"Encoder in {x_t.shape}")
            STATE['time_encoder_in'] = x_t.detach().cpu().numpy()
            x_t = STATE["encoder"](x_t)
            # decode_logger.debug(f"Encoder out {x_t.shape}")
            # x_t = in_band_filter(x_t, STATE['ks'], STATE['IFFT_LENGTH'])
            x_t = np.clip(x_t.real, a_min=-3.0, a_max=3.0)
            STATE['time_encoder_out'] = x_t.detach().cpu().numpy()
        return x_t.real.detach().cpu().numpy().tolist()
    else:
        STATE['time_encoder_in'] = np.array(x_t)
        STATE['time_encoder_out'] = STATE['time_encoder_in']
        return STATE['time_encoder_in'].tolist()

def validate_time_decoder(y_t):
    # Move functionality to modulate_OFDM_python.py
    return y_t

def verify_synchronization_TX(real, imag, f_low, f_high, subcarrier_spacing):
    STATE['verify_synchronization'] = True
    Nf, Nt = STATE['Nf'], STATE['Nt']
    probe = 10
    # Generate Single Tone
    X_test = torch.zeros(Nt, Nf, dtype=torch.complex64)
    X_test[:, probe] = 1
    if STATE['normalize_power']:
        X_test = X_test / (torch.mean(X_test.abs().pow(2), dim=1).sqrt())
    zeros = torch.zeros((X_test.shape[0], STATE['num_zeros']), dtype=X_test.dtype, device=X_test.device)
    out_full = torch.cat([zeros, X_test], dim=1)
    return out_full.real.numpy().tolist(), out_full.imag.numpy().tolist()

def verify_synchronization_Ry():
    pass



NOISY_SAMPLE_SIZE = 1 # Arbitrary
def symbols_for_noise(num_freqs:int, two_carrier=True):
    if "distinct_symbol" not in NOISY_STATE:
        NOISY_STATE["distinct_symbol"] = 0
        NOISY_STATE["symbol_iteration"] = 0
        make_new = True
    else:
        NOISY_STATE["symbol_iteration"] += 1
        if NOISY_STATE["symbol_iteration"] == NOISY_SAMPLE_SIZE:
            NOISY_STATE["distinct_symbol"] += 1
            NOISY_STATE["symbol_iteration"] = 0
            make_new = True
        else:
            make_new = False


    if not make_new:
        return NOISY_STATE["prior_symbol"]

    Nt = STATE['Nt']
    Nf = STATE['Nf']

    jitter_power = 1e-2
    jitter = np.sqrt(jitter_power/2)*(torch.randn(Nt, Nf) + 1j * torch.randn(Nt, Nf))
      #Grab constellation object
    constellation = get_constellation(mode="m7_apsk_constellation")
    bits = torch.randint(0, 2, size=(constellation.modulation_order * Nf * Nt, )).numpy()
    bit_strings = [str(x) for x in bits]
    bits = "".join(bit_strings)
    s_samples = torch.tensor(constellation.bits_to_symbols(bits)).reshape(Nt, Nf)
    s_samples += jitter


    # Symbol range -4,4
    out = torch.zeros((num_freqs), dtype=torch.complex64).unsqueeze(0)
    out[:, NOISY_STATE["distinct_symbol"] % num_freqs] = s_samples[:, 0]

    if two_carrier:
        out[:, (NOISY_STATE["distinct_symbol"] // num_freqs) % num_freqs] = s_samples[:, 1]

    if STATE['normalize_power']:
        out = out / (out.abs().pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-12)
    NOISY_STATE["prior_symbol"] = out
    return out

STATE['carrier_counter'] = 0
def test_channel_ici_TX(real, imag, f_low, f_high, subcarrier_spacing):
    test_carrier_indices = torch.arange(STATE['Nf'])[::10]
    STATE['test_carrier_indices'] = test_carrier_indices
    try:
        curr_carrierIdx = test_carrier_indices[STATE['carrier_counter']]
    except Exception as e:
        print("Too many cycles!")
    # Set to energy 1
    out = torch.zeros(STATE['Nt'], STATE['Nf'], device=device, dtype=torch.complex64)
    out[:, curr_carrierIdx] = 1.0 + 0j

    # real_part = torch.randn(STATE['Nt'], STATE['Nf'])
    # imag_part = torch.randn(STATE['Nt'], STATE['Nf'])
    # out = real_part + 1j * imag_part
    if STATE['normalize_power']:
        out = out / (out.abs().pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-12)
    STATE['last_sent'] = out
    # Attach back the zeros.
    num_zeros = STATE['num_zeros']
    zeros = torch.zeros((out.shape[0], num_zeros), dtype=out.dtype, device=out.device)
    out_full = torch.cat([zeros, out], dim=1)
    return out_full.real.detach().cpu().numpy().tolist(), out_full.imag.detach().cpu().numpy().tolist()

def plot_histograms(energy_np, log_energy_np):
    fig, axs = plt.subplots(1, 1, figsize=(20, 6))

    # axs[0].bar(np.arange(len(energy_np)), energy_np, color="steelblue")
    # axs[0].set_title("Linear Energy per Carrier")
    # axs[0].set_xlabel("Subcarrier Index")
    # axs[0].set_ylabel("Energy")

    axs.bar(np.arange(len(log_energy_np)), log_energy_np, color="darkorange")
    axs.set_title("Log10 Energy per Carrier")
    axs.set_xlabel("Subcarrier Index")
    axs.set_ylabel("log10(Energy)")

    fig.tight_layout()
    return fig

def test_channel_ici_RY(real, imag):
    real = torch.tensor(real, dtype=torch.float32, device=device).reshape(STATE['Nt'], STATE['Nf'])
    imag = torch.tensor(imag, dtype=torch.float32, device=device).reshape(STATE['Nt'], STATE['Nf'])
    out = real + 1j * imag
    energy_per_carrier = torch.mean(torch.square(out.abs()), dim=0)
    log_energy_per_carrier = torch.log10(energy_per_carrier)
    # Determine if energy leaked into other carriers
    curr_carrier_idx = STATE['test_carrier_indices'][STATE['carrier_counter']]
    curr_carrier = out[:, curr_carrier_idx].clone()
    energy_at_carrier = torch.square(out[:, curr_carrier_idx].abs()).item()
    out[:, curr_carrier_idx] = 0
    total_energy = torch.mean(torch.square(out.abs()), dim=1).item()
    relative_ICI = total_energy / energy_at_carrier


    # Convert to NumPy
    energy_np = energy_per_carrier.detach().cpu().numpy()
    log_energy_np = log_energy_per_carrier.detach().cpu().numpy()

    fig = plot_histograms(energy_np, log_energy_np)

    wandb.log({
        "Carrier energy histograms": wandb.Image(fig)
    }, step=STATE['carrier_counter'])

    plt.close(fig)
    print(curr_carrier)
    print(f"Total energy not at index {curr_carrier_idx}: {total_energy: .3f} | Energy at carrier {energy_at_carrier}")
    print(f"Relative ICI {relative_ICI}")
    print(f"Sent Carrier {STATE['last_sent'][0, curr_carrier_idx].item()} | received {curr_carrier[0].item()}")
    STATE['carrier_counter'] += 1
    out = out.flatten()
    return out.real.detach().cpu().contiguous().numpy().tolist(), out.imag.detach().cpu().contiguous().numpy().tolist()

def solve_channel_noise_TX(real, imag, f_low, f_high, subcarrier_spacing):
    # real = torch.tensor(real, dtype=torch.float32, device=device)
    # imag = torch.tensor(imag, dtype=torch.float32, device=device)
    Nt = STATE["Nt"]
    Nf = STATE["Nf"]

    out = symbols_for_noise(Nf, two_carrier=True)
    STATE['encoder_in'] = out

    # Save prediction for loss calculation
    if STATE['normalize_power']:
        out = out / (out.abs().pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-12)
    STATE['last_sent'] = out
    # Attach back the zeros.
    zeros = torch.zeros((out.shape[0], STATE['num_zeros']), dtype=out.dtype, device=out.device)
    out_full = torch.cat([zeros, out], dim=1)
    return out_full.real.detach().cpu().numpy().tolist(), out_full.imag.detach().cpu().numpy().tolist()

def solve_channel_noise_RY(real, imag):
    real = torch.tensor(real, dtype=torch.float32, device=device).reshape(STATE['Nt'], STATE['Nf'])
    imag = torch.tensor(imag, dtype=torch.float32, device=device).reshape(STATE['Nt'], STATE['Nf'])
    out = real + 1j * imag
    STATE['decoder_in'] = out
    # Update local dataset
    # append_to_npz(r"C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled\data\channel_inputs_outputs.npz", STATE["last_sent"], out)
    append_symbol_frame(STATE['last_sent'],
                        STATE['last_freq_symbol_received'],
                        freqs=STATE['frequencies'],
                        noise_sample_indexing=False,
                        received_time=STATE['last_time_symbol_received'],
                        h5_path= r"C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled\data\channel_measurements\wideband_3.1V.h5")
    print(f"Appended new symbol, distinct symbol #{NOISY_STATE['distinct_symbol']}")
    STATE['decoder_out'] = out
    out = out.flatten()
    return out.real.detach().cpu().contiguous().numpy().tolist(), out.imag.detach().cpu().contiguous().numpy().tolist()

def gather_data_TX(real, imag, f_low, f_high, subcarrier_spacing):
    Nt = STATE['Nt']
    Nf = STATE['Nf']

    # generator = torch.Generator()
    # generator.manual_seed(42)

    # Grab constellation and add small complex noise
    jitter_power = 1e-1
    jitter = np.sqrt(jitter_power/2)*(torch.randn(Nt, Nf) + 1j * torch.randn(Nt, Nf))
    # Grab constellation object
    constellation = get_constellation(mode="m7_apsk_constellation")
    bits = torch.randint(0, 2, size=(constellation.modulation_order * Nf * Nt, )).numpy()
    bit_strings = [str(x) for x in bits]
    bits = "".join(bit_strings)
    out = torch.tensor(constellation.bits_to_symbols(bits)).reshape(Nt, Nf)
    out += jitter
    if STATE['normalize_power']:
        out = out / (out.abs().pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-12)

    out = out / (out.abs().pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-12)
    out = out * random.uniform(0.5, 3.0) # variable power for training richness

    STATE['encoder_in'] = out
    STATE['last_sent'] = out

    # Attach back the zeros.
    zeros = torch.zeros((out.shape[0], STATE['num_zeros']), dtype=out.dtype, device=out.device)
    out_full = torch.cat([zeros, out], dim=1)
    return out_full.real.detach().cpu().numpy().tolist(), out_full.imag.detach().cpu().numpy().tolist()

def gather_data_RY(real, imag):
    real = torch.tensor(real, dtype=torch.float32, device=device).reshape(STATE['Nt'], STATE['Nf'])
    imag = torch.tensor(imag, dtype=torch.float32, device=device).reshape(STATE['Nt'], STATE['Nf'])
    out = real + 1j * imag
    STATE['decoder_in'] = out

    append_symbol_frame(
        STATE['last_sent'],
        STATE['last_freq_symbol_received'],
        None, # No sent time
        STATE['last_time_symbol_received'],
        STATE['frequencies'],
        zarr_path= r"C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled\data\channel_measurements\channel_3e5-7.6MHz_2.66.V_0.126A_scale2_dynamic_power_0.5-3_v2.zarr")
    STATE['decoder_out'] = out
    out = out.flatten() # Must flatten along batch dimension to output from labview
    return out.real.detach().cpu().contiguous().numpy().tolist(), out.imag.detach().cpu().contiguous().numpy().tolist()

def debug_pipeline_Tx(real, imag, f_low, f_high, subcarrier_spacing):
    Nf, Nt = STATE['Nf'], STATE['Nt']
    out =  torch.zeros(Nt, Nf, dtype=torch.complex64)
    out[:, 300] = 10

    zeros = torch.zeros((out.shape[0], STATE['num_zeros']), dtype=out.dtype, device=out.device)
    out_full = torch.cat([zeros, out], dim=1)
    return out_full.real.detach().cpu().numpy().tolist(), out_full.imag.detach().cpu().numpy().tolist()

def debug_pipeline_Ry(real, imag):
    real = torch.tensor(real, dtype=torch.float32, device=device).reshape(STATE['Nt'], STATE['Nf'])
    imag = torch.tensor(imag, dtype=torch.float32, device=device).reshape(STATE['Nt'], STATE['Nf'])
    out = real + 1j * imag
    out = out.flatten() # Must flatten along batch dimension to output from labview
    return out.real.detach().cpu().contiguous().numpy().tolist(), out.imag.detach().cpu().contiguous().numpy().tolist()

def stop_validation_model():
    wandb.finish()
    pass

def plot_SNR_vs_freq(step: int, save_path: str = None):

    try:
        # Get frequencies
        data_frequencies = STATE['frequencies']

        last_received = STATE['received_symbols'][-1]  # [Nt, Nf]
        last_predicted = STATE['predicted_received_symbols'][-1]  # [Nt, Nf]

        num = torch.mean(torch.abs(last_predicted - last_received), dim=0) # Mean across Nt
        denom = torch.mean(torch.abs(last_received), dim=0)

        # Calculate EVM by freq
        evm_by_freq = num / denom
        assert len(data_frequencies) == len(evm_by_freq)

        snr_by_freq = 1 / (evm_by_freq ** 2 + 1e-8)
        snr_by_freq_dB = 10 * torch.log10(snr_by_freq + 1e-8)
        snr_by_freq_dB = snr_by_freq_dB.detach().cpu().numpy()

        # Now, calculate integral of SNR over freq ot get information bandwidth
        freqs = data_frequencies.detach().cpu().numpy()
        snr_by_freq = snr_by_freq.detach().cpu().numpy()
        freqs = np.array(freqs)
        snr_by_freq = np.array(snr_by_freq)

        integrated_snr = np.trapz(snr_by_freq, freqs)
        bandwidth = freqs[-1] - freqs[0]
        mean_snr = integrated_snr / bandwidth
        mean_snr_dB = 10 * np.log10(mean_snr)
        C_total = np.trapz(np.log2(1 + snr_by_freq), freqs)

        # Plot SNR vs Frequency
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(freqs, snr_by_freq_dB, marker='o', linestyle='-')
        ax.set_title(f"SNR vs Frequency @ Step {step} | Mean SNR (dB) {mean_snr_dB: .2f} | Estimated C {C_total: .3e}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("SNR (Log Scale dB)")
        ax.grid(True)

        # Save and log
        if save_path is not None:
            fig.savefig(save_path, dpi=150)

        os.makedirs("wandb_constellations", exist_ok=True)
        plot_path = f"wandb_constellations/snr_freq_step_{step}.png"
        fig.savefig(plot_path, dpi=150)
        wandb.log({"SNR vs Frequency": wandb.Image(plot_path)}, step=step)
        os.remove(plot_path)
        plt.close(fig)

        # Flatten both along [Nt * Nf]
        recv_flat = last_received.flatten().detach().cpu().numpy()
        pred_flat = last_predicted.flatten().detach().cpu().numpy()

        # Create plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(pred_flat.real, pred_flat.imag, s=15, label="Predicted", alpha=0.6)
        ax.scatter(recv_flat.real, recv_flat.imag, s=15, label="Received", alpha=0.6, marker='x', c='red')

        ax.set_title(f"Constellation Diagram (Last OFDM Frame) - Step {step}")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        ax.grid(True)
        ax.legend()

        # Save + log to wandb
        os.makedirs("wandb_constellations", exist_ok=True)
        plot_path = f"wandb_constellations/last_frame_constellation_step_{step}.png"
        fig.savefig(plot_path, dpi=150)
        wandb.log({"Constellation (Last Frame)": wandb.Image(plot_path)}, step=step)
        os.remove(plot_path)
        plt.close(fig)

    except Exception as e:
        print(f"Failed to plot SNR vs Frequency at step {step}: {e}")
        traceback.print_exc()

def log_constellation(step, freqs=None, evm_loss=-99):
    """
    Logs a 2x2 subplot showing encoder/decoder constellations.
    Adds:
    - Encoder input shown as gray 'x' in all plots.
    - Frequency-colored constellation points (if freqs provided).
    """

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    fig.suptitle(f"Constellation Flow Symbol 1 @ Step {step}", fontsize=14)

    try:
        # Extract data from STATE
        enc_in = STATE['encoder_in']
        enc_out = STATE['encoder_out']
        dec_in = STATE['decoder_in']
        dec_out = STATE['decoder_out']

        def to_numpy(x):
            x = x.detach().cpu() if torch.is_tensor(x) else x
            x = x[0] if x.ndim == 2 else x
            return x.numpy()

        enc_in_np = to_numpy(enc_in)
        data = {
            "Encoder Input": to_numpy(enc_in),
            "Encoder Output": to_numpy(enc_out),
            "Decoder Input": to_numpy(dec_in),
            f"Decoder Output | Frame BER {round(STATE['frame_BER'], 3)} | Frame Loss {round(evm_loss.item(), 5)}": to_numpy(dec_out),
        }

        # Normalize frequency for color mapping (if provided)
        if freqs is not None:
            freqs = np.asarray(freqs)
            norm_freqs = (freqs - freqs.min()) / (freqs.max() - freqs.min())
            point_colors = cm.viridis(norm_freqs)
        else:
            point_colors = None

        for ax, (label, symbols) in zip(axs.flat, data.items()):
            if point_colors is not None and len(symbols) == len(point_colors):
                ax.scatter(symbols.real, symbols.imag, s=8, c=point_colors,label=label)
            else:
                ax.scatter(symbols.real, symbols.imag, s=8, alpha=0.8, label=label)

            ax.scatter(enc_in_np.real, enc_in_np.imag, s=20, c='gray', marker='x', label='Encoder Input')
            ax.set_title(label)
            ax.set_xlabel("Re")
            ax.set_ylabel("Im")
            ax.grid(True)

        # Add a single shared colorbar if using frequency coloring
        if freqs is not None:
            norm = colors.Normalize(vmin=freqs.min(), vmax=freqs.max())
            sm = cm.ScalarMappable(norm=norm, cmap='viridis')
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.02, pad=0.02)
            cbar.set_label("Carrier Frequency (Hz)")

        # fig.tight_layout(rect=[0, 0, 1, 0.95])

        os.makedirs("wandb_constellations", exist_ok=True)
        plot_path = f"wandb_constellations/constellation_step_{step}.png"
        fig.savefig(plot_path, dpi=150)
        wandb.log({"Constellation Diagram": wandb.Image(plot_path)}, step=step)
        os.remove(plot_path)
        plt.close(fig)

    except Exception as e:
        print(f"Failed to plot constellation at step {step}: {e}")

def log_channel_estimate(step, H_est: torch.Tensor, freqs: torch.Tensor):
    """
    Plots estimated complex channel H_k for each subcarrier,
    colored by carrier frequency.

    Args:
        step: Training step
        H_est: Tensor of shape [Nt, Nf] or [Nf] (complex)
        freqs: Tensor of shape [Nf] (float)
    """
    try:
        # Average across time if needed
        if H_est.ndim == 2:
            H_mean = H_est.mean(dim=0)  # [Nf]
        else:
            H_mean = H_est  # already [Nf]

        H_mean = H_mean.detach().cpu().numpy()
        freqs = freqs.detach().cpu().numpy()

        norm_freqs = (freqs - freqs.min()) / (freqs.max() - freqs.min())
        point_colors = cm.viridis(norm_freqs)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(H_mean.real, H_mean.imag, c=point_colors, s=15, alpha=0.8)
        ax.set_title(f"Estimated Channel H_k (Step {step})")
        ax.set_xlabel("Re(H_k)")
        ax.set_ylabel("Im(H_k)")
        ax.grid(True)
        ax.set_aspect("equal")

        # Colorbar
        norm = colors.Normalize(vmin=freqs.min(), vmax=freqs.max())
        sm = cm.ScalarMappable(norm=norm, cmap='viridis')
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label("Carrier Frequency (Hz)")

        os.makedirs("wandb_constellations", exist_ok=True)
        plot_path = f"wandb_constellations/estimated_Hk_step_{step}.png"
        fig.savefig(plot_path, dpi=150)
        wandb.log({"Estimated H_k Constellation": wandb.Image(plot_path)}, step=step)
        os.remove(plot_path)
        plt.close(fig)

    except Exception as e:
        print(f"Failed to plot estimated H_k at step {step}: {e}")
        traceback.print_exc()

def log_channel_magnitude_phase(step, H_est: torch.Tensor, freqs: torch.Tensor):
    """
    Plots the magnitude and phase of estimated channel H_k across frequency.

    Args:
        step: Training step
        H_est: Tensor of shape [Nt, Nf] or [Nf] (complex)
        freqs: Tensor of shape [Nf] (float)
    """
    try:
        # Average across time if needed
        if H_est.ndim == 2:
            H_mean = H_est.mean(dim=0)  # [Nf]
        else:
            H_mean = H_est  # [Nf]

        H_mean = H_mean.detach().cpu().numpy()
        freqs = freqs.detach().cpu().numpy()

        magnitude = np.abs(H_mean)
        magnitude_dB = 20 * np.log10(magnitude + 1e-12)
        phase = np.unwrap(np.angle(H_mean))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(f"Channel Estimate Magnitude and Phase @ Step {step}", fontsize=14)

        ax1.plot(freqs, magnitude_dB, label="|H(f)|", color='blue')
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True)

        ax2.plot(freqs, phase * (180 / np.pi), label="Phase(H(f))", color='green')
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (deg)")
        ax2.grid(True)

        os.makedirs("wandb_constellations", exist_ok=True)
        plot_path = f"wandb_constellations/estimated_Hk_mag_phase_step_{step}.png"
        fig.savefig(plot_path, dpi=150)
        wandb.log({"Estimated H_k Mag/Phase": wandb.Image(plot_path)}, step=step)
        os.remove(plot_path)
        plt.close(fig)

    except Exception as e:
        print(f"[Error] Failed to plot H_k magnitude and phase at step {step}: {e}")
        traceback.print_exc()

def log_encoder_frequency_sensitivity(step, freqs, fixed_symbol=1 + 1j):
    """
    Logs how the encoder output varies with frequency for a fixed input symbol.
    Useful to verify that encoder encodes frequency information.

    Args:
        step (int): Current training step for logging.
        freqs (torch.Tensor): Tensor of shape [Nf] or [1, Nf] containing frequency values.
        fixed_symbol (complex): Complex symbol repeated across frequencies.
    """
    try:
        # Ensure freqs shape is [1, Nf]
        if freqs.ndim == 1:
            freqs = freqs.unsqueeze(0)

        B, Nf = freqs.shape

        # Fixed complex input symbol repeated across frequencies
        x = torch.full((B, Nf), fill_value=fixed_symbol, dtype=torch.cfloat, device=freqs.device)

        # Forward pass through encoder
        with torch.no_grad():
            out = STATE['encoder'](x, freqs)  # [1, Nf] complex

        out_np = out.squeeze(0).cpu().numpy()  # [Nf]
        freqs_np = freqs.squeeze(0).cpu().numpy()

        # Normalize freqs for color
        norm_freqs = (freqs_np - freqs_np.min()) / (freqs_np.max() - freqs_np.min())
        point_colors = cm.viridis(norm_freqs)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(out_np.real, out_np.imag, c=point_colors, s=15)
        ax.set_title(f"Encoder Output vs Frequency (Fixed Symbol {fixed_symbol}) @ Step {step}")
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.grid(True)
        ax.set_aspect("equal")

        # Add colorbar
        norm = colors.Normalize(vmin=freqs_np.min(), vmax=freqs_np.max())
        sm = cm.ScalarMappable(norm=norm, cmap='viridis')
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Carrier Frequency (Hz)")
        fig.tight_layout()

        # Save + log
        os.makedirs("wandb_constellations", exist_ok=True)
        plot_path = f"wandb_constellations/encoder_freq_sensitivity_step_{step}.png"
        fig.savefig(plot_path, dpi=150)
        wandb.log({"Encoder Frequency Sensitivity": wandb.Image(plot_path)}, step=step)
        os.remove(plot_path)
        plt.close(fig)

    except Exception as e:
        print(f"[Error] Encoder frequency sensitivity plot failed at step {step}: {e}")
        traceback.print_exc()


def log_evm_vs_time(step: int = None):
    """
    Logs the EVM (Error Vector Magnitude) between all predicted and received symbols
    stored in STATE['predicted_received_symbols'] and STATE['received_symbols'].
    Plots EVM vs. frame/time and logs to wandb.
    """
    try:
        preds = STATE['predicted_received_symbols']
        recvs = STATE['received_symbols']
        if len(preds) == 0 or len(recvs) == 0:
            print("[log_evm_vs_time] No data to log.")
            return

        # Stack to [num_frames, Nt, Nf] or [num_frames, Nf]
        preds = torch.stack(preds, dim=0)
        recvs = torch.stack(recvs, dim=0)

        # Compute EVM per frame: mean over all symbols in each frame
        evm_per_frame = torch.mean(torch.abs(preds - recvs), dim=(1, 2) if preds.ndim == 3 else 1)  # [num_frames]

        # Convert to numpy for plotting
        evm_per_frame_np = evm_per_frame.detach().cpu().numpy()
        wandb.log({"train/evm_diff_frame": evm_per_frame_np}, step=step)


    except Exception as e:
        print(f"[Error] Failed to log EVM vs time: {e}")

def plot_received_vs_predicted(received_symbols: torch.Tensor,
                               predicted_symbols: torch.Tensor,
                               step: int,
                               freqs: torch.Tensor = None):
    """
    Plots the predicted vs received symbols on a constellation diagram,
    computes EVM, and logs to wandb.

    Args:
        received_symbols (torch.Tensor): Actual received symbols [Nt, Nf]
        predicted_symbols (torch.Tensor): Model-predicted received symbols [Nt, Nf]
        step (int): Current training step for logging
        freqs (torch.Tensor, optional): Frequencies corresponding to subcarriers [Nf] or [1, Nf]
    """

    # Compute EVM
    evm = torch.mean((received_symbols.real - predicted_symbols.real) ** 2 +
                     (received_symbols.imag - predicted_symbols.imag) ** 2).item()

    # Detach and convert to numpy
    recv_np = received_symbols.detach().cpu().numpy().flatten()
    pred_np = predicted_symbols.detach().cpu().numpy().flatten()

    # Convert freqs if provided
    if freqs is not None:
        freqs = freqs.detach().cpu().numpy()
        if freqs.ndim == 2:
            freqs = freqs[0]  # Take first symbol if batched
        norm = colors.Normalize(vmin=freqs.min(), vmax=freqs.max())
        colors_mapped = cm.viridis(norm(freqs))
    else:
        colors_mapped = None

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    if colors_mapped is not None and len(colors_mapped) == len(pred_np):
        ax.scatter(pred_np.real, pred_np.imag, c=colors_mapped, s=10, label="Predicted", alpha=0.8)
        sm = cm.ScalarMappable(norm=norm, cmap='viridis')
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Subcarrier Frequency (Hz)")
    else:
        ax.scatter(pred_np.real, pred_np.imag, s=10, label="Predicted", alpha=0.8)

    ax.scatter(recv_np.real, recv_np.imag, s=10, label="Received", alpha=0.8, marker='x', color='red')

    ax.set_title(f"Predicted vs Received Constellation @ Batch {step}\nEVM: {evm:.2e}")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')

    # Save and log
    os.makedirs("wandb_constellations", exist_ok=True)
    plot_path = f"wandb_constellations/received_vs_predicted_step_{step}.png"
    fig.savefig(plot_path, dpi=150)
    wandb.log({"Predicted vs Received Constellation": wandb.Image(plot_path), "eval/evm_received_vs_pred": evm}, step=step)
    plt.close(fig)
    os.remove(plot_path)

def append_symbol_frame(
    sent, received, sent_time, received_time, freqs,
    zarr_path= r"example.zarr",
    metadata: dict = None,
    noise_sample_indexing = False,

):
    """
    Append a sent/received symbol frame to an HDF5 database with auto-incremented frame index.

    Args:
        sent (Tensor or ndarray): [Nt, Nf] complex64 sent symbols
        received (Tensor or ndarray): [Nt, Nf] complex64 received symbols
        freqs (Tensor or ndarray): [Nf] float32 subcarrier frequencies
        h5_path (str): path to .h5 file
        metadata (dict): optional dictionary of scalar metadata (e.g., {'snr': 32.1})
        noise_sample_indexing: boolean for whether we're measuring noise or not
    """
    # Convert to numpy
    def to_numpy(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    sent = to_numpy(sent).astype(np.complex64)
    received = to_numpy(received).astype(np.complex64)
    freqs = to_numpy(freqs).astype(np.float32)

    try:
        f = zarr.open(zarr_path, mode='a')

        time_stamp = int(time.time())
        group_name = f"frame_{time_stamp}"
        grp = f.create_group(group_name, overwrite=False)
        grp.create_array("sent", sent, compressor="default", chunks=True)
        grp.create_array("received", received, compressor="default", chunks=True)
        grp.create_array("freqs", freqs, compressor="default", chunks=True)
        
        if received_time is not None:
            cp_length = STATE['cp_length']
            num_points_symbol = STATE['num_points_symbol']
            grp.attrs["num_points_symbol"] = num_points_symbol
            grp.attrs["cp_length"] = cp_length
            grp.create_array(
                "received_time",
                received_time.astype(np.float32),
                compressor="default",
                chunks=True         
            )

        if sent_time is not None:
            grp.create_array(
                "sent_time",
                received_time.astype(np.float32),
                compressor="default",
                chunks=True         
            )

       
        # Store metadata if provided
        if metadata:
            for key, value in metadata.items():
                try:
                    grp.attrs[key] = value
                except Exception as e:
                    print(f"Could not save metadata key '{key}': {e}")

    except Exception as e:
        decode_logger.warning(f"[append skipped] Error {e}")
        return None

    return time_stamp