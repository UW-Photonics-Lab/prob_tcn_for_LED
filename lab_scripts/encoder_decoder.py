
import sys
project_root = r"C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from training_state import STATE
import zarr
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import wandb
import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import time
from lab_scripts.constellation_diagram import QPSK_Constellation, get_constellation, RingShapedConstellation
from modules.models import TCN
from modules.utils import evm_loss, load_runs_final_artifact
import torch.nn.functional as F
from scipy.signal import resample_poly
import random
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
from lab_scripts.logging_code import *
decode_logger = setup_logger(log_file=r"C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled\debug_logs\test3.txt")

STATE['validate_model'] = True # Variable
STATE['normalize_power'] = False

load_model = True # Variable
LOAD_DIR = ""
if load_model:
    model_name = "eager-grass-413" # Variable
    STATE['encoder'], STATE['decoder'], hyperparams = load_runs_final_artifact(model_name, device=torch.device('cpu'), model_type="encoder_decoder", root_dir=project_root)
    # Start Weights and Biases session
    wandb.init(project="mldrivenpeled",
            config=hyperparams)
    config = wandb.config

    print("WandB run info:")
    print(f"  Name: {wandb.run.name}")
    print(f"  ID: {wandb.run.id}")
    print(f"  URL: {wandb.run.url}")
    print("Chosen hyperparameters for this session:")

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


# Unfortunate global state variables because of LabVIEW-Python integration
STATE['loss_accumulator'] = []
STATE['predicted_received_symbols'] = []
STATE['encoder_out_buffer'] = []
STATE['decoder_in_buffer'] = []
STATE['evms'] = []
STATE['residuals'] = []
STATE['run_model'] = True

def validate_time_encoder(x_t):
    if STATE['run_model']:
        with torch.no_grad():
            x_t = torch.tensor(x_t, dtype=torch.float32)
            # decode_logger.debug(f"Encoder in {x_t.shape}")
            STATE['time_encoder_in'] = x_t.detach().cpu().numpy()
            x_t = STATE["encoder"](x_t)
            # decode_logger.debug(f"Encoder out {x_t.shape}")
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
        zarr_path= r"C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled\data\channel_measurements\test.zarr")
    STATE['decoder_out'] = out
    out = out.flatten() # Must flatten along batch dimension to output from labview
    return out.real.detach().cpu().contiguous().numpy().tolist(), out.imag.detach().cpu().contiguous().numpy().tolist()

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
        grp = f.create_group(group_name)
        grp.create_dataset("sent", data=sent)
        grp.create_dataset("received", data=received)
        grp.create_dataset("freqs", data=freqs)
        
        if received_time is not None:
            cp_length = STATE['cp_length']
            num_points_symbol = STATE['num_points_symbol']
            grp.attrs["num_points_symbol"] = num_points_symbol
            grp.attrs["cp_length"] = cp_length
            grp.create_dataset(
                "received_time",
                data=received_time.astype(np.float32)   
            )

        if sent_time is not None:
            grp.create_dataset(
                "sent_time",
                data=received_time.astype(np.float32),       
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