from training_state import STATE
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from lab_scripts.logging_code import *

encoder_logger = setup_logger(log_file=r"C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled\debug_logs\encoder_log.txt")

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "..", "config.yml")
with open(config_path, "r") as f:
    hyperparams = yaml.safe_load(f)


# Start Weights and Biases session
wandb.init(project="mldrivenpeled",
           config=hyperparams)
config = wandb.config


# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps") # for M chip Macs
else:
    device = torch.device("cpu")

# Add a loss accumulator to state
STATE['loss_accumulator'] = []
STATE['cycle_count'] = 0

class FrequencyPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.linear = nn.Linear(2, d_model)


    def forward(self, x, freq):
        '''
        Applies positional encoding

        Args:
            x: [B, N, d_model]
            freq: [B, N]
        '''

        B, N, _ = x.shape

        # Frequency embedding
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device) * -np.log(1e4) / self.d_model
            ).unsqueeze(0).unsqueeze(0) # [1, 1, d_model//2]
        pe = torch.zeros(B, N, self.d_model).to(device)
        freq = freq.unsqueeze(-1) # [B, N, 1]
        angles = div_term * freq # [B, N, dmodel//2]
        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles)
        x = x + pe[:x.size(0)]
        return x
    
class SymbolEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        assert d_model % 2 == 0
        self.linear = nn.Linear(2, d_model)
        self.d_model = d_model

    def forward(self, x: torch.tensor) -> torch.tensor:
        x_real = x.real.unsqueeze(-1) # [B, N, 1]
        x_imag = x.imag.unsqueeze(-1) # [B, N, 1]
        combined = torch.cat([x_real, x_imag], dim=-1) # [B, N, 2]
        return self.linear(combined) # [B, N, 2] -> [B, N, d_model]



class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, nlayers, dim_feedforward, dropout):
        super().__init__()
        self.frequency_embed = FrequencyPositionalEmbedding(d_model)
        self.symbol_embed = SymbolEmbedding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers)
        self.output = nn.Linear(d_model, 2)

    def forward(self, x: torch.tensor, freqs: torch.tensor) -> torch.tensor:
        embedded_symbols = self.symbol_embed(x) # [B, N, d_model]
        freq_embedded_symbols = self.frequency_embed(embedded_symbols, freqs) # [B, N, d_model]
        out = self.transformer(freq_embedded_symbols) # [B, N, d_model]
        out = self.output(out) #[B, N, 2]
        return out[..., 0] + 1j * out[..., 1] # Real and Imag
    
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, nlayers, dim_feedforward, dropout):
        super().__init__()
        self.sym_embed = SymbolEmbedding(d_model)
        decoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(decoder_layer, nlayers)
        self.output = nn.Linear(d_model, 2)

    def forward(self, x):
        x_embed = self.sym_embed(x)
        out = self.transformer(x_embed)
        out = self.output(out)
        return out[..., 0] + 1j * out[..., 1]
    
def evm_loss(true_symbols, predicted_symbols):
        return torch.mean((true_symbols.real - predicted_symbols.real) ** 2 + (true_symbols.imag - predicted_symbols.imag) ** 2)

# === Initialize models on device ===
encoder = TransformerEncoder(d_model=config.d_model,
                             nhead=config.nhead,
                             nlayers=config.nlayers,
                             dim_feedforward=config.dim_feedforward,
                             dropout=config.dropout).to(device)

decoder = TransformerDecoder(d_model=config.d_model,
                             nhead=config.nhead,
                             nlayers=config.nlayers,
                             dim_feedforward=config.dim_feedforward,
                             dropout=config.dropout).to(device)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=float(config.lr))
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-6)

# Add these models to STATE to pass information among scripts

STATE['encoder'] = encoder
STATE['decoder'] = decoder
STATE['optimizer'] = optimizer
STATE['scheduler'] = scheduler

# Called in Labview
def run_encoder(real, imag, f_low, f_high, subcarrier_spacing):
    """
    Inputs: real, imag, freqs - Python lists [B][N]
    Returns: encoded_real, encoded_imag - same shape
    """

    
    real = torch.tensor(real, dtype=torch.float32, device=device)
    imag = torch.tensor(imag, dtype=torch.float32, device=device)
    freqs = torch.tensor(np.arange(f_low, f_high, subcarrier_spacing), dtype=torch.float32, device=device)

    x = real + 1j * imag

    out = encoder(x, freqs)
    STATE['encoder_out'] = out

    # encoder_logger.debug("Checking types:", out.dtype)
    # Store out in STATE to preserve computational graph
    return out.real.detach().cpu().numpy().tolist(), out.imag.detach().cpu().numpy().tolist()

def run_decoder(real, imag):
    real = torch.tensor(real, dtype=torch.float32, device=device)
    imag = torch.tensor(imag, dtype=torch.float32, device=device)

    x = real + 1j * imag
    out = decoder(x)
    # Store out in STATE to preserve computational graph
    STATE['decoder_out'] = out
    STATE['cycle_count'] += 1
    return out.real.detach().cpu().numpy().tolist(), out.imag.detach().cpu().numpy().tolist()


def update_weights(evm_loss, batch_size=config.batch_size):
    if evm_loss is not None:
        STATE['loss_accumulator'].append(evm_loss)
        STATE['scheduler'].step(evm_loss)
        if len(STATE['loss_accumulator']) == batch_size:
            batch_avg_loss = torch.mean(torch.stack(STATE['loss_accumulator']))
            # Track with WandB
            wandb.log({"train/loss" :batch_avg_loss.item()})
            STATE['optimizer'].zero_grad()
            batch_avg_loss.backward()
            STATE['optimizer'].step(batch_avg_loss)
            # Clear accumulator
            STATE['loss_accumulator'].clear()


def stop_training():
    if 'encoder_out' in STATE:
        # Save models
        model_save_dir = os.path.join(script_dir, "..", "saved_models")
        os.makedirs(model_save_dir, exist_ok=True)
        encoder_path = os.path.join(model_save_dir, "encoder.pt")
        decoder_path = os.path.join(model_save_dir, "decoder.pt")
        torch.save(STATE['encoder'].state_dict(), encoder_path)
        torch.save(STATE['decoder'].state_dict(), decoder_path)
        artifact = wandb.Artifact("transformer-models", type="model")
        artifact.add_file(encoder_path)
        artifact.add_file(decoder_path)
        wandb.log_artifact(artifact)
        wandb.finish()