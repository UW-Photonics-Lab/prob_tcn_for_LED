from training_state import STATE
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

with open("config.yml", "r") as f:
    hyperparams = yaml.safe_load(f)


# Start Weights and Biases session
wandb.init(project="mldrivenpeled", config=hyperparams)
config = wandb.config


# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps") # for M chip Macs
else:
    device = torch.device("cpu")

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
encoder = TransformerEncoder(d_model=config.dmodel,
                             nhead=config.nhead,
                             nlayers=config.nlayers,
                             dim_feedforward=config.dim_feedforward,
                             dropout=config.dropout).to(device)

decoder = TransformerDecoder(d_model=config.dmodel,
                             nhead=config.nhead,
                             nlayers=config.nlayers,
                             dim_feedforward=config.dim_feedforward,
                             dropout=config.dropout).to(device)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=config.lr)

# Create scheduler that reduces LR if validation loss plateaus
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True, min_lr=1e-6)

# Called in Labview
def run_encoder(real, imag, freqs):
    """
    Inputs: real, imag, freqs - Python lists [B][N]
    Returns: encoded_real, encoded_imag - same shape
    # """
    real = torch.tensor(real, dtype=torch.float32, device=device)
    imag = torch.tensor(imag, dtype=torch.float32, device=device)
    freqs = torch.tensor(freqs, dtype=torch.float32, device=device)

    x = real + 1j * imag

    out = encoder(x, freqs)
    return out.real.detach().cpu().numpy().tolist(), out.imag.detach().cpu().squeeze().numpy().tolist()

def run_decoder(real, imag):
    real = torch.tensor(real, dtype=torch.float32, device=device)
    imag = torch.tensor(imag, dtype=torch.float32, device=device)

    x = real + 1j * imag
    out = decoder(x)
    
    return out.real.detach().cpu().numpy().tolist(), out.imag.detach().cpu().squeeze().numpy().tolist()