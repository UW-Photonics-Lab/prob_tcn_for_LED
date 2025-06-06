from training_state import STATE
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import time
import pprint
from transformers import get_linear_schedule_with_warmup

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "..", "config.yml")
with open(config_path, "r") as f:
    hyperparams = yaml.safe_load(f)


# Start Weights and Biases session
wandb.init(project="mldrivenpeled",
           config=hyperparams)
config = wandb.config

print(f"WandB run info:")
print(f"  Name: {wandb.run.name}")
print(f"  ID: {wandb.run.id}")
print(f"  URL: {wandb.run.url}")
print("Chosen hyperparameters for this session:")
pprint.pprint(config)  

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
STATE['batch_count'] = 0

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
            x: [Nt, f, d_model]
            freq: [Nt, Nf]
        '''

        Nt, Nf, _ = x.shape

        # Frequency embedding
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device) * -np.log(1e4) / self.d_model
            ).unsqueeze(0).unsqueeze(0) # [1, 1, d_model//2]
        pe = torch.zeros(Nt, Nf, self.d_model).to(device)
        freq = freq.unsqueeze(-1) # [Nt, Nf, 1]
        angles = div_term * freq # [Nt, Nf, dmodel//2]
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
        x_real = x.real.unsqueeze(-1) # [Nt, Nf, 1]
        x_imag = x.imag.unsqueeze(-1) # [Nt, Nf 1]
        combined = torch.cat([x_real, x_imag], dim=-1) # [Nt, Nf, 2]
        return self.linear(combined) # [Nt, Nf, 2] -> [Nt, Nf, d_model]



class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, nlayers, dim_feedforward, dropout):
        super().__init__()
        self.frequency_embed = FrequencyPositionalEmbedding(d_model)
        self.symbol_embed = SymbolEmbedding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=dropout, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers)
        self.output = nn.Linear(d_model, 2)

    def forward(self, x: torch.tensor, freqs: torch.tensor) -> torch.tensor:
        embedded_symbols = self.symbol_embed(x) # [Nt, Nf, d_model]
        freq_embedded_symbols = self.frequency_embed(embedded_symbols, freqs) # [Nt, Nf, d_model]
        out = self.transformer(freq_embedded_symbols) # [Nt, Nf, d_model]
        out = self.output(out) #[Nt, Nf, 2]
        return out[..., 0] + 1j * out[..., 1] # Real and Imag

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, nlayers, dim_feedforward, dropout):
        super().__init__()
        self.sym_embed = SymbolEmbedding(d_model)
        decoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=dropout, norm_first=True) # bathcing along Nt
        self.transformer = nn.TransformerEncoder(decoder_layer, nlayers)
        self.output = nn.Linear(d_model, 2)

    def forward(self, x):
        x_embed = self.sym_embed(x)
        out = self.transformer(x_embed)
        out = self.output(out)
        return out[..., 0] + 1j * out[..., 1]

class TimeDomainCNN(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=32, f3dB=float(config.f3dB), num_layers=3, OFDM_period=float(config.OFDM_period), num_points_time=int(config.num_points_time)):
        super().__init__()

        # Calculate kernel size based of f3dB response of LED
        kernel_time_length = 3 / (2 * np.pi * f3dB) # factor of 3
        kernel_percentage = kernel_time_length / OFDM_period
        num_points_kernel = int(kernel_percentage * num_points_time)

        layers = [] 
        for i in range(num_layers):
            layers.append(nn.Conv1d(
                in_channels if i == 0 else hidden_channels,
                hidden_channels,
                kernel_size=num_points_kernel,
                padding=num_points_kernel // 2
            ))
        layers.append(nn.ReLU())

        layers.append(nn.Conv1d(hidden_channels, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)


    def forward(self, x):
        '''
        Args:
            x: [1, num_points_time]

        Returns:
            [1, num_points_time]
        
        '''
        original_len = x.shape[-1]
        # Add batch dimension for Pytorch [1, 1, num_points_time]
        x = x.unsqueeze(0)
        out = self.net(x)
        out = out[..., -original_len:]
        return out.squeeze(0) # [1, num_points_time]


def evm_loss(true_symbols, predicted_symbols):
        return torch.mean((true_symbols.real - predicted_symbols.real) ** 2 + (true_symbols.imag - predicted_symbols.imag) ** 2)

# === Initialize models on device ===
freq_encoder = TransformerEncoder(d_model=config.d_model,
                             nhead=config.nhead,
                             nlayers=config.nlayers,
                             dim_feedforward=config.dim_feedforward,
                             dropout=config.dropout).to(device)


time_encoder = TimeDomainCNN(
    hidden_channels=32,
    num_layers=3
).to(device)

decoder = TransformerDecoder(d_model=config.d_model,
                             nhead=config.nhead,
                             nlayers=config.nlayers,
                             dim_feedforward=config.dim_feedforward,
                             dropout=config.dropout).to(device)


def apply_weight_init(model, method):
    if method != "default":
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if method == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif method == "kaiming":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                elif method == "normal":
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0)

apply_weight_init(freq_encoder, config.weight_init)
apply_weight_init(time_encoder, config.weight_init)
apply_weight_init(decoder, config.weight_init)

optimizer = optim.Adam(list(freq_encoder.parameters()) + list(decoder.parameters()), lr=float(config.lr))
if config.scheduler_type == "reduce_lr_on_plateu":
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
elif config.scheduler_type == "warmup":
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.epochs
    )
# Add these models to STATE to pass information among scripts

STATE['freq_encoder'] = freq_encoder
STATE['time_encoder'] = time_encoder
STATE['decoder'] = decoder
STATE['optimizer'] = optimizer
STATE['scheduler'] = scheduler


def evm_loss_func(true_symbols, predicted_symbols):
    """
    Computes the mean squared error between true and predicted complex symbols.
    Args:
        true_symbols: torch.Tensor of shape [N] (complex)
        predicted_symbols: torch.Tensor of shape [N] (complex)
    Returns:
        torch scalar (mean EVM)
    """
    return torch.mean((true_symbols.real - predicted_symbols.real) ** 2 +
                      (true_symbols.imag - predicted_symbols.imag) ** 2)

# Called in Labview
def run_freq_encoder(real, imag, f_low, f_high, subcarrier_spacing):
    """
    Inputs: real, imag, freqs - Python lists [B, N]
    Returns: encoded_real, encoded_imag - same shape
    """

    # Grab current time
    start_time = time.time()
    STATE['start_time'] = start_time
    STATE['ML_time'] = 0

    real = torch.tensor(real, dtype=torch.float32, device=device)
    imag = torch.tensor(imag, dtype=torch.float32, device=device)
    freqs = torch.tensor(np.arange(0, f_high, subcarrier_spacing), dtype=torch.float32, device=device) # Start at 0 frequency for shape matching
    k_min = int(np.floor(f_low / subcarrier_spacing))
    x = real + 1j * imag

    data_freqs = torch.tensor(np.arange(f_low, f_high, subcarrier_spacing), dtype=torch.float32, device=device)

    out = STATE['freq_encoder'](x, freqs)
    STATE['freq_encoder_in'] = x[:, k_min:]
    STATE['freq_encoder_out'] = out[:, k_min:] # Remove 0 frequency carriers
    STATE['frequencies'] = data_freqs
    # Store out in STATE to preserve computational graph

    STATE['ML_time'] += (time.time() - start_time)
    return out.real.detach().cpu().numpy().tolist(), out.imag.detach().cpu().numpy().tolist()


def run_time_encoder(real_x_t):
    if config.cnn:
        cnn_equalizer = STATE['time_encoder']
    else:
        cnn_equalizer = nn.Identity()
    real_x_t = torch.tensor(real_x_t, dtype=torch.float32, device=device)
    STATE['time_decoder_in'] = real_x_t
    out = cnn_equalizer(real_x_t)[:]
    STATE['time_decoder_out'] = out

    # Now create encoder_out to backprop through pseudochannel
    STATE['encoder_out'] = STATE['time_decoder_out']
    return out.detach().cpu().contiguous().numpy().tolist()

def differentiable_channel(encoder_out: torch.tensor, decoder_in: torch.tensor):

    if config.cnn == True:
        Nf = STATE['Nf']
        Nt = STATE['Nt']
        I_s = torch.fft.ifft(decoder_in, n=int(config.num_points_time)//Nt, axis=1).flatten()
        remainder_zeros = torch.zeros(int(config.num_points_time) % int(config.num_symbols_per_frame), device=I_s.device, dtype=I_s.dtype)
        I_s = torch.cat([I_s, remainder_zeros])
        # Ensure length is a multiple of Nt
        total_len = (I_s.numel() // Nt) * Nt
        def H(x):
            x_flat = (x * (I_s[:total_len] / encoder_out.flatten()[:total_len]))
            return torch.fft.fft(x_flat.reshape(Nt, -1), axis=1)[:Nf]
        return H
    else:
        return lambda x: x * (decoder_in / encoder_out) # Divide equal sized tensors to get H(jw)

def run_decoder(real, imag):
    # --- Reshape input to [Nt, Nf] ---
    Nt, Nf = STATE['Nt'], STATE['Nf']
    start_time = time.time()
    real = torch.tensor(real, dtype=torch.float32, device=device).reshape(Nt, Nf)
    imag = torch.tensor(imag, dtype=torch.float32, device=device).reshape(Nt, Nf)
    x = real + 1j * imag
    STATE['decoder_in'] = x
    STATE['H_channel'] = differentiable_channel(STATE['encoder_out'], STATE['decoder_in'])

    out =  STATE['decoder'](STATE['H_channel'](STATE['encoder_out']))
    # Store out in STATE to preserve computational graph
    STATE['decoder_in'] = x
    STATE['decoder_out'] = out
    out = out.flatten() # Must flatten along batch dimension to output from labview
    STATE['ML_time'] += (time.time() - start_time)
    return out.real.detach().cpu().contiguous().numpy().tolist(), out.imag.detach().cpu().contiguous().numpy().tolist()


def update_weights(batch_size=config.batch_size) -> bool:
    '''Updates weights

    Returns:
        cancel_early: bool flags whether to stop current training session
    
    
    '''
    output = False
    STATE['cycle_count'] += 1
    true_symbols = STATE['encoder_in']
    predicted_symbols = STATE['decoder_out']
    evm_loss = evm_loss_func(true_symbols, predicted_symbols)
    if evm_loss is not None:
        start_time = time.time()
        STATE['loss_accumulator'].append(evm_loss)

        elapsed_time = time.time() - STATE['start_time']
        # Calculate time for ML parts
        STATE['ML_time'] += (time.time() - start_time)
        
        # Push plots to wand
        if len(STATE['loss_accumulator']) == batch_size:
            STATE['batch_count'] += 1
            batch_avg_loss = torch.mean(torch.stack(STATE['loss_accumulator']))
            if config.scheduler_type == "reduce_lr_on_plateu":
                scheduler.step(batch_avg_loss)
            elif config.scheduler_type == "warmup":
                scheduler.step()

            # Track with WandB
            wandb.log({"train/loss" :batch_avg_loss.item()}, step=STATE['cycle_count'])
            STATE['optimizer'].zero_grad()
            batch_avg_loss.backward()
            STATE['optimizer'].step()
            # Clear accumulator
            STATE['loss_accumulator'].clear()

            # Save final loss so that optuna can use it later
            with open("final_loss.txt", "w") as f:
                    f.write(str(batch_avg_loss.item()))
            
            
            wandb.log({"perf/time_for_ML": STATE['ML_time']}, step=STATE['cycle_count'])
            wandb.log({"perf/time_encode_to_decode": elapsed_time}, step=STATE['cycle_count'])
            # Percentage ML time
            wandb.log({"perf/percent_time_for_ML": STATE['ML_time'] / elapsed_time}, step=STATE['cycle_count'])

            # Check if need to cancel run early
            
            if STATE['batch_count'] > config.EARLY_STOP_PATIENCE:

                if batch_avg_loss > config.EARLY_STOP_THRESHOLD:
                    # Cancel training

                        # Cancel training
                    msg = (
                        f"Early stopping triggered at batcg {STATE['batch_count']}! "
                        f"Batch avg loss {batch_avg_loss:.4f} exceeded threshold {config.EARLY_STOP_THRESHOLD}."
                    )
                    print(msg)
                    output = True


    if STATE['cycle_count'] % config.plot_frequency == 0:
        step = STATE['cycle_count']

        log_constellation(step=step, freqs=STATE['frequencies'], evm_loss=evm_loss)

        lr = optimizer.param_groups[0]["lr"]
        wandb.log({"train/lr": lr}, step=step)
        wandb.log({"perf/frame_BER": STATE['frame_BER']}, step=step)

        for model_name in ['encoder', 'decoder']:
            model = STATE[model_name]
            for name, param in model.named_parameters():
                # Log weights
                wandb.log({f"weights/{model_name}/{name}": wandb.Histogram(param.data.cpu())}, step=step)

                # Log gradients and accumulate norm
                if param.grad is not None:
                    grad = param.grad.detach().cpu()
                    wandb.log({f"grads/{model_name}/{name}": wandb.Histogram(grad)}, step=step)

                    # Calculate norm
                    grad_norm = grad.norm().item()
                    wandb.log({f"grad_norms/{model_name}/{name}": grad_norm}, step=step)
                else:
                    if len(STATE['loss_accumulator']) == batch_size:
                        print("Gradient is None Incorrectly!")

    if STATE['cycle_count'] % config.save_model_frequency == 0:
        model_save_dir = os.path.join(script_dir, "..", "saved_models")
        os.makedirs(model_save_dir, exist_ok=True)

        for model_name in ['encoder', 'decoder']:
            model = STATE[model_name]
            save_path = os.path.join(model_save_dir, f"{model_name}_step{STATE['cycle_count']}.pt")
            torch.save(model.state_dict(), save_path)
            artifact = wandb.Artifact(f"{model_name}_ckpt_step{STATE['cycle_count']}", type="model")
            artifact.add_file(save_path)
            wandb.log_artifact(artifact)
    return output

def stop_training():
    if 'encoder_out' in STATE:
        print("Finished all epochs!")
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

def log_constellation(step, freqs=None, evm_loss=-99):
    """
    Logs a 2x2 subplot showing encoder/decoder constellations.
    Adds:
    - Encoder input shown as gray 'x' in all plots.
    - Frequency-colored constellation points (if freqs provided).
    """

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Constellation Flow @ Step {step}", fontsize=14)

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
                scatter = ax.scatter(symbols.real, symbols.imag, s=8, c=point_colors,label=label)
            else:
                scatter = ax.scatter(symbols.real, symbols.imag, s=8, alpha=0.8, label=label)

            ax.scatter(enc_in_np.real, enc_in_np.imag, s=20, c='gray', marker='x', label='Encoder Input')
            ax.set_title(label)
            ax.set_xlabel("Re")
            ax.set_ylabel("Im")
            ax.grid(True)
            ax.legend()

        # Add a single shared colorbar if using frequency coloring
        if freqs is not None:
            norm = colors.Normalize(vmin=freqs.min(), vmax=freqs.max())
            sm = cm.ScalarMappable(norm=norm, cmap='viridis')
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.02, pad=0.02)
            cbar.set_label("Carrier Frequency (Hz)")

        os.makedirs("wandb_constellations", exist_ok=True)
        plot_path = f"wandb_constellations/constellation_step_{step}.png"
        fig.savefig(plot_path, dpi=150)
        wandb.log({"Constellation Diagram": wandb.Image(plot_path)}, step=step)
        os.remove(plot_path)

    except Exception as e:
        print(f"Failed to plot constellation at step {step}: {e}")
