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
import traceback

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

print("Device", device)
# Add a loss accumulator to state
STATE['loss_accumulator'] = []
STATE['predicted_received_symbols'] = []
STATE['received_symbols'] = []
STATE['cycle_count'] = 0
STATE['batch_count'] = 0
STATE['encoder_out_buffer'] = [] 
STATE['decoder_in_buffer'] = []

class FrequencyPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model

    def forward(self, x, freq):
        '''
        Applies positional encoding

        Args:
            x: [Nt, Nf, d_model]
            freq: [Nt, Nf]
        '''

        Nt, Nf, _ = x.shape

        # Frequency embedding
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device) * -np.log(1e4) / self.d_model
            ).unsqueeze(0).unsqueeze(0) # [1, 1, d_model//2]
        pe = torch.zeros(Nt, Nf, self.d_model).to(device)
        freq = freq.unsqueeze(-1) # [B, Nf, 1]
        # print("Frequencies", freq)
        angles = div_term * freq # [Nt, Nf, dmodel//2]
        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles)
        x = x + pe[:x.size(0)]
        return x

class FrequencyEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(STATE['Nf'], d_model)

    def forward(self, x, freqs: torch.Tensor):
        """
        Args:
            x: [Nt, Nf, d_model]  - embedded symbols
            freqs: [Nt, Nf]       - subcarrier indices (ints in [0, Nf-1])
        Returns:
            [Nt, Nf, d_model]     - embedded symbols + freq info
        """
        if freqs.ndim == 1:
            # Expand to [1, Nf] for batch compatibility
            freqs = freqs.unsqueeze(0)
        assert freqs.shape[1] == STATE['Nf']
        freq_indices = torch.arange(STATE['Nf']).repeat(STATE['Nt']).reshape(STATE['Nt'], STATE['Nf'])
        freq_embed = self.embedding(freq_indices)  # [Nt, Nf, d_model]
        return x + freq_embed
    

class SymbolEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        assert d_model % 2 == 0
        self.linear = nn.Linear(2, d_model)
        self.d_model = d_model

    def forward(self, x: torch.tensor) -> torch.tensor:
        x_real = x.real.unsqueeze(-1) # [Nt, Nf, 1]
        x_imag = x.imag.unsqueeze(-1) # [Nt, Nf, 1]
        combined = torch.cat([x_real, x_imag], dim=-1) # [Nt, Nf, 2]
        return self.linear(combined) # [Nt, Nf, 2] -> [Nt, Nf, d_model]



class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, nlayers, dim_feedforward, dropout):
        super().__init__()
        self.frequency_embed = FrequencyPositionalEmbedding(d_model)
        self.symbol_embed = SymbolEmbedding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=dropout, norm_first=config.pre_layer_norm)
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers)
        self.output = nn.Linear(d_model, 2)

    def forward(self, x: torch.tensor, freqs: torch.tensor) -> torch.tensor:
        embedded_symbols = self.symbol_embed(x) # [Nt, Nf, d_model]
        embedded_symbols = self.frequency_embed(embedded_symbols, freqs) # [Nt, Nf, d_model] testing no frequency embedding
        out = self.transformer(embedded_symbols) # [Nt, Nf, d_model]
        out = self.output(out) #[Nt, Nf, 2]
        return out[..., 0] + 1j * out[..., 1] # Real and Imag

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, nlayers, dim_feedforward, dropout):
        super().__init__()
        self.sym_embed = SymbolEmbedding(d_model)
        decoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=dropout, norm_first=config.pre_layer_norm)
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

# encoder = encoder.half()
# decoder = decoder.half()


def apply_weight_init(model, method):
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

apply_weight_init(encoder, config.weight_init)
apply_weight_init(decoder, config.weight_init)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=float(config.lr))
if config.scheduler_type == "reduce_lr_on_plateu":
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)
elif config.scheduler_type == "warmup":
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.epochs
    )
# Add these models to STATE to pass information among scripts

STATE['encoder'] = encoder
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
def run_encoder(real, imag, f_low, f_high, subcarrier_spacing):
    """
    Inputs: real, imag, freqs - Python lists [Nt, Nf]
    Returns: encoded_real, encoded_imag - same shape
    """

    # Grab current time
    start_time = time.time()
    STATE['start_time'] = start_time
    STATE['ML_time'] = 0

    real = torch.tensor(real, dtype=torch.float32, device=device)
    imag = torch.tensor(imag, dtype=torch.float32, device=device)
    freqs = torch.tensor(np.arange(0, f_high, subcarrier_spacing), dtype=torch.float32, device=device) # Start at 0 frequency for shape matching

    # Update freqs to match shape [Nt, Nf]
    freqs = freqs.unsqueeze(0).repeat(STATE['Nt'], 1)
    k_min = int(np.floor(f_low / subcarrier_spacing))
    freqs = freqs[:, k_min:]
    x = real + 1j * imag

    # print("Shapes", x.shape, freqs.shape)
    STATE['encoder_in'] = x[:, k_min:]
    out = STATE['encoder'](STATE['encoder_in'], freqs)
    out = out / (out.abs().pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-12) # Unit average power
    # Store out in STATE to preserve computational graph
    STATE['encoder_out'] = out 
    STATE['frequencies'] = freqs[0, :].detach() 
    STATE['ML_time'] += (time.time() - start_time)

    # Attach back the zeros.
    zeros = torch.zeros((out.shape[0], k_min), dtype=out.dtype, device=out.device)
    out_full = torch.cat([zeros, out], dim=1)

    return out_full.real.detach().cpu().numpy().tolist(), out_full.imag.detach().cpu().numpy().tolist()

def differentiable_channel(encoder_out: torch.tensor, received_symbols: torch.tensor, type):

    # The following code can allow for an estimate of SNR vs freq.
    X_list = STATE['encoder_out_buffer']
    Y_list = STATE['decoder_in_buffer']
    X = torch.cat(X_list, dim=0)  # Shape: [W * Nt, Nf]
    Y = torch.cat(Y_list, dim=0) 
    X = X.detach()
    Y = Y.detach()
    regularization_constant = config.matrix_regularization
    regularization_matrix = torch.eye(int(STATE['Nf'])) * regularization_constant
    # regularization_matrix = 0
    gram_matrix = X.t() @ X # [Nf, Nf]
    inverse_term = torch.linalg.inv(gram_matrix + regularization_matrix)
    H_k = inverse_term @ X.t() @ Y # [Nf, Nf]

    Y_pred = STATE['encoder_out'].detach() @ H_k
    STATE['predicted_received_symbols'].append(Y_pred.detach())
    STATE['received_symbols'].append(received_symbols.detach())

    cond = torch.linalg.cond(gram_matrix)
    # Log the condition number with wandb
    STATE['gram_cond'] = cond
    if type == 'linear':
        assert received_symbols.shape == encoder_out.shape
        H_est = received_symbols.detach() / (encoder_out.detach() + 1e-12)
        if STATE['cycle_count'] % config.plot_frequency == 0:
            log_channel_estimate(step=STATE['cycle_count'], H_est=H_est, freqs=STATE['frequencies'])
            log_channel_magnitude_phase(step=STATE['cycle_count'], H_est=H_est, freqs=STATE['frequencies'])
        return H_est # Divide equal sized tensors to get H(jw)
    elif type == 'ici_matrix':
        # Keep track of how much error there is between XH and Y_true
        Y_pred = STATE['encoder_out'].detach() @ H_k
        evm_diff = torch.mean(torch.abs(Y_pred - received_symbols))
        wandb.log({"train/evm_diff_ICI_matrix": evm_diff}, step=STATE['cycle_count'])
        return H_k #[Nf, Nf]

def run_decoder(real, imag):
    # --- Reshape input to [Nt, Nf] ---
    Nt, Nf = STATE['encoder_out'].shape
    start_time = time.time()
    real = torch.tensor(real, dtype=torch.float32, device=device).reshape(Nt, Nf)
    imag = torch.tensor(imag, dtype=torch.float32, device=device).reshape(Nt, Nf)
    x = real + 1j * imag
    STATE['decoder_in'] = x
    # Update window buffers
    STATE['encoder_out_buffer'].append(STATE['encoder_out'].detach())
    STATE['decoder_in_buffer'].append(STATE['decoder_in'].detach())

    # Truncate to most recent W entries
    W = config.ici_window_length
    STATE['encoder_out_buffer'] = STATE['encoder_out_buffer'][-W:]
    STATE['decoder_in_buffer'] = STATE['decoder_in_buffer'][-W:]
    STATE['H_channel'] = differentiable_channel(STATE['encoder_out'], STATE['decoder_in'], type=config.channel_derivative_type)

    if config.channel_derivative_type == 'linear':
        out =  STATE['decoder'](STATE['H_channel'] * STATE['encoder_out'])
    elif config.channel_derivative_type == 'ici_matrix':
        out =  STATE['decoder'](STATE['encoder_out'] @ STATE['H_channel'])
    # Store out in STATE to preserve computational graph
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
            # Track with WandB
            wandb.log({"train/loss" :batch_avg_loss.item()}, step=STATE['cycle_count'])
            STATE['optimizer'].zero_grad()
            batch_avg_loss.backward()
            STATE['optimizer'].step()
            # Clear accumulator
            STATE['loss_accumulator'].clear()

            if config.scheduler_type == "reduce_lr_on_plateu":
                STATE['scheduler'].step(batch_avg_loss)
            else:
                STATE['scheduler'].step()

            # Save final loss so that optuna can use it later
            with open("final_loss.txt", "w") as f:
                    f.write(str(batch_avg_loss.item()))


            wandb.log({"perf/time_for_ML": STATE['ML_time']}, step=STATE['cycle_count'])
            wandb.log({"perf/time_encode_to_decode": elapsed_time}, step=STATE['cycle_count'])
            # Percentage ML time
            wandb.log({"perf/percent_time_for_ML": STATE['ML_time'] / elapsed_time}, step=STATE['cycle_count'])
            if config.channel_derivative_type == "ici_matrix":
                wandb.log({"perf/Condition Number of Gram Matrix log_10": np.log10(STATE['gram_cond'].item())}, step=STATE['cycle_count'])

            # Check if need to cancel run early

            if STATE['batch_count'] > config.EARLY_STOP_PATIENCE:

                if batch_avg_loss > config.EARLY_STOP_THRESHOLD:
                    # Cancel training
                    msg = (
                        f"Early stopping triggered at batcg {STATE['batch_count']}! "
                        f"Batch avg loss {batch_avg_loss:.4f} exceeded threshold {config.EARLY_STOP_THRESHOLD}."
                    )
                    print(msg)
                    output = True

    if STATE['cycle_count'] % config.plot_frequency == 0:
        step = STATE['cycle_count']
        # Compute EVM grid (per symbol, per subcarrier)
        evm_grid = (true_symbols.real - predicted_symbols.real) ** 2 + (true_symbols.imag - predicted_symbols.imag) ** 2
        evm_grid = evm_grid.detach().cpu().numpy()  # Convert to numpy for plotting

        # Call the heatmap plot function
        plot_evm_heatmap(evm_grid, step=STATE['cycle_count'])

        log_constellation(step=step, freqs=STATE['frequencies'], evm_loss=evm_loss)

        # plot_SNR_vs_freq(step=step)

        log_encoder_frequency_sensitivity(step=STATE['cycle_count'], freqs=STATE['frequencies'])
        attn_weights = get_attention_map(STATE['encoder'], STATE['encoder_in'], STATE['frequencies'])
        log_attention_heatmap(attn_weights[0, 0], step=STATE['cycle_count'])  # [N, N] map for head 0
        plot_SNR_vs_freq(step=STATE['cycle_count'])

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

    STATE['cycle_count'] += 1
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


def get_attention_map(model, x: torch.Tensor, freqs: torch.Tensor):
    # Forward through embedding layers
    with torch.no_grad():
        symbol_embed = model.symbol_embed(x)
        freq_embed = model.frequency_embed(symbol_embed, freqs)

        # Use the first encoder layer only
        attn_layer = model.transformer.layers[0]  # type: nn.TransformerEncoderLayer
        mha = attn_layer.self_attn
        attn_output, attn_weights = mha(freq_embed, freq_embed, freq_embed, need_weights=True, average_attn_weights=False)
        return attn_weights  # Shape: [B, num_heads, N, N]

def plot_SNR_vs_freq(step: int, save_path: str = None):

    try:
        # Get frequencies
        data_frequencies = STATE['frequencies']

        # Get sent and received symbols
        received_symbols = STATE['received_symbols']
        received_symbols = torch.stack(received_symbols, dim=0) # [k, Nt, Nf]
        predicted_symbols = STATE['predicted_received_symbols']
        predicted_symbols = torch.stack(predicted_symbols, dim=0) # [k, Nt, Nf]

        # Create large tensors of shape [Nt * k, Nf]
        # Collapse k and Nt -> shape becomes [k * Nt, Nf]
        received_symbols = received_symbols.reshape(-1, received_symbols.shape[-1])  # [k * Nt, Nf]
        predicted_symbols = predicted_symbols.reshape(-1, predicted_symbols.shape[-1])  # [k * Nt, Nf]

        # Calculate EVM by freq
        
        num = torch.mean(torch.square(torch.abs(received_symbols - predicted_symbols)), dim=0) # [Nf]
        denom = torch.mean(torch.square(torch.abs(predicted_symbols)), dim=0)
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


        # --- Plot constellation diagram for last OFDM frame ---
        last_received = STATE['received_symbols'][-1]  # [Nt, Nf]
        last_predicted = STATE['predicted_received_symbols'][-1]  # [Nt, Nf]

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
        



def plot_evm_heatmap(evms: np.ndarray, step: int = 0, save_path: str = None):
    """
    Plots a heatmap of EVM loss for an [Nt, Nf] grid.
    Green = low loss, Red = high loss.
    """
    try:
        Nt, Nf = evms.shape
        fig, ax = plt.subplots(figsize=(10, 4))
        cmap = plt.get_cmap('RdYlGn_r')  # reversed so green is low, red is high
        im = ax.imshow(evms, aspect='auto', cmap=cmap, interpolation='nearest', origin='upper')
        fig.colorbar(im, ax=ax, label='EVM Loss')
        ax.set_title(f"EVM Loss Heatmap @ Step {step}")
        ax.set_xlabel("Subcarrier (Nf)")
        ax.set_ylabel("OFDM Symbol (Nt)")

        # Set y-axis to integer bins
        ax.set_yticks(np.arange(Nt))
        ax.set_ylim(Nt-0.5, -0.5)  # So 0 is at the top, Nt-1 at the bottom

        # No tight_layout if using constrained_layout elsewhere
        if save_path is not None:
            fig.savefig(save_path, dpi=150)
        os.makedirs("wandb_constellations", exist_ok=True)
        plot_path = f"wandb_constellations/ofdm_grid_step_{step}.png"
        fig.savefig(plot_path, dpi=150)
        wandb.log({"OFDM Grid": wandb.Image(plot_path)}, step=step)
        os.remove(plot_path)
        plt.close(fig)
    except Exception as e:
        print(f"Failed to plot constellation grid at step {step}: {e}")


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
        phase = np.angle(H_mean, deg=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(f"Channel Estimate Magnitude and Phase @ Step {step}", fontsize=14)

        ax1.plot(freqs, magnitude, label="|H(f)|", color='blue')
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True)

        ax2.plot(freqs, phase, label="Phase(H(f))", color='green')
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


def log_attention_heatmap(attn_weights: torch.Tensor, step: int):
    try:
        # Handle different possible shapes
        if attn_weights.ndim == 2:
            attn_map = attn_weights.detach().cpu().numpy()
        else:
            print(f"[Error] Unexpected attn_weights shape: {attn_weights.shape}")
            return

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(attn_map, cmap='viridis', aspect='auto')
        fig.colorbar(im, ax=ax, label='Attention Weight')
        ax.set_title(f"Self-Attention Map @ Step {step}")
        ax.set_xlabel("Key Subcarrier Index")
        ax.set_ylabel("Query Subcarrier Index")

        os.makedirs("wandb_constellations", exist_ok=True)
        plot_path = f"wandb_constellations/attention_step_{step}.png"
        fig.savefig(plot_path, dpi=150)
        wandb.log({"Attention Map": wandb.Image(plot_path)}, step=step)
        os.remove(plot_path)
        plt.close(fig)

    except Exception as e:
        print(f"[Error] Failed to log attention map at step {step}: {e}")


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