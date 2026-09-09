import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.studentT import StudentT
import matplotlib.pyplot as plt



ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
}


def make_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}, expected one of {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


def maybe_weight_norm(conv, enabled):
    '''Reparameterize a conv's weight as magnitude x direction (Salimans & Kingma 2016)
    when enabled. Normalizes the weights'''
    return torch.nn.utils.parametrizations.weight_norm(conv) if enabled else conv


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation, weight_norm=False):
        super().__init__()
        self.conv = maybe_weight_norm(nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0
        ), weight_norm)
        self.padding = (kernel_size - 1) * dilation
        self.activation = make_activation(activation)
        self.resample = None
        if in_channels != out_channels:
            self.resample = maybe_weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1), weight_norm)

    def forward(self, x):
        out = F.pad(x, (self.padding, 0))
        out = self.conv(out)
        out = self.activation(out)
        if self.resample:
            x = self.resample(x)
        return out + x # residual connection

class TCN(nn.Module):
    def __init__(self, nlayers=3, dilation_base=2, kernel_size=10, hidden_channels=32,
                 *, activation):
        super().__init__()
        layers = []
        in_channels = 1
        for i in range(nlayers):
            dilation = dilation_base ** i
            layers.append(
                TCNBlock(in_channels, hidden_channels, kernel_size, dilation, activation)
            )
            in_channels = hidden_channels
        self.tcn = nn.Sequential(*layers)
        self.readout = nn.Conv1d(hidden_channels, 1, kernel_size=1)

        # Calculate the total receptive field for the whole TCN stack
        self.receptive_field = 1
        for i in range(nlayers):
            dilation = dilation_base ** i
            self.receptive_field += (kernel_size - 1) * dilation

    def forward(self, xin):
        x = xin.unsqueeze(1)    # [B,1,T]
        out = self.tcn(x)     # [B,H,T]
        out = self.readout(out).squeeze(1)
        out = out - out.mean(dim=1, keepdim=True)  # [B,T]
        return out

    def get_num_params(self):
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
        return total_params

class TCN_channel(nn.Module):
    def __init__(self, nlayers=3, dilation_base=2, kernel_size=10,
                 hidden_channels=32, learn_noise=False, gaussian=True, weight_norm=False, *, activation):
        super().__init__()
        layers = []
        in_channels = 1
        for i in range(nlayers):
            dilation = dilation_base ** i
            layers.append(
                TCNBlock(in_channels, hidden_channels, kernel_size, dilation, activation, weight_norm=weight_norm)
            )
            in_channels = hidden_channels
        self.learn_noise = learn_noise
        self.tcn = nn.Sequential(*layers)
        if gaussian:
            self.readout = maybe_weight_norm(nn.Conv1d(hidden_channels, 2, kernel_size=1), weight_norm) # 2 channels mean | std
        else:
            self.readout = maybe_weight_norm(nn.Conv1d(hidden_channels, 3, kernel_size=1), weight_norm) # 3 channels mean | std | nu
        self.kernel_size = kernel_size
        self.gaussian = gaussian

        # Calculate the total receptive field for the whole TCN stack
        self.receptive_field = 1
        for i in range(nlayers):
            dilation = dilation_base ** i
            self.receptive_field += (kernel_size - 1) * dilation

        if not gaussian:
            with torch.no_grad():
                # Initialize nu bias towards Gaussian for stability
                self.readout.bias[2].fill_(48)


    def sample_student_t_pytorch(self, mean, std, nu):
        '''Sample a Student-t via PyTorch's built-in dist; rsample() keeps
        gradients flowing through std and nu.'''
        nu = torch.clamp(nu, min=2.001)
        std = torch.clamp(std, min=1e-6)
        dist = StudentT(df=nu, loc=mean, scale=std)
        return dist.rsample()


    def sample_student_t_mps(self, mean, std, nu):
        '''Wilson-Hilferty chi^2 approximation for a scaled/shifted Student-t
        (StudentT.rsample is unsupported on MPS).'''
        z = torch.randn_like(mean)
        z_chi = torch.randn_like(mean)
        chi2_approx = nu * (1 - 2/(9*nu) + z_chi * torch.sqrt(2/(9*nu))).pow(3)
        scale = torch.sqrt(nu / (chi2_approx + 1e-6))
        return mean + std * z * scale


    def forward(self, xin):
        x = xin.unsqueeze(1)    # [B,1,T]
        out = self.tcn(x)     # [B,H,T]
        out = self.readout(out) # [B, 3, T] mean | std | nu
        mean_out = out[:, 0, :]
        log_std_out = torch.clamp(out[:, 1, :], min=-15.0, max=10.0)
        std_out = torch.exp(log_std_out)
        if not self.gaussian:
            log_nu_out = out[:, 2, :]
            nu_out = torch.nn.functional.softplus(log_nu_out)
            nu_out = torch.clamp(nu_out, 2, 50) # nu between 2 and 50
        mean_out = mean_out - mean_out.mean(dim=1, keepdim=True)  # [B ,T]

        # # Produce noisy output
        if self.gaussian:
            z = torch.randn_like(mean_out)
            noisy_out = mean_out + std_out * z
            nu_out = torch.full_like(mean_out, float('inf')) # nu = inf for Gaussian
        else:
            if xin.device.type == "mps":
                noisy_out = self.sample_student_t_mps(mean_out, std_out, nu_out)
            else:
                noisy_out = self.sample_student_t_pytorch(mean_out, std_out, nu_out)

        if self.learn_noise:
            return noisy_out, mean_out, std_out, nu_out
        else:
            return mean_out

    def get_num_params(self):
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
        return total_params


def _complex_diag_scan(a_re, a_im, b_re, b_im):
    '''Parallel prefix scan of the diagonal recurrence x_k = a_k x_{k-1} + b_k
    (complex, elementwise) via Hillis-Steele doubling: O(T log T), no time loop.

    a, b have shape [*, T, N]. The scan composes (a_i, b_i) then (a_j, b_j) as
    (a_j a_i, a_j b_i + b_j) and returns the state sequence x_k as (re, im).
    Real and imaginary parts are kept separate for MPS, which lacks native
    complex support.
    '''
    T = a_re.shape[-2]
    d = 1
    while d < T:
        # shift by d and pad with the identity element (a=1, b=0)
        pa_re = F.pad(a_re, (0, 0, d, 0), value=1.0)[..., :T, :]
        pa_im = F.pad(a_im, (0, 0, d, 0), value=0.0)[..., :T, :]
        pb_re = F.pad(b_re, (0, 0, d, 0), value=0.0)[..., :T, :]
        pb_im = F.pad(b_im, (0, 0, d, 0), value=0.0)[..., :T, :]

        # compose: new_a = a·pa, new_b = a·pb + b
        new_a_re = a_re * pa_re - a_im * pa_im
        new_a_im = a_re * pa_im + a_im * pa_re
        new_b_re = a_re * pb_re - a_im * pb_im + b_re
        new_b_im = a_re * pb_im + a_im * pb_re + b_im

        a_re, a_im, b_re, b_im = new_a_re, new_a_im, new_b_re, new_b_im
        d *= 2
    return b_re, b_im


class LRU(nn.Module):
    '''Linear Recurrent Unit layer (Orvieto et al., 2023). A linear diagonal
    complex-state recurrence run with a parallel scan, plus a skip-connected
    output projection.

    N = state_dim (recurrent state size), H = model_dim (in/out feature size).
    Input/output are real sequences of shape [B, T, H].
    '''
    def __init__(self, state_dim, model_dim, r_min=0.0, r_max=1.0, max_phase=6.28):
        super().__init__()
        N, H = state_dim, model_dim
        self.N = N
        self.H = H

        # Lambda ~ uniform on the complex ring between r_min and r_max, phase in
        # [0, max_phase]; stored as its stable log-parametrisation (nu, theta).
        u1 = torch.rand(N)
        u2 = torch.rand(N)

        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(max_phase * u2)
        self.nu_log = nn.Parameter(nu_log)
        self.theta_log = nn.Parameter(theta_log)

        # Glorot-initialized input/output projections (real + imag parts).
        self.B_re = nn.Parameter(torch.randn(N, H) / math.sqrt(2 * H))
        self.B_im = nn.Parameter(torch.randn(N, H) / math.sqrt(2 * H))
        self.C_re = nn.Parameter(torch.randn(H, N) / math.sqrt(N))
        self.C_im = nn.Parameter(torch.randn(H, N) / math.sqrt(N))
        self.D = nn.Parameter(torch.randn(H))

        # Normalization of the recurrence
        diag_lambda = torch.exp(-torch.exp(nu_log) + 1j * torch.exp(theta_log))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2) + 1e-8)
        self.gamma_log = nn.Parameter(gamma_log)

    def forward(self, u):
        # u: [B, T, H]
        B, T, _ = u.shape
        N = self.N

        # Materialize diagonal lambda 
        modulus = torch.exp(-torch.exp(self.nu_log))          # [N]
        phase = torch.exp(self.theta_log)                     # [N]
        lam_re = modulus * torch.cos(phase)
        lam_im = modulus * torch.sin(phase)

        # Normalized input projection B_norm = (B_re + iB_im) * exp(gamma).
        gamma = torch.exp(self.gamma_log).unsqueeze(-1)       # [N, 1]
        Bn_re = self.B_re * gamma
        Bn_im = self.B_im * gamma

        # Bu_k = B_norm @ u_k for the whole sequence (u is real): [B, T, N].
        Bu_re = u @ Bn_re.t()
        Bu_im = u @ Bn_im.t()

        a_re = lam_re.view(1, 1, N).expand(1, T, N)
        a_im = lam_im.view(1, 1, N).expand(1, T, N)
        x_re, x_im = _complex_diag_scan(a_re, a_im, Bu_re, Bu_im)  # states x_k

        # y_k = Re(C @ x_k) + D * u_k.
        y = x_re @ self.C_re.t() - x_im @ self.C_im.t()       # [B, T, H]
        y = y + u * self.D
        return y


class LRUBlock(nn.Module):
    '''One residual LRU block: pre-norm => LRU => GELU => GLU'''
    def __init__(self, state_dim, model_dim, dropout=0.0,
                 r_min=0.0, r_max=1.0, max_phase=6.28):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.lru = LRU(state_dim, model_dim, r_min, r_max, max_phase)
        self.glu_w = nn.Linear(model_dim, model_dim)
        self.glu_v = nn.Linear(model_dim, model_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = self.norm(x)
        z = self.lru(z)
        z = self.drop(F.gelu(z))
        z = self.glu_w(z) * torch.sigmoid(self.glu_v(z))      # gated linear unit
        z = self.drop(z)
        return x + z                                          # residual connection


class LRU_channel(nn.Module):
    '''Stacked Linear Recurrent Unit channel model
    '''
    def __init__(self, state_dim=64, hidden_dim=64, n_layers=2, dropout=0.0,
                 r_min=0.0, r_max=1.0, max_phase=6.28,
                 learn_noise=False, gaussian=True):
        super().__init__()
        self.learn_noise = learn_noise
        self.gaussian = gaussian

        self.encoder = nn.Linear(1, hidden_dim)               # per-timestep 1 -> H
        self.blocks = nn.ModuleList([
            LRUBlock(state_dim, hidden_dim, dropout, r_min, r_max, max_phase)
            for _ in range(n_layers)
        ])
        # readout channels: mean | (mean, std) | (mean, std, nu)
        if not learn_noise:
            out_channels = 1
        elif gaussian:
            out_channels = 2
        else:
            out_channels = 3
        self.readout = nn.Linear(hidden_dim, out_channels)
        self.receptive_field = 1

        if learn_noise and not gaussian:
            with torch.no_grad():
                # Bias nu large so the Student-t starts near-Gaussian (stability).
                self.readout.bias[2].fill_(48)

    def sample_student_t_pytorch(self, mean, std, nu):
        '''Samples from a Student's t via PyTorch's built-in dist; rsample()
        keeps gradients flowing through std and nu.'''
        nu = torch.clamp(nu, min=2.001)
        std = torch.clamp(std, min=1e-6)
        dist = StudentT(df=nu, loc=mean, scale=std)
        return dist.rsample()

    def sample_student_t_mps(self, mean, std, nu):
        '''Wilson-Hilferty chi^2 approximation for a scaled/shifted Student-t
        (StudentT.rsample is unsupported on MPS).'''
        z = torch.randn_like(mean)
        z_chi = torch.randn_like(mean)
        chi2_approx = nu * (1 - 2 / (9 * nu) + z_chi * torch.sqrt(2 / (9 * nu))).pow(3)
        scale = torch.sqrt(nu / (chi2_approx + 1e-6))
        return mean + std * z * scale

    def forward(self, xin):
        # xin: [B, T]
        x = self.encoder(xin.unsqueeze(-1))                   # [B, T, H]
        for block in self.blocks:
            x = block(x)
        out = self.readout(x)                                 # [B, T, out_channels]

        mean_out = out[..., 0]                                # [B, T]
        mean_out = mean_out - mean_out.mean(dim=1, keepdim=True)
        if not self.learn_noise:
            return mean_out

        std_out = torch.exp(torch.clamp(out[..., 1], min=-15.0, max=10.0))
        if self.gaussian:
            z = torch.randn_like(mean_out)
            noisy_out = mean_out + std_out * z
            nu_out = torch.full_like(mean_out, float('inf'))  # nu = inf for Gaussian
        else:
            nu_out = torch.nn.functional.softplus(out[..., 2])
            nu_out = torch.clamp(nu_out, 2, 50)               # nu between 2 and 50
            if xin.device.type == "mps":
                noisy_out = self.sample_student_t_mps(mean_out, std_out, nu_out)
            else:
                noisy_out = self.sample_student_t_pytorch(mean_out, std_out, nu_out)

        return noisy_out, mean_out, std_out, nu_out

    def get_num_params(self):
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
        return total_params


class LSTM_channel(nn.Module):
    '''Stacked LSTM channel model
    '''
    def __init__(self, hidden_dim=64, n_layers=2, dropout=0.0,
                 learn_noise=False, gaussian=True):
        super().__init__()
        self.learn_noise = learn_noise
        self.gaussian = gaussian

        self.encoder = nn.Linear(1, hidden_dim)               # per-timestep 1 -> H
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        out_channels = 2 if (learn_noise and gaussian) else 1
        self.readout = nn.Linear(hidden_dim, out_channels)
        self.receptive_field = 1

    def forward(self, xin):
        # xin: [B, T]
        x = self.encoder(xin.unsqueeze(-1))                   # [B, T, H]
        x, _ = self.lstm(x)                                   # [B, T, H]
        out = self.readout(x)                                 # [B, T, out_channels]

        mean_out = out[..., 0]                                # [B, T]
        mean_out = mean_out - mean_out.mean(dim=1, keepdim=True)
        if not self.learn_noise:
            return mean_out

        std_out = torch.exp(torch.clamp(out[..., 1], min=-15.0, max=10.0))
        z = torch.randn_like(mean_out)
        noisy_out = mean_out + std_out * z
        nu_out = torch.full_like(mean_out, float('inf'))      # nu = inf for Gaussian
        return noisy_out, mean_out, std_out, nu_out

    def get_num_params(self):
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
        return total_params


class TTHNet_channel(nn.Module):
    '''Two tributaries heterogeneous NN channel model. A linear tributary and a
    ReLU tributary over a symmetric (non-causal) window of samples,
    summed.
    '''
    def __init__(self, window, hidden1, hidden2,
                 learn_noise=False, gaussian=True):
        super().__init__()
        self.learn_noise = learn_noise
        self.gaussian = gaussian
        self.receptive_field = window
        pad = window // 2

        self.lin1 = nn.Conv1d(1, hidden1, kernel_size=window, padding=pad)
        self.lin2 = nn.Conv1d(hidden1, hidden2, kernel_size=1)
        self.nl1 = nn.Conv1d(1, hidden1, kernel_size=window, padding=pad)
        self.nl2 = nn.Conv1d(hidden1, hidden2, kernel_size=1)

        out_channels = 2 if (learn_noise and gaussian) else 1
        self.readout = nn.Conv1d(hidden2, out_channels, kernel_size=1)

    def forward(self, xin):
        x = xin.unsqueeze(1)                                  # [B, 1, T]

        lin = self.lin2(self.lin1(x))
        nl = F.relu(self.nl2(F.relu(self.nl1(x))))
        out = self.readout(lin + nl)                          # [B, out_channels, T]

        mean_out = out[:, 0, :]                                # [B, T]
        mean_out = mean_out - mean_out.mean(dim=1, keepdim=True)
        if not self.learn_noise:
            return mean_out

        std_out = torch.exp(torch.clamp(out[:, 1, :], min=-15.0, max=10.0))
        z = torch.randn_like(mean_out)
        noisy_out = mean_out + std_out * z
        nu_out = torch.full_like(mean_out, float('inf'))      # nu = inf for Gaussian
        return noisy_out, mean_out, std_out, nu_out

    def get_num_params(self):
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
        return total_params


class memory_polynomial_channel(nn.Module):
    def __init__(self,
                 weights,
                memory_linear,
                memory_nonlinear,
                nonlinearity_order,
                device
                 ):
        super().__init__()
        if device == torch.device('mps'):
            print("MPS not supported. . . switching to CPU")
            device = torch.device('cpu')
        if weights:
            self.weights = torch.tensor(weights, device=device)
        else:
            self.weights = None
        self.memory_linear = memory_linear
        self.memory_nonlinear = memory_nonlinear
        self.nonlinearity_order = nonlinearity_order
        self.device = device

    def get_num_regressors(self):
        return (self.memory_linear + 1) + (self.memory_nonlinear + 1) * (self.nonlinearity_order - 1)

    def _iter_feature_chunks(self, X, Y, batch_chunk):
        # Chunk over the BATCH dim so each chunk holds complete sequences
        B, T = X.shape
        for b0 in range(0, B, batch_chunk):
            Xb = X[b0:b0 + batch_chunk]                 # [bc, T]
            A_blk = self._create_regressors(Xb)         # [bc*T, P] only this chunk lives in memory
            y_blk = Y[b0:b0 + batch_chunk].reshape(-1)  # [bc*T]
            yield A_blk, y_blk

    def _fit_normal_eqs(self, X, Y, batch_chunk=8, ridge=0.0):
        P = self.get_num_regressors()
        G = torch.zeros(P, P, dtype=torch.float64, device=self.device)
        c = torch.zeros(P,    dtype=torch.float64, device=self.device)
        for A_blk, y_blk in self._iter_feature_chunks(X, Y, batch_chunk):
            A_blk = A_blk.double()
            y_blk = y_blk.double()
            G += A_blk.T @ A_blk
            c += A_blk.T @ y_blk
            del A_blk, y_blk            # release the chunk before building the next one
        G += ridge * torch.eye(P, dtype=torch.float64, device=self.device)
        self.weights = torch.linalg.solve(G, c)

    def _create_regressors(self, X):
        B, T = X.shape
        # Build the regressor matrix A of shape [B*T, num_regressors].
        batched_regressor_cols = []
        num_regressors = (
            (self.memory_linear + 1) +
            (self.memory_nonlinear + 1) * (self.nonlinearity_order - 1)
        )
        regressor_length = T * B
        for i in range(self.memory_linear + 1):
            X_shifted = torch.roll(X, i, dims=1)
            X_shifted[:, :i] = 0.0
            batched_regressor_cols.append(X_shifted)

        for k in range(2, self.nonlinearity_order + 1):
            for j in range(self.memory_nonlinear + 1):
                X_shifted = torch.roll(X, j, dims=1)
                X_shifted[:, :j] = 0.0
                batched_regressor_cols.append(torch.pow(X_shifted, k))

        stack = torch.stack(batched_regressor_cols) # [features, B, T]
        stack = stack.permute(1, 2, 0) # [B, T, freatures]
        A = stack.reshape(regressor_length, num_regressors)
        return A.to(self.device)

    def show_terms(self, plot=False):
        weights = self.weights.detach().cpu()
        terms = []
        linear_weights = []
        idx = 0
        for i in range(self.memory_linear + 1):
            terms.append(f"x[{-i}]")
            linear_weights.append(weights[idx].item())
            idx += 1
        if plot:
            plt.plot(linear_weights)
            plt.title("Plot of Linear Weights vs. Memory Length")
            plt.xlabel("Memory Tap")
            plt.ylabel("Weight Value")
            plt.show()

        for k in range(2, self.nonlinearity_order + 1):
            k_th_weights = []
            for j in range(self.memory_nonlinear + 1):
                terms.append(f"x[{-j}]^{k}")
                k_th_weights.append(weights[idx].item())
                idx += 1

            if plot:
                plt.plot(k_th_weights)
                plt.title(f"Plot of Weights Order {k} vs. Memory Length")
                plt.xlabel("Memory Tap")
                plt.ylabel("Weight Value")
                plt.show()

        weights = None
        if self.weights is not None:
            weights = self.weights.detach().cpu().tolist()

        return terms, weights

    @torch.no_grad()
    def calculate_err(self, X, Y, batch_chunk=8, plot=False):
        X = X.to(self.device)
        Y = Y.to(self.device)
        n_regressors = self.get_num_regressors()
        G = torch.zeros(n_regressors, n_regressors, dtype=torch.float64, device=self.device)
        c = torch.zeros(n_regressors, dtype=torch.float64, device=self.device)
        total_variance = torch.zeros((), dtype=torch.float64, device=self.device)
        for A_blk, y_blk in self._iter_feature_chunks(X, Y, batch_chunk):
            A_blk = A_blk.double()
            y_blk = y_blk.double()
            G += A_blk.T @ A_blk
            c += A_blk.T @ y_blk
            total_variance += (y_blk * y_blk).sum()
            del A_blk, y_blk
        # Cholesky G = R^T R, then solve R^T g = c for g = Q^T b. Each g_i^2 is a
        # regressor's variance contribution; ERR_i = g_i^2 / total_variance * 100.
        R = torch.linalg.cholesky(G, upper=True)
        g = torch.linalg.solve_triangular(R.T, c.unsqueeze(-1), upper=False).squeeze(-1)
        component_variances = g ** 2
        terms, _ = self.show_terms(plot=False)
        ERR_values = (component_variances / total_variance) * 100
        err_list = ERR_values.cpu().tolist()

        num_linear = self.memory_linear + 1
        total_linear_err = torch.sum(ERR_values[:num_linear]).item()
        total_nonlinear_err = torch.sum(ERR_values[num_linear:]).item()
        ranked_data = list(zip(terms, err_list))
        ranked_data.sort(key=lambda x: x[1], reverse=True) # sort by ERR magnitude
        if plot:
            print("-" * 50)
            print(f"{'Rank':<5} | {'Term String':<20} | {'ERR (%)':<15}")
            print("-" * 50)

            cumulative_err = 0.0
            for i, (term, err) in enumerate(ranked_data):
                cumulative_err += err
                print(f"{i+1:<5} | {term:<20} | {err:.6f}%")

            print("-" * 50)
            print(f"Total Variance Explained: {cumulative_err:.4f}%")
            print(f"  > Linear Contribution:    {total_linear_err:.4f}%")
            print(f"  > Nonlinear Contribution: {total_nonlinear_err:.4f}%")
            print("-" * 50)
        return terms, ERR_values


    def fit(self, X, Y, batch_chunk=8, ridge=0.0):
        X = X.to(self.device); Y = Y.to(self.device)
        self._fit_normal_eqs(X, Y, batch_chunk=batch_chunk, ridge=ridge)
        self.weights = self.weights.to(X.dtype)
        return self.weights

    @torch.no_grad()
    def predict(self, X, batch_chunk=8):     # chunked forward
        X = X.to(self.device)
        B, T = X.shape
        out = torch.empty(B, T, dtype=self.weights.dtype, device=self.device)
        for b0 in range(0, B, batch_chunk):
            A_blk = self._create_regressors(X[b0:b0 + batch_chunk])
            out[b0:b0 + batch_chunk] = (A_blk @ self.weights).reshape(-1, T)
            del A_blk
        return out

    def forward(self, X):
        A_x = self._create_regressors(X)
        B, T = X.shape
        y_pred = A_x @ self.weights
        y_pred = y_pred.reshape(B, T)
        return y_pred

    def get_num_params(self):
        assert self.weights is not None, "Model must be fitted before getting number of parameters."
        return self.weights.numel()
    

class GeneralizedMemoryPolynomial(memory_polynomial_channel):
    def __init__(self, weights, memory_linear, memory_nonlinear, nonlinearity_order, cross_term_depth, device):
        super().__init__(weights, memory_linear, memory_nonlinear, nonlinearity_order, device)
        self.cross_term_depth = cross_term_depth

    def _iter_cross_shifts(self):
        """Yield (k, j, d, shift, kind) for every cross-term, applying the causal
        guard on leading terms."""
        for k in range(2, self.nonlinearity_order + 1):
            for j in range(self.memory_nonlinear + 1):
                for d in range(1, self.cross_term_depth + 1):
                    # lagging: powered factor further in the past
                    yield (k, j, d, j + d, "lag")
                    # leading: powered factor nearer the present, causal only
                    if j - d >= 0:
                        yield (k, j, d, j - d, "lead")

    def get_num_regressors(self):
        n_cross = sum(1 for _ in self._iter_cross_shifts())
        return (
            (self.memory_linear + 1)
            + (self.memory_nonlinear + 1) * (self.nonlinearity_order - 1)
            + n_cross
        )

    def _create_regressors(self, X):
        B, T = X.shape
        # Build the regressor matrix A of shape [B*T, num_regressors].

        X_powers = {k: torch.pow(X, k) for k in range(1, self.nonlinearity_order + 1)}
        batched_regressor_cols = []
        num_regressors = self.get_num_regressors()
        regressor_length = T * B

        # Linear terms
        for i in range(self.memory_linear + 1):
            X_shifted = torch.roll(X, i, dims=1)
            X_shifted[:, :i] = 0.0
            batched_regressor_cols.append(X_shifted)

        # Aligned (diagonal) nonlinear terms
        for k in range(2, self.nonlinearity_order + 1):
            X_pow_k = X_powers[k]
            for j in range(self.memory_nonlinear + 1):
                X_shifted_pow = torch.roll(X_pow_k, j, dims=1)
                X_shifted_pow[:, :j] = 0.0
                batched_regressor_cols.append(X_shifted_pow)

        # Cross-terms (lagging + leading), driven by the shared iterator
        for k, j, d, shift, kind in self._iter_cross_shifts():
            X_pow_k = X_powers[k - 1]
            X_shifted_base = torch.roll(X, j, dims=1)
            X_shifted_base[:, :j] = 0.0
            X_shifted_pow = torch.roll(X_pow_k, shift, dims=1)
            X_shifted_pow[:, :shift] = 0.0
            batched_regressor_cols.append(X_shifted_pow * X_shifted_base)

        stack = torch.stack(batched_regressor_cols)  # [features, B, T]
        stack = stack.permute(1, 2, 0)               # [B, T, features]
        A = stack.reshape(regressor_length, num_regressors)
        return A.to(self.device)

    def show_terms(self, plot=False):
        weights = self.weights.detach().cpu()
        terms = []
        linear_weights = []
        idx = 0

        # Linear
        for i in range(self.memory_linear + 1):
            terms.append(f"x[{-i}]")
            linear_weights.append(weights[idx].item())
            idx += 1
        if plot:
            plt.plot(linear_weights)
            plt.title("Plot of Linear Weights vs. Memory Length")
            plt.xlabel("Memory Tap")
            plt.ylabel("Weight Value")
            plt.show()

        # Aligned nonlinear
        for k in range(2, self.nonlinearity_order + 1):
            k_th_weights = []
            for j in range(self.memory_nonlinear + 1):
                terms.append(f"x[{-j}]^{k}")
                k_th_weights.append(weights[idx].item())
                idx += 1
            if plot:
                plt.plot(k_th_weights)
                plt.title(f"Plot of Weights Order {k} vs. Memory Length")
                plt.xlabel("Memory Tap")
                plt.ylabel("Weight Value")
                plt.show()

        # Cross-terms (must use the same iterator as _create_regressors)
        for k, j, d, shift, kind in self._iter_cross_shifts():
            terms.append(f"x[{-j}] * x[{-shift}]^{k - 1}  ({kind})")
            idx += 1

        weights = None
        if self.weights is not None:
            weights = self.weights.detach().cpu().tolist()

        return terms, weights