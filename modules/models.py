import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.studentT import StudentT
import matplotlib.pyplot as plt

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0
        )
        self.padding = (kernel_size - 1) * dilation
        self.relu = nn.ReLU()
        self.resample = None
        if in_channels != out_channels:
            self.resample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = F.pad(x, (self.padding, 0))
        out = self.conv(out)
        out = self.relu(out)
        if self.resample:
            x = self.resample(x)
        return out + x # residual connection

class TCN(nn.Module):
    def __init__(self, nlayers=3, dilation_base=2, num_taps=10, hidden_channels=32):
        super().__init__()
        layers = []
        in_channels = 1
        for i in range(nlayers):
            dilation = dilation_base ** i
            layers.append(
                TCNBlock(in_channels, hidden_channels, num_taps, dilation)
            )
            in_channels = hidden_channels
        self.tcn = nn.Sequential(*layers)
        self.readout = nn.Conv1d(hidden_channels, 1, kernel_size=1)

        # Calculate the total receptive field for the whole TCN stack
        self.receptive_field = 1
        for i in range(nlayers):
            dilation = dilation_base ** i
            self.receptive_field += (num_taps - 1) * dilation

    def forward(self, xin):
        x = xin.unsqueeze(1)    # [B,1,T]
        out = self.tcn(x)     # [B,H,T]
        out = self.readout(out).squeeze(1)
        out = out - out.mean(dim=1, keepdim=True)  # [B,T]
        return out
    

class TCN_channel(nn.Module):
    def __init__(self, nlayers=3, dilation_base=2, num_taps=10,
                 hidden_channels=32, learn_noise=False, gaussian=True):
        super().__init__()
        layers = []
        in_channels = 1
        for i in range(nlayers):
            dilation = dilation_base ** i
            layers.append(
                TCNBlock(in_channels, hidden_channels, num_taps, dilation)
            )
            in_channels = hidden_channels
        self.learn_noise = learn_noise
        self.tcn = nn.Sequential(*layers)
        if gaussian:
            self.readout = nn.Conv1d(hidden_channels, 2, kernel_size=1) # 2 channels mean | std
        else:
            self.readout = nn.Conv1d(hidden_channels, 3, kernel_size=1) # 3 channels mean | std | nu
        self.num_taps = num_taps
        self.gaussian = gaussian

        if not gaussian:
            with torch.no_grad():
                # Initialize nu bias towards Gaussian for stability
                self.readout.bias[2].fill_(48)


    def sample_student_t_pytorch(self, mean, std, nu):
        """
        Samples from a Student's t-distribution using PyTorch's built-in implementation.
        Uses rsample() to maintain gradients for 'std' and 'nu'.
        """
        nu = torch.clamp(nu, min=2.001) 
        std = torch.clamp(std, min=1e-6)
        dist = StudentT(df=nu, loc=mean, scale=std)
        return dist.rsample()
    

    def sample_student_t_mps(self, mean, std, nu):
        '''
        Wilson-Hilferty Approximation for chi^2 converted to scaled and shifted student t
        '''
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
        log_std_out = out[:, 1, :]
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
            nu_out = torch.zeros_like(mean_out)
        else:
            if xin.device.type == "mps":
                noisy_out = self.sample_student_t_mps(mean_out, std_out, nu_out)
            else:
                noisy_out = self.sample_student_t_pytorch(mean_out, std_out, nu_out)
            
        if self.learn_noise:
            return noisy_out, mean_out, std_out, nu_out
        else:
            return mean_out


class memory_polynomial_channel(nn.Module):
    def __init__(self,
                 weights,
                memory_linear,
                memory_nonlinear,
                nonlinearity_order,
                device
                 ):
        super().__init__()
        if weights:
            self.weights = torch.tensor(weights, device=device)
        else:
            self.weights = None
        self.memory_linear = memory_linear
        self.memory_nonlinear = memory_nonlinear
        self.nonlinearity_order = nonlinearity_order

    def _create_regressors(self, X):
        B, T = X.shape
        # Each example and target will get a matrix and column vector. All will be stacked
        # to form a A with shape [NxT, memory_linear + memory_nonlinearxnonlinear_order] regressor matrix
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
        return A
    
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
    
    def calculate_err(self, X, Y, plot=False):
        A = self._create_regressors(X)
        Q, R = torch.linalg.qr(A, mode='reduced')
        b = Y.flatten()
        # Project onto columns of Q
        g = torch.matmul(Q.T, b)
        total_variance = torch.sum(b ** 2)
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


    def fit(self, X, Y):
        A = self._create_regressors(X)
        Y_flat = Y.flatten()

        weights, residuals, rank, s = torch.linalg.lstsq(A, Y_flat)
        # print("Solved Weights:", weights)
        y_pred = A @ weights
        # Reshape back to (B, T) for analysis
        B, T = X.shape
        y_pred = y_pred.reshape(B, T)
        residuals = Y - y_pred
        self.weights = weights
        return weights, A, residuals

    def forward(self, X):
        A_x = self._create_regressors(X)
        B, T = X.shape
        y_pred = A_x @ self.weights
        y_pred = y_pred.reshape(B, T)
        return y_pred
