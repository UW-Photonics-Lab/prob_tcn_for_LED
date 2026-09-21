'''
Model adapters + registry for the channel-model grid search.

Each adapter wraps one model family from modules.models behind a single
interface so the grid search never needs to know how a given model trains
(TCN = iterative SGD, GMP = closed-form least squares).
'''
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from modules.models import (
    TCN_channel, GeneralizedMemoryPolynomial, LRU_channel, LSTM_channel, TTHNet_channel,
)


_DIST_TO_FLAGS = {
    "none":       {"learn_noise": False, "gaussian": True},
    "gaussian":   {"learn_noise": True,  "gaussian": True},
    "students_t": {"learn_noise": True,  "gaussian": False},
}


def _burst_power_weights(sent_bursts):
    '''Per-burst importance weights 1/P_b (P_b = mean(x²), the specified symbol
    power).'''
    return 1.0 / (sent_bursts ** 2).mean(dim=-1)


def _power_weighted_beta_nll_reduce(per_element_nll, std, beta, burst_weights):
    weighted_nll = burst_weights[:, None] * per_element_nll
    if not beta:
        return torch.mean(weighted_nll)
    return torch.mean((std.detach() ** (2.0 * beta)) * weighted_nll)


def power_weighted_gaussian_nll(residual, std, burst_weights, beta=0.0):
    per_element_nll = 0.5 * torch.log(2 * torch.pi * std ** 2) + 0.5 * (residual / std) ** 2
    return _power_weighted_beta_nll_reduce(per_element_nll, std, beta, burst_weights)


def power_weighted_students_t_loss(residual, std, nu, burst_weights, beta=0.0):
    z = residual / std
    term1 = (-torch.lgamma((nu + 1) / 2) + 0.5 * torch.log(torch.pi * nu)
             + torch.lgamma(nu / 2) + torch.log(std + 1e-8))
    term2 = ((nu + 1) / 2) * torch.log(1 + torch.square(z) / nu + 1e-8)
    return _power_weighted_beta_nll_reduce(term1 + term2, std, beta, burst_weights)


@runtime_checkable
class ChannelModel(Protocol):
    name: str

    @classmethod
    def from_config(cls, params: dict, device: str, shared: dict = None) -> "ChannelModel": ...

    @classmethod
    def load(cls, params: dict, checkpoint: Path, device: str, shared: dict = None) -> "ChannelModel":
        '''Rebuild a trained adapter from a checkpoint written by save().'''
        ...

    def fit(self, X, Y, X_val=None, Y_val=None) -> dict:
        '''Train and return a history dict (e.g. {"loss": [...], "val_loss": [...]});
        {} if closed-form. X_val/Y_val are optional held-out validation tensors used
        to record a per-epoch validation loss for trainable models.'''
        ...

    def predict(self, X): ...

    def val_nll(self, X, Y):
        '''Validation negative log-likelihood for probabilistic models; None for
        deterministic ones (used to add `val_nll` to the metrics).'''
        ...

    def num_params(self) -> int: ...

    def save(self, path: Path) -> None: ...


class TCNAdapter:
    name = "tcn"
    ARCH_KEYS = ("nlayers", "dilation_base", "kernel_size", "hidden_channels", "distribution", "learn_noise", "gaussian", "activation", "weight_norm")
    TRAIN_KEYS = ("epochs", "lr", "batch_size", "factor", "patience", "min_lr", "beta_nll")

    def __init__(self, model: TCN_channel, train_params: dict, device: str, shared: dict = None):
        self.model = model
        self.train_params = train_params
        self.device = device
        self.shared = shared or {}
        self.beta_nll = float(self.train_params.get("beta_nll", self.shared.get("BETA_NLL", 0.0)))
        self.eval_chunk_size = int(self.shared["EVAL_CHUNK_SIZE"])

    @classmethod
    def from_config(cls, params: dict, device: str, shared: dict = None) -> "TCNAdapter":
        arch = {k: params[k] for k in cls.ARCH_KEYS if k in params}
        if "distribution" in arch:
            arch.update(_DIST_TO_FLAGS[arch.pop("distribution")])

        model = TCN_channel(**arch).to(device)
        train_params = {k: params[k] for k in cls.TRAIN_KEYS if k in params}
        return cls(model, train_params, device, shared=shared)

    @classmethod
    def load(cls, params: dict, checkpoint: Path, device: str, shared: dict = None) -> "TCNAdapter":
        adapter = cls.from_config(params, device, shared=shared)
        adapter.model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        adapter.model.eval()
        return adapter

    def _loss(self, xb, yb, beta=None):
        '''The training objective on one batch: Gaussian/Student-t NLL when the
        model learns noise, MSE otherwise, each weighted per burst by 1/P_b so
        low-power bursts are not discounted (see _burst_power_weights). beta
        overrides self.beta_nll (used by _val_loss to report the true beta=0 NLL).'''
        if beta is None:
            beta = self.beta_nll
        burst_weights = _burst_power_weights(xb)
        if self.model.learn_noise:
            _, y_pred, y_pred_std, y_pred_nu = self.model(xb)
            residual = yb - y_pred
            if self.model.gaussian:
                return power_weighted_gaussian_nll(residual, y_pred_std, burst_weights, beta=beta)
            return power_weighted_students_t_loss(residual, y_pred_std, y_pred_nu, burst_weights,
                                                  beta=beta)
        squared_error = (self.model(xb) - yb) ** 2
        return torch.mean(burst_weights[:, None] * squared_error)

    def _val_loss(self, X_val, Y_val) -> float:
        '''True power-weighted NLL (beta=0), NOT the beta-NLL training value: the
        beta-weighted number is sigma-scale-dependent and rises as sigma shrinks,
        which misleads the LR scheduler and cross-run val_nll comparisons.

        Evaluated in chunks and recombined as a burst-count-weighted mean'''
        self.model.eval()
        total_loss = 0.0
        total_bursts = 0
        with torch.no_grad():
            for start in range(0, X_val.shape[0], self.eval_chunk_size):
                sent_chunk = X_val[start:start + self.eval_chunk_size]
                received_chunk = Y_val[start:start + self.eval_chunk_size]
                total_loss += self._loss(sent_chunk, received_chunk, beta=0).item() * sent_chunk.shape[0]
                total_bursts += sent_chunk.shape[0]
        self.model.train()
        return total_loss / total_bursts

    def val_nll(self, X, Y):
        '''Validation negative log-likelihood for probabilistic (learn_noise)
        models; None otherwise (plain-MSE TCNs have no likelihood).'''
        if not self.model.learn_noise:
            return None
        return self._val_loss(X.to(self.device), Y.to(self.device))

    def fit(self, X, Y, X_val=None, Y_val=None) -> dict:
        X, Y = X.to(self.device), Y.to(self.device)
        has_val = X_val is not None
        if has_val:
            X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)
        epochs = self.train_params["epochs"]
        batch_size = self.train_params["batch_size"]
        loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.train_params["lr"]))

        scheduler = None
        if "patience" in self.train_params:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min",
                factor=float(self.train_params.get("factor", 0.5)),
                patience=int(self.train_params["patience"]),
                min_lr=float(self.train_params.get("min_lr", 1e-6)),
            )

        grad_clip = float(self.train_params.get("grad_clip", 1.0))

        self.model.train()
        history = {"loss": [], "lr": []}
        if self.model.learn_noise:
            history["train_nll"] = []  # true beta=0 NLL, same scale as val_loss
        if has_val:
            history["val_loss"] = []
        skipped_batches = 0
        best_val_loss = float("inf")
        best_state = None
        for _ in range(epochs):
            epoch_loss, n_batches = 0.0, 0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = self._loss(xb, yb)
                if not torch.isfinite(loss):
                    skipped_batches += 1
                    continue
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                if not torch.isfinite(grad_norm):
                    skipped_batches += 1
                    continue

                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            train_loss = epoch_loss / max(n_batches, 1)
            history["loss"].append(train_loss)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            if self.model.learn_noise:
                history["train_nll"].append(self._val_loss(X, Y))
            if has_val:
                val_loss = self._val_loss(X_val, Y_val)
                history["val_loss"].append(val_loss)
            if scheduler is not None:
                scheduler.step(val_loss if has_val else train_loss)

            # keep the best-monitored weights so save() writes the optimum rather
            # than the final epoch
            monitored_loss = val_loss if has_val else train_loss
            if monitored_loss < best_val_loss:
                best_val_loss = monitored_loss
                best_state = {name: tensor.detach().cpu().clone()
                              for name, tensor in self.model.state_dict().items()}

        if skipped_batches:
            print(f"    [fit] skipped {skipped_batches} non-finite batches (grad_clip={grad_clip})")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def predict(self, X):
        '''Chunked forward pass'''
        self.model.eval()
        chunk_outputs = []
        with torch.no_grad():
            for start in range(0, X.shape[0], self.eval_chunk_size):
                chunk_outputs.append(self.model(X[start:start + self.eval_chunk_size].to(self.device)))

        if isinstance(chunk_outputs[0], tuple):
            return tuple(torch.cat(parts) for parts in zip(*chunk_outputs))
        return torch.cat(chunk_outputs)

    def num_params(self) -> int:
        return self.model.get_num_params()

    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)


class LRUAdapter(TCNAdapter):
    '''Linear Recurrent Unit (deep linear-RNN / state-space) baseline.

    Trains exactly like the TCN (same AdamW loop, Gaussian/Student-t NLL or MSE
    objective, and probabilistic `distribution` switch), so it reuses everything
    on TCNAdapter and only swaps in the LRU_channel construction. LRU_channel
    shares TCN_channel's forward contract (mean, or (noisy, mean, std, nu)) and
    exposes `receptive_field` (=1: an RNN has no fixed lookback window).
    '''
    name = "lru"
    ARCH_KEYS = ("state_dim", "hidden_dim", "n_layers", "dropout",
                 "r_min", "r_max", "max_phase",
                 "distribution", "learn_noise", "gaussian")
    # TRAIN_KEYS inherited from TCNAdapter (epochs, lr, batch_size, factor, patience, min_lr)

    @classmethod
    def from_config(cls, params: dict, device: str, shared: dict = None) -> "LRUAdapter":
        arch = {k: params[k] for k in cls.ARCH_KEYS if k in params}
        if "distribution" in arch:
            arch.update(_DIST_TO_FLAGS[arch.pop("distribution")])
        model = LRU_channel(**arch).to(device)
        train_params = {k: params[k] for k in cls.TRAIN_KEYS if k in params}
        return cls(model, train_params, device, shared=shared)


class LSTMAdapter(TCNAdapter):
    '''Stacked-LSTM channel baseline. Trains like the TCN; swaps in LSTM_channel.'''
    name = "lstm"
    ARCH_KEYS = ("hidden_dim", "n_layers", "dropout",
                 "distribution", "learn_noise", "gaussian")

    @classmethod
    def from_config(cls, params: dict, device: str, shared: dict = None) -> "LSTMAdapter":
        arch = {k: params[k] for k in cls.ARCH_KEYS if k in params}
        if "distribution" in arch:
            arch.update(_DIST_TO_FLAGS[arch.pop("distribution")])
        model = LSTM_channel(**arch).to(device)
        train_params = {k: params[k] for k in cls.TRAIN_KEYS if k in params}
        return cls(model, train_params, device, shared=shared)


class TTHNetAdapter(TCNAdapter):
    '''Two tributaries heterogeneous NN baseline. Trains like the TCN; swaps in TTHNet_channel.'''
    name = "tthnet"
    ARCH_KEYS = ("window", "hidden1", "hidden2",
                 "distribution", "learn_noise", "gaussian")

    @classmethod
    def from_config(cls, params: dict, device: str, shared: dict = None) -> "TTHNetAdapter":
        arch = {k: params[k] for k in cls.ARCH_KEYS if k in params}
        if "distribution" in arch:
            arch.update(_DIST_TO_FLAGS[arch.pop("distribution")])
        model = TTHNet_channel(**arch).to(device)
        train_params = {k: params[k] for k in cls.TRAIN_KEYS if k in params}
        return cls(model, train_params, device, shared=shared)


class GMPAdapter:
    name = "gmp"
    ARCH_KEYS = ("memory_linear", "memory_nonlinear", "nonlinearity_order", "cross_term_depth")
    FIT_KEYS = ("ridge", "batch_chunk")

    def __init__(self, model: GeneralizedMemoryPolynomial, fit_params: dict, device: str, shared: dict = None):
        self.model = model
        self.fit_params = fit_params
        self.device = device
        self.shared = shared or {}
        self.eval_chunk_size = int(self.shared["EVAL_CHUNK_SIZE"])

    @classmethod
    def from_config(cls, params: dict, device: str, shared: dict = None) -> "GMPAdapter":
        model = GeneralizedMemoryPolynomial(
            weights=None,
            memory_linear=params["memory_linear"],
            memory_nonlinear=params["memory_nonlinear"],
            nonlinearity_order=params["nonlinearity_order"],
            cross_term_depth=params["cross_term_depth"],
            device=torch.device(device),
        )
        fit_params = {k: params[k] for k in cls.FIT_KEYS if k in params}
        return cls(model, fit_params, device, shared=shared)

    @classmethod
    def load(cls, params: dict, checkpoint: Path, device: str, shared: dict = None) -> "GMPAdapter":
        adapter = cls.from_config(params, device, shared=shared)
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        adapter.model.weights = ckpt["weights"]
        return adapter

    def fit(self, X, Y, X_val=None, Y_val=None) -> dict:
        self.model.fit(X, Y, **self.fit_params)
        return {}  # closed-form: no per-epoch history (val loss measured by the grid search)

    def val_nll(self, X, Y):
        return None  # deterministic least-squares fit, no likelihood

    def predict(self, X):
        return self.model.predict(X)

    def num_params(self) -> int:
        return self.model.get_num_regressors()

    def save(self, path: Path) -> None:
        torch.save({"weights": self.model.weights, "fit_params": self.fit_params}, path)

MODEL_REGISTRY = {
    TCNAdapter.name: TCNAdapter,
    LRUAdapter.name: LRUAdapter,
    LSTMAdapter.name: LSTMAdapter,
    TTHNetAdapter.name: TTHNetAdapter,
    GMPAdapter.name: GMPAdapter,
}
