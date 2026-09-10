'''
ChannelModelGridSearch: runs a config-driven grid over channel-model
architectures and writes a self-describing, resumable run folder.

Layout produced under data/experiments/<name>_<timestamp>/:

    experiment_config.yaml   manifest: grid + dataset path + git hash + seed
    runs.jsonl               one line per finished run (the "run table")
    runs/<run_id>/
        config.yaml          this run's resolved params
        model.pt              checkpoint (adapter.save)
        metrics.json          eval metrics (per_burst_rrmse_pct + val_per_burst_rrmse_pct) + num_params + train_seconds
        history.csv          per-epoch train/val loss (omitted for closed-form models)
        plots/loss.png       train vs validation loss curve (trainable models only)
    summary/leaderboard.csv  aggregated view of runs.jsonl

run_id is a deterministic hash of the point, so a finished run (metrics.json
present) is skipped on re-run -> crash-resumable for free. See base.py for the
shared run-folder bookkeeping reused by EncoderDecoderGridSearch.

VAL_FRACTION (grid-wide) holds out a random slice of frames for validation;
channel models are then selected on val_per_burst_rrmse_pct (see select_channel_models).
SELECTED_RUN_IDS can pin exactly which runs advance to the encoder/decoder stage.
'''
import json
import math
from pathlib import Path

import json

import numpy as np
import torch
import yaml
from matplotlib.figure import Figure

from modules.grid_search.adapters import MODEL_REGISTRY
from modules.grid_search.base import GridSearchBase
from modules.grid_search.grid import expand_grid, resolve_runtime
from modules.utils import calculate_per_burst_rrmse_pct_loss, load_ofdm_dataset


class ChannelModelGridSearch(GridSearchBase):
    # keys consumed by the grid search itself, not forwarded to adapters as shared
    _GRID_CONTROL_KEYS = ("models", "VAL_FRACTION", "SELECTED_RUN_IDS",
                          "SELECTION_MODE", "PROB_SELECTION_KEY")

    def __init__(self,
                 grid_config: dict,
                 dataset_path,
                 experiments_dir=None,
                 device="cpu",
                 seed=0,
                 experiment_name="channel_models",
                 val_fraction=None,
                 run_prefix=None,
                 ):

        self.dataset_path = Path(dataset_path)
        self.val_fraction = float(grid_config.get("VAL_FRACTION", val_fraction or 0.0))
        self.eval_chunk_size = int(grid_config["EVAL_CHUNK_SIZE"])
        self.ofdm_config = None  # set in _prepare; needed for the frequency-resolved val plot

        points = expand_grid(grid_config["models"])
        # every CHANNEL_GRID_SEARCH key except grid-control settings is a grid-wide
        # setting (e.g. RECEPTIVE_FIELD) handed to every adapter via from_config(shared=...)
        shared_params = {k: v for k, v in grid_config.items() if k not in self._GRID_CONTROL_KEYS}
        super().__init__(points, grid_config, shared_params, experiments_dir, device, seed,
                          experiment_name, run_prefix=run_prefix, extra_manifest={
                              "dataset_path": str(self.dataset_path),
                              "val_fraction": self.val_fraction,
                          })

    @classmethod
    def from_experiment_config(cls,
                               config_path,
                               device=None,
                               seed=None,
                               experiment_name="channel_models",
                               experiments_dir=None,
                               dataset_path=None,
                               run_prefix=None,
                               ):
        '''Build the grid search from the unified end-to-end config.

        dataset_path overrides DATA_COLLECTION.DATASET_PATH from the config file;
        use this when the dataset was just created and the YAML still shows null.
        '''
        with open(Path(config_path), encoding="utf-8") as f:
            full = yaml.safe_load(f)

        grid_config = full["CHANNEL_GRID_SEARCH"]
        if dataset_path is None:
            dataset_path = full["DATA_COLLECTION"]["DATASET_PATH"]
        device, seed = resolve_runtime(full, device, seed)

        return cls(grid_config, dataset_path=dataset_path, device=device, seed=seed,
                   experiment_name=experiment_name, experiments_dir=experiments_dir,
                   run_prefix=run_prefix)

    # ------------------------------------------------------------- data/eval
    def _prepare(self, data=None):
        '''data: optional (X, Y) tuple; if None, loaded from self.dataset_path.
        Returns (X_train, Y_train, X_val, Y_val); the val tensors are None when
        VAL_FRACTION is 0.'''
        if data is not None:
            sent, received = data
        else:
            sent, received, config = load_ofdm_dataset(str(self.dataset_path), self.device)
            self.ofdm_config = config
            # channel model trains on the OFDM symbol only (CP + payload); drop the preamble
            preamble_length = sent.shape[1] - config.baseband_fft_length - config.cyclic_prefix_length
            sent, received = sent[:, preamble_length:], received[:, preamble_length:]
        return self._split_train_val(sent, received)

    def _split_train_val(self, X, Y):
        n = X.shape[0]
        n_val = int(round(n * self.val_fraction))
        if n_val == 0:
            return X, Y, None, None
        # deterministic shuffle so the same seed reproduces the same split
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]

    def _evaluate(self, adapter, X, Y) -> dict:
        '''Per-burst rRMSE over the whole split, accumulated one chunk at a time.'''
        weighted_rrmse = 0.0
        total_bursts = 0
        for start in range(0, X.shape[0], self.eval_chunk_size):
            sent_chunk = X[start:start + self.eval_chunk_size]
            received_chunk = Y[start:start + self.eval_chunk_size]

            predicted_chunk = adapter.predict(sent_chunk)
            if isinstance(predicted_chunk, tuple):  # TCN learn_noise: (noisy, mean, std, nu)
                predicted_chunk = predicted_chunk[1]
            received_chunk = received_chunk.to(predicted_chunk.device)

            if getattr(adapter, "exclude_warmup", False):
                warmup = adapter._warmup_slice(predicted_chunk.shape[-1])
                predicted_chunk = predicted_chunk[..., warmup:]
                received_chunk = received_chunk[..., warmup:]

            chunk_rrmse = calculate_per_burst_rrmse_pct_loss(received_chunk, predicted_chunk)
            weighted_rrmse += chunk_rrmse * sent_chunk.shape[0]
            total_bursts += sent_chunk.shape[0]

        return {"per_burst_rrmse_pct": weighted_rrmse / total_bursts}

    # ----------------------------------------------------------------- run
    def _run_point(self, point, run_dir, context) -> dict:
        X, Y, X_val, Y_val = context
        adapter = MODEL_REGISTRY[point["model"]].from_config(
            point["params"], self.device, shared=self.shared_params)

        history = adapter.fit(X, Y, X_val, Y_val)
        metrics = self._evaluate(adapter, X, Y)
        if X_val is not None:
            # selection metric: held-out validation per-burst rRMSE (see select_channel_models)
            metrics["val_per_burst_rrmse_pct"] = self._evaluate(adapter, X_val, Y_val)["per_burst_rrmse_pct"]
            # probabilistic (learn_noise) models also report validation NLL
            val_nll = adapter.val_nll(X_val, Y_val)
            if val_nll is not None:
                metrics["val_nll"] = val_nll
            # frequency-resolved validation error, aggregated into a final experiment plot
            if self.ofdm_config is not None:
                np.save(run_dir / "val_evm_vs_freq.npy",
                        self._val_evm_per_carrier(adapter, X_val, Y_val))
        metrics["num_params"] = int(adapter.num_params())

        # receptive field (TCN uses model attribute; GMP uses max memory + 1)
        model_type = point["model"]
        if model_type == "gmp":
            mem_lin = point["params"].get("memory_linear", 0)
            mem_nonlin = point["params"].get("memory_nonlinear", 0)
            metrics["receptive_field"] = max(mem_lin, mem_nonlin) + 1
        elif hasattr(adapter, "model") and hasattr(adapter.model, "receptive_field"):
            metrics["receptive_field"] = int(adapter.model.receptive_field)

        # distribution (for GMP always "none"; for TCN track the learned noise type)
        dist = point["params"].get("distribution", "none")
        metrics["distribution"] = dist

        adapter.save(run_dir / "model.pt")
        self._write_history(run_dir, history)
        return metrics

    def _val_evm_per_carrier(self, adapter, X_val, Y_val):
        '''Per-active-carrier EVM% of the model's mean prediction against the real received
        symbol on the held-out validation split, i.e. the frequency-resolved val error.'''
        cyclic_prefix_length = self.ofdm_config.cyclic_prefix_length
        active = self.ofdm_config.active_carrier_indices.cpu().numpy()

        def carrier_symbols(symbol_block):
            payload = symbol_block[:, cyclic_prefix_length:].detach().cpu().numpy()
            return np.fft.fft(payload, norm="ortho", axis=1)[:, active]

        residual_power_sum = np.zeros(len(active))
        reference_power_sum = np.zeros(len(active))
        for start in range(0, X_val.shape[0], self.eval_chunk_size):
            predicted_chunk = adapter.predict(X_val[start:start + self.eval_chunk_size])
            if isinstance(predicted_chunk, tuple):
                predicted_chunk = predicted_chunk[1]

            real = carrier_symbols(Y_val[start:start + self.eval_chunk_size])
            modelled = carrier_symbols(predicted_chunk)
            residual_power_sum += (np.abs(modelled - real) ** 2).sum(0)
            reference_power_sum += (np.abs(real) ** 2).sum(0)

        return np.sqrt(residual_power_sum / reference_power_sum) * 100

    def _plot_val_evm_vs_frequency(self, top_k=8):
        '''Final experiment plot: per-carrier validation EVM% vs frequency for the best
        channel models, overlaid. Saved once at the experiment root.'''
        if self.ofdm_config is None or not self.runs_jsonl.exists():
            return
        rows = [json.loads(line) for line in self.runs_jsonl.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r.get("val_per_burst_rrmse_pct") is not None]
        if not rows:
            return
        rows.sort(key=lambda r: r["val_per_burst_rrmse_pct"])
        freqs_mhz = self.ofdm_config.subcarrier_freqs_hz.cpu().numpy() / 1e6

        fig = Figure(figsize=(7, 4.5))
        ax = fig.subplots()
        for row in rows[:top_k]:
            evm_path = self.exp_dir / "runs" / row["run_id"] / "val_evm_vs_freq.npy"
            if not evm_path.exists():
                continue
            ax.plot(freqs_mhz, np.load(evm_path), marker=".", ms=3, lw=1.0,
                    label=f"{row['run_id']}  {row['val_per_burst_rrmse_pct']:.2f}%")
        ax.set_xlabel("subcarrier frequency (MHz)")
        ax.set_ylabel("validation EVM (%)")
        ax.set_title("Channel-model validation EVM vs frequency (best models)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(self.exp_dir / "val_evm_vs_frequency.png", dpi=150, bbox_inches="tight")

    def run(self, **prepare_kwargs):
        exp_dir = super().run(**prepare_kwargs)
        self._plot_val_evm_vs_frequency()
        return exp_dir


def select_channel_models(exp_dir, mode="best", metric="val_per_burst_rrmse_pct",
                          prob_metric="val_per_burst_rrmse_pct", run_ids=None):
    '''Pick channel models out of a finished ChannelModelGridSearch run for the
    encoder/decoder stage.

    run_ids: optional explicit list of run_ids to forward (config-level override);
        when given, exactly those runs are returned and `mode` is ignored.
    prob_metric: the ranking key for probabilistic (learn_noise) runs, e.g.
        "val_per_burst_rrmse_pct" (mean-prediction accuracy) or "val_nll"
        (likelihood); Nonprob runs always rank on `metric`.
    mode="best": the best run per channel form (model family split into
        prob/nonprob, e.g. "prob TCN" vs "nonprob TCN"), all ranked by held-out
        validation per-burst rRMSE (mean-prediction accuracy, which is what the
        E/D stage consumes), falling back through the legacy globally-normalized
        keys and then train rRMSE for experiments predating either the metric
        switch or a validation split.
    mode="best_per_size": like "best" but per (form, octave size bucket)
        round(log2(num_params)), so one winner per factor-of-2 parameter tier
        advances per form.
    mode="all": every finished run.'''
    exp_dir = Path(exp_dir)
    rows = [json.loads(line) for line in (exp_dir / "runs.jsonl").read_text().splitlines() if line.strip()]

    def score(row):
        for key in (metric, "val_rrmse_pct", "per_burst_rrmse_pct", "rrmse_pct"):
            if row.get(key) is not None:
                return row[key]
        return float("inf")

    def prob_score(row):
        for key in (prob_metric, "val_per_burst_rrmse_pct", "val_rrmse_pct", "per_burst_rrmse_pct", "rrmse_pct"):
            if row.get(key) is not None:
                return row[key]
        return float("inf")

    if run_ids:
        by_id = {row["run_id"]: row for row in rows}
        missing = [rid for rid in run_ids if rid not in by_id]
        if missing:
            raise ValueError(f"run_ids not found in {exp_dir}: {missing}")
        rows = [by_id[rid] for rid in run_ids]
    elif mode in ("best", "best_per_size"):
        best = {}
        for row in rows:
            is_prob = row.get("distribution", "none") not in (None, "none")
            key = (row["model"], "prob" if is_prob else "nonprob")
            if mode == "best_per_size":
                key = (*key, int(round(math.log2(row["num_params"]))))
            rank = prob_score if is_prob else score
            cur = best.get(key)
            if cur is None or rank(row) < rank(cur):
                best[key] = row
        rows = list(best.values())
    elif mode != "all":
        raise ValueError(f"unknown mode {mode!r}, expected 'all', 'best' or 'best_per_size'")

    selected = []
    for row in rows:
        run_dir = exp_dir / "runs" / row["run_id"]
        with open(run_dir / "config.yaml") as f:
            params = yaml.safe_load(f)["params"]
        selected.append({
            "run_id": row["run_id"],
            "model": row["model"],
            "params": params,
            "checkpoint": run_dir / "model.pt",
            "receptive_field": row.get("receptive_field"),
            "distribution": row.get("distribution", "none"),
        })
    return selected
