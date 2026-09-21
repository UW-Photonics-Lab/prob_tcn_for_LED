"""
Aggregate multi-DC-offset results and generate paper figures.

Edit RUN_CONFIGS and PLOT_PATH at the top, then:
    cd <repo_root>/experiments
    python summarize_results.py
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import seaborn as sns
from scipy import stats
import zarr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.grid_search.adapters import MODEL_REGISTRY
from modules.utils import calculate_per_burst_rrmse_pct_loss, load_ofdm_dataset
from modules.grid_search.encoder_decoder import ARCH_KEYS as _ED_ARCH_KEYS


# USER CONFIG
RUN_CONFIGS = [
    {
        "label": "50 mA",
        "dc_ma": 50,
        "channel_exp_dir": "data/experiments/train_and_validate/nice_rain_channel_models_20260915_2032",
        "ed_exp_dir":      "data/experiments/train_and_validate/nice_rain_encoder_decoder_20260916_0654",
        "ed_val_exp_dir":  "data/experiments/train_and_validate/nice_rain_ed_validation_20260917_1929",
        "dataset_path":    "data/sweeps/prime_coast_dc0.05A_fmin1e+06_fmax7.6e+06_20260724_2101.zarr",
    },
    # {
    #     "label": "60 mA",
    #     "dc_ma": 60,
    #     "channel_exp_dir": "data/experiments/train_and_validate/tiny_cliff_channel_models_20260810_1622",
    #     "ed_exp_dir":      "data/experiments/train_and_validate/tiny_cliff_encoder_decoder_20260811_0335",
    #     "ed_val_exp_dir":  "data/experiments/train_and_validate/tiny_cliff_ed_validation_20260811_0457",
    #     "dataset_path":    "data/sweeps/fair_ledge_dc0.06A_fmin1e+06_fmax9.2e+06_20260726_1115.zarr",
    # },
    # {
    #     "label": "80 mA",
    #     "dc_ma": 80,
    #     "channel_exp_dir": "data/experiments/train_and_validate/fleet_sand_channel_models_20260812_1903",
    #     "ed_exp_dir":      "data/experiments/train_and_validate/calm_coast_encoder_decoder_20260815_1103",
    #     "ed_val_exp_dir":  "data/experiments/train_and_validate/calm_coast_ed_validation_20260815_1237",
    #     "dataset_path":    "data/sweeps/calm_heath_dc0.08A_fmin1e+06_fmax1.08e+07_20260729_1339.zarr",
    # },
    # {
    #     "label": "120 mA",
    #     "dc_ma": 120,
    #     "channel_exp_dir": "data/experiments/train_and_validate/light_sea_channel_models_20260813_2059",
    #     "ed_exp_dir":      "data/experiments/train_and_validate/tame_flare_encoder_decoder_20260816_2213",
    #     "ed_val_exp_dir":  "data/experiments/train_and_validate/tame_flare_ed_validation_20260817_0004",
    #     "dataset_path":    "data/sweeps/mild_star_dc0.12A_fmin1e+06_fmax1.3e+07_20260802_2119.zarr",
    # },
]

# Restrict which channel-model families appear in the plots. None means all.
MODELS_TO_PLOT: set[str] | None = None

PLOT_PATH       = Path(__file__).resolve().parent.parent / "data/plots"
DEVICE          = os.environ.get("SUMMARIZE_DEVICE", "cpu")
N_POWER_BINS    = 10       # bins for plot 1
MAX_QQ_SAMPLES  = 10_000  # downsample for Q-Q speed
SHOW_RUN_IN_TITLE = False 
PARETO_TITLE = "Median Pareto Front for 10 E/Ds per Channel Model"
PRED_VS_ACTUAL_TITLE = "Predicted vs. Actual EVM% across Pareto Sweep Runs"
PACKET_TITLE = "Predicted Response of Best Gaussian TCN Channel Model"
VAL_RRMSE_TITLE = "Validation RRMSE% vs. Sent Power"
VAL_RRMSE_VS_PARAMS_TITLE = "Channel Model Validation RRMSE% vs. Parameter Count"
NOISE_FLOOR_TITLE = "Best E/D End-to-End Residual EVM% vs. Estimated Noise Floor"

# Style
_FONT = 7
_SMALL = 5
_mm = 1 / 25.4
_fw  = 88  * _mm   # single-column width
_fh  = 88  * _mm
_fw2 = 180 * _mm   # double-column width

plt.rcParams.update({
    "font.size": _FONT,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.labelsize": _FONT,
    "axes.titlesize": _FONT,
    "xtick.labelsize": _FONT,
    "ytick.labelsize": _FONT,
    "legend.fontsize": _FONT,
    "figure.dpi": 300,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.edgecolor": "black",
})
sns.set_style("whitegrid")
plt.rcParams.update({
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
})

_CMAP = plt.get_cmap("viridis")


MODEL_STYLES: dict[tuple[str, str], dict] = {
    ("gmp", "none"):       {"color": "#333333", "marker": "o", "linestyle": "-"},
    ("tthnet", "none"):    {"color": "#8C564B", "marker": "h", "linestyle": "-"},
    ("tcn", "none"):       {"color": "#0072B2", "marker": "^", "linestyle": "--"},
    ("tcn", "gaussian"):   {"color": "#E69F00", "marker": "s", "linestyle": "-."},
    ("tcn", "students_t"): {"color": "#D55E00", "marker": "D", "linestyle": ":"},
    ("lru", "none"):       {"color": "#009E73", "marker": "p", "linestyle": "--"},
    ("lru", "gaussian"):   {"color": "#CC79A7", "marker": "X", "linestyle": "-."},
    ("lru", "students_t"): {"color": "#56B4E9", "marker": "v", "linestyle": ":"},
    ("lstm", "none"):      {"color": "#9467BD", "marker": "<", "linestyle": "--"},
    ("lstm", "gaussian"):  {"color": "#17BECF", "marker": ">", "linestyle": "-."},
}
_FALLBACK_STYLES = [
    {"color": "#F0E442", "marker": "*", "linestyle": "-"},
    {"color": "#000000", "marker": "+", "linestyle": "--"},
]
_extra_style_cache: dict[tuple, dict] = {}


def _get_style(model: str, dist: str) -> dict:
    """Return the shared plot style for a (model, distribution) key."""
    key = (model, dist or "none")
    if key in MODEL_STYLES:
        return MODEL_STYLES[key]
    if key not in _extra_style_cache:
        style_index = len(_extra_style_cache) % len(_FALLBACK_STYLES)
        _extra_style_cache[key] = _FALLBACK_STYLES[style_index]
    return _extra_style_cache[key]


REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _run_name(cfg) -> str:
    for key in ("ed_val_exp_dir", "ed_exp_dir", "channel_exp_dir"):
        path = cfg.get(key)
        if not path:
            continue
        name = Path(path).name
        for marker in ("_ed_validation", "_encoder_decoder", "_channel_models"):
            name = name.split(marker)[0]
        return name
    return "?"


def _run_suffix(cfg) -> str:
    return f" (from {_run_name(cfg)})" if SHOW_RUN_IN_TITLE else ""


# DATA HELPERS
def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_channel_runs(channel_exp_dir: str) -> list[dict]:
    """Load runs.jsonl from a channel-model grid search and annotate with is_prob."""
    rows = _read_jsonl(_resolve(channel_exp_dir) / "runs.jsonl")
    for row in rows:
        distribution = row.get("distribution") or "none"
        row["is_tcn"] = row.get("model") == "tcn"
        row["is_prob"] = distribution in ("gaussian", "students_t")
    return rows


def load_adapter(run: dict, channel_exp_dir: str, device: str = DEVICE):
    """Rebuild a trained channel-model adapter from its run directory."""
    run_dir = _resolve(channel_exp_dir) / "runs" / run["run_id"]
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    return MODEL_REGISTRY[run["model"]].load(cfg["params"], run_dir / "model.pt", device)


def load_dataset(dataset_path: str, device: str = DEVICE):
    """
    Return (X_sent, Y_recv, ks_indices, symbol_length) from a zarr dataset.
    """
    root = zarr.open_group(_resolve(dataset_path), mode="r")
    attrs = dict(root.attrs)
    active_carrier_indices = np.array(attrs["active_carrier_indices"])
    cyclic_prefix_length = int(attrs["cyclic_prefix_length"])

    if "sent_burst" in root:
        preamble_length = int(attrs.get("preamble_length", 0))
        sent_burst = torch.tensor(root["sent_burst"][:], dtype=torch.float32, device=device)
        received_burst = torch.tensor(root["received_burst"][:], dtype=torch.float32, device=device)
        symbol_offset = preamble_length + cyclic_prefix_length
        X, Y = sent_burst[:, symbol_offset:], received_burst[:, symbol_offset:]
    else:
        X = torch.tensor(root["sent_baseband"][:], dtype=torch.float32, device=device)
        Y = torch.tensor(root["received_baseband"][:], dtype=torch.float32, device=device)

    return X, Y, active_carrier_indices, X.shape[1]


def _predict_mean(adapter, X: torch.Tensor) -> torch.Tensor:
    """Get the mean prediction from any adapter"""
    out = adapter.predict(X)
    return out[1] if isinstance(out, tuple) else out


def _best_by(runs: list[dict], key: str, *fallback_keys: str) -> dict | None:
    """Return the run with the minimum value for key, trying fallback_keys in
    order when key is absent."""
    def score(r):
        for k in (key, *fallback_keys):
            if r.get(k) is not None:
                return float(r[k])
        return float("inf")
    return min(runs, key=score, default=None)


def per_trial_evm(ed_val_exp_dir: str, ks_indices: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute per-trial EVM% from the time-domain signals stored in validation.zarr.
    Returns {run_id: array of shape [num_trials]}.

    sent_time and received_time are symbol-only (no CP, no preamble), so we
    rfft directly and index by ks_indices.
    """
    root = zarr.open_group(_resolve(ed_val_exp_dir) / "validation.zarr", mode="r")
    evm_by_run = {}
    for run_id in root.keys():
        group = root[run_id]
        sent = group["sent_time"][:]     # (num_trials, symbol_length)
        received = group["received_time"][:]

        trial_evms = []
        for trial in range(sent.shape[0]):
            sent_spectrum = np.fft.rfft(sent[trial], norm="ortho")[ks_indices]
            received_spectrum = np.fft.rfft(received[trial], norm="ortho")[ks_indices]
            signal_power = np.mean(np.abs(sent_spectrum) ** 2)
            residual_power = np.mean(np.abs(sent_spectrum - received_spectrum) ** 2)
            trial_evms.append(float(np.sqrt(residual_power / (signal_power + 1e-12)) * 100))
        evm_by_run[run_id] = np.array(trial_evms)
    return evm_by_run


def _binned_rrmse(
    X: torch.Tensor,
    Y: torch.Tensor,
    adapter,
    n_bins: int = N_POWER_BINS,
  
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-bin per-burst RRMSE (%) binned by sent-power (mean squared amplitude)."""
    power = X.square().mean(dim=-1)
    bin_edges = torch.linspace(power.min(), power.max(), n_bins + 1, device=power.device)
    bin_index = torch.bucketize(power, bin_edges)
    bin_centers = (0.5 * (bin_edges[:-1] + bin_edges[1:])).cpu().numpy()

    predicted = _predict_mean(adapter, X)
    target = Y

    rrmse_per_bin = []
    for bin_number in range(n_bins):
        in_bin = bin_index == bin_number
        if in_bin.any():
            rrmse_per_bin.append(float(calculate_per_burst_rrmse_pct_loss(target[in_bin], predicted[in_bin])))
        else:
            rrmse_per_bin.append(float("nan"))
    return bin_centers, np.array(rrmse_per_bin)



def _model_type_label(model: str, dist: str) -> str:
    dist_name = {"none": "nonprob", "gaussian": "Gaussian", "students_t": "Student's-t"}.get(dist, dist)
    return f"{model.upper()} {dist_name}"


def _model_type_sort_key(mt: tuple[str, str]) -> tuple:
    return ({"gmp": 0, "tthnet": 1, "tcn": 2, "lru": 3, "lstm": 4}.get(mt[0], 99),
            {"none": 0, "gaussian": 1, "students_t": 2}.get(mt[1], 99))


def _include_model(model: str) -> bool:
    return MODELS_TO_PLOT is None or model in MODELS_TO_PLOT


def _discover_model_types(run_configs: list[dict]) -> list[tuple[str, str]]:
    seen = []
    for cfg in run_configs:
        for r in load_channel_runs(cfg["channel_exp_dir"]):
            if not _include_model(r["model"]):
                continue
            mt = (r["model"], r.get("distribution") or "none")
            if mt not in seen:
                seen.append(mt)
    return sorted(seen, key=_model_type_sort_key)


# Grid plots share one panel per DC bias. The grid is 1x1, 1x2, or 2x2 depending
# on how many biases are active.
_PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]


def _panel_grid(n: int, sharex: bool = True):
    """Build a panel grid sized for n panels and return
    (fig, axes_flat, needs_ylabel, needs_xlabel)."""
    n_cols = min(n, 2)
    n_rows = math.ceil(n / n_cols)
    fig_width = _fw2 if n_cols == 2 else _fw2 / 2

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_width, n_rows * _fh),
                             sharex=sharex, sharey=True,
                             constrained_layout=True, squeeze=False)
    axes_flat = axes.flatten()

    needs_ylabel = [panel_index % n_cols == 0 for panel_index in range(n)]
    needs_xlabel = [panel_index >= n - n_cols for panel_index in range(n)]

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    return fig, axes_flat, needs_ylabel, needs_xlabel


def _decade_only_log_xaxis(ax) -> None:
    """Log x-axis labeled at decades only (10^3, 10^4), minor ticks unlabeled.
    Leaves the limits autoscaled so the axis tracks the data range."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_val_rrmse_vs_power(run_configs: list[dict]) -> None:
    model_types = _discover_model_types(run_configs)
    n = len(run_configs)

    fig, axes_flat, needs_ylabel, needs_xlabel = _panel_grid(n)
    for panel_index, cfg in enumerate(run_configs):
        ax = axes_flat[panel_index]
        X, Y = _load_preamble_stripped(cfg)
        runs = load_channel_runs(cfg["channel_exp_dir"])

        for model, dist in model_types:
            style = _get_style(model, dist)
            candidates = [r for r in runs
                          if r["model"] == model and (r.get("distribution") or "none") == dist]
            if not candidates:
                continue
            best    = _best_by(candidates, "val_per_burst_rrmse_pct", "val_rrmse_pct")
            adapter = load_adapter(best, cfg["channel_exp_dir"])
            centers, rrmse = _binned_rrmse(X, Y, adapter)
            mask = ~np.isnan(rrmse)
            if not mask.any():
                continue
            ax.plot(
                centers[mask], rrmse[mask],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=2.5,
                label=_model_type_label(model, dist),
            )

        ax.set_title(f"{cfg['dc_ma']} mA", fontsize=_FONT)
        ax.grid(True)
        ax.text(0.05, 0.95, _PANEL_LABELS[panel_index], transform=ax.transAxes,
                fontsize=_FONT, va="top", ha="left")
        ax.tick_params(labelbottom=True)
        if needs_ylabel[panel_index]:
            ax.set_ylabel("Val RRMSE (%)")

        if needs_xlabel[panel_index]:
            ax.set_xlabel("Sent Power (Mean Squared Amplitude)")

    fig.suptitle(VAL_RRMSE_TITLE + _run_suffix(run_configs[0]), fontsize=_FONT)

    axes_flat[0].legend(fontsize=_FONT, handlelength=3, labelspacing=0.5)

    plt.savefig(PLOT_PATH / "val_rrmse_vs_power.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "val_rrmse_vs_power.png", bbox_inches="tight")
    plt.show()


# PLOT 2: channel-model accuracy vs size - validation per-burst RRMSE% vs parameter count.
def plot_val_rrmse_vs_params(run_configs: list[dict]) -> None:
    n = len(run_configs)

    fig, axes_flat, needs_ylabel, needs_xlabel = _panel_grid(n)

    for panel_index, cfg in enumerate(run_configs):
        ax = axes_flat[panel_index]
        runs = load_channel_runs(cfg["channel_exp_dir"])

        validation_rows = _read_jsonl(_resolve(cfg["ed_val_exp_dir"]) / "runs.jsonl")
        trained_channel_ids = {row["channel_run_id"] for row in validation_rows}

        traces: dict[tuple, list] = {}
        for run in runs:
            if run["run_id"] not in trained_channel_ids:
                continue
            if not _include_model(run["model"]):
                continue
            key = (run["model"], run.get("distribution") or "none")
            traces.setdefault(key, []).append((run["num_params"], run["val_per_burst_rrmse_pct"]))

        for key in sorted(traces.keys(), key=_model_type_sort_key):
            model, dist = key
            style = _get_style(model, dist)
            points = sorted(traces[key])
            param_counts = np.array([params for params, _ in points], dtype=float)
            rrmse_values = np.array([rrmse for _, rrmse in points])

            ax.plot(param_counts, rrmse_values, color=style["color"], linestyle="-",
                    linewidth=1.2, alpha=0.85, marker=style["marker"], markersize=3,
                    label=_model_type_label(model, dist))

        # channel sizes were chosen one per power-of-2 bucket, so a log axis spaces them evenly
        _decade_only_log_xaxis(ax)
        ax.set_title(f"{cfg['dc_ma']} mA", fontsize=_FONT)
        ax.grid(True, which="both")
        ax.text(0.05, 0.95, _PANEL_LABELS[panel_index], transform=ax.transAxes,
                fontsize=_FONT, va="top", ha="left")
        ax.tick_params(labelbottom=True)
        if needs_ylabel[panel_index]:
            ax.set_ylabel("Val RRMSE (%)")

        if needs_xlabel[panel_index]:
            ax.set_xlabel("Channel Model Parameter Count")

    fig.suptitle(VAL_RRMSE_VS_PARAMS_TITLE + _run_suffix(run_configs[0]), fontsize=_FONT)

    axes_flat[0].legend(fontsize=_SMALL, handlelength=2, markerscale=0.8,
                        labelspacing=0.3, borderpad=0.4)

    plt.savefig(PLOT_PATH / "val_rrmse_vs_params.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "val_rrmse_vs_params.png", bbox_inches="tight")
    plt.show()


# PLOT 3: Pareto - channel model params vs experimental EVM%

SEED_DOT_ALPHA = 0.2
SEED_DOT_SIZE = 3
PARETO_SHOW_SEED_DOTS = True   # scatter each E/D seed's EVM behind the pareto line

def plot_pareto_evm(run_configs: list[dict]) -> None:
    n = len(run_configs)

    fig, axes_flat, needs_ylabel, needs_xlabel = _panel_grid(n)

    for panel_index, cfg in enumerate(run_configs):
        ax = axes_flat[panel_index]
        _, _, active_carrier_indices, _ = load_dataset(cfg["dataset_path"])
        channel_by_run_id = {row["run_id"]: row for row in load_channel_runs(cfg["channel_exp_dir"])}
        validation_rows = _read_jsonl(_resolve(cfg["ed_val_exp_dir"]) / "runs.jsonl")
        trial_evms = per_trial_evm(cfg["ed_val_exp_dir"], active_carrier_indices)

        # every E/D validated against a channel model, one mean EVM per seed
        seeds_per_channel: dict[str, list] = {}
        for validation_row in validation_rows:
            channel_run_id = validation_row.get("channel_run_id")
            if channel_run_id is None:
                continue
            trials = trial_evms.get(validation_row["run_id"], np.array([validation_row.get("evm_pct", float("nan"))]))
            seeds_per_channel.setdefault(channel_run_id, []).append(float(np.nanmean(trials)))

        traces: dict[tuple, list] = {}
        for channel_run_id, seed_evms in seeds_per_channel.items():
            channel_run = channel_by_run_id.get(channel_run_id)
            if channel_run is None:
                continue
            if not _include_model(channel_run.get("model", "unknown")):
                continue
            key = (channel_run.get("model", "unknown"), channel_run.get("distribution") or "none")
            traces.setdefault(key, []).append({
                "num_params": channel_run["num_params"],
                "median_evm": float(np.median(seed_evms)),
                "seed_evms": np.asarray(seed_evms),
            })

        for key in sorted(traces.keys(), key=_model_type_sort_key):
            model, dist = key
            style = _get_style(model, dist)
            points = sorted(traces[key], key=lambda point: point["num_params"])
            param_counts = np.array([point["num_params"] for point in points], dtype=float)
            median_evms = np.array([point["median_evm"] for point in points])

            # solid line tracks the pareto front (best median so far with increasing params)
            pareto_evms = np.minimum.accumulate(median_evms)
            ax.plot(param_counts, pareto_evms, color=style["color"], linestyle="-",
                    linewidth=1.2, alpha=0.85, zorder=3,
                    drawstyle="steps-post", label=_model_type_label(model, dist))
            on_front = median_evms == pareto_evms
            ax.plot(param_counts[on_front], median_evms[on_front], linestyle="none",
                    marker=style["marker"], markersize=3,
                    color=style["color"], alpha=0.85, zorder=4)

            if PARETO_SHOW_SEED_DOTS:
                for point in points:
                    ax.plot(np.full(len(point["seed_evms"]), point["num_params"]),
                            point["seed_evms"], linestyle="none", marker="o",
                            markersize=SEED_DOT_SIZE, markerfacecolor=style["color"],
                            markeredgecolor="none", alpha=SEED_DOT_ALPHA, zorder=2)

        _decade_only_log_xaxis(ax)
        ax.set_ylim(top=15)
        ax.set_title(f"{cfg['dc_ma']} mA", fontsize=_FONT)
        ax.grid(True, which="both")
        ax.text(0.05, 0.95, _PANEL_LABELS[panel_index], transform=ax.transAxes,
                fontsize=_FONT, va="top", ha="left")
        ax.tick_params(labelbottom=True)
        if needs_ylabel[panel_index]:
            ax.set_ylabel("Experimental EVM (%)")

        if needs_xlabel[panel_index]:
            ax.set_xlabel("Channel Model Parameter Count")

    fig.suptitle(PARETO_TITLE + _run_suffix(run_configs[0]), fontsize=_FONT)

    axes_flat[0].legend(fontsize=_SMALL, handlelength=2, markerscale=0.8,
                        labelspacing=0.3, borderpad=0.4)

    plt.savefig(PLOT_PATH / "pareto_evm.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "pareto_evm.png", bbox_inches="tight")
    plt.show()


# PLOT 4: channel per-burst RRMSE vs paired-encoder EVM,
RRMSE_EVM_SCATTER_TITLE = "Channel Per-Burst RRMSE Vs. Paired-Encoder EVM"


def _operating_power_band(validation_rows: list[dict]) -> tuple[float, float]:
    """Sent-power band the validated encoders operate in, from their output powers."""
    powers = np.array([row["encoder_power_mean"] for row in validation_rows], dtype=float)
    return float(powers.min()), float(powers.max())


def _load_preamble_stripped(cfg):
    sent, received, ofdm_config = load_ofdm_dataset(_resolve(cfg["dataset_path"]).as_posix(), DEVICE)
    preamble_length = sent.shape[1] - ofdm_config.baseband_fft_length - ofdm_config.cyclic_prefix_length
    return sent[:, preamble_length:], received[:, preamble_length:]


def _channel_val_split(cfg):
    exp_config = yaml.safe_load((_resolve(cfg["channel_exp_dir"]) / "experiment_config.yaml").read_text())
    seed = int(exp_config["seed"])
    val_fraction = float(exp_config["val_fraction"])

    sent, received = _load_preamble_stripped(cfg)

    n = sent.shape[0]
    n_val = round(n * val_fraction)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    val_idx = perm[:n_val]
    return sent[val_idx], received[val_idx]


def _channel_rrmse_aggregate_and_band(X, Y, adapter, band: tuple[float, float]):
    predicted = _predict_mean(adapter, X)
    target = Y.to(predicted.device)

    power = X.square().mean(dim=-1)
    band_low, band_high = band
    in_band = (power >= band_low) & (power <= band_high)

    aggregate = calculate_per_burst_rrmse_pct_loss(target, predicted)
    operating = calculate_per_burst_rrmse_pct_loss(target[in_band], predicted[in_band])
    return aggregate, operating, int(in_band.sum())


def plot_rrmse_evm_scatter(run_configs: list[dict]) -> None:
    cfg = run_configs[0]  # the biases currently share one 50 mA dataset

    _, _, active_carrier_indices, _ = load_dataset(cfg["dataset_path"])
    X_val, Y_val = _channel_val_split(cfg)
    channel_by_run_id = {row["run_id"]: row for row in load_channel_runs(cfg["channel_exp_dir"])}
    validation_rows = _read_jsonl(_resolve(cfg["ed_val_exp_dir"]) / "runs.jsonl")
    trial_evms = per_trial_evm(cfg["ed_val_exp_dir"], active_carrier_indices)

    band = _operating_power_band(validation_rows)

    seeds_per_channel: dict[str, list] = {}
    for validation_row in validation_rows:
        channel_run_id = validation_row.get("channel_run_id")
        if channel_run_id is None:
            continue
        trials = trial_evms.get(validation_row["run_id"],
                                np.array([validation_row.get("evm_pct", float("nan"))]))
        seeds_per_channel.setdefault(channel_run_id, []).append(float(np.nanmean(trials)))

    points = []
    for channel_run_id, seed_evms in seeds_per_channel.items():
        channel_run = channel_by_run_id.get(channel_run_id)
        if channel_run is None:
            continue

        adapter = load_adapter(channel_run, cfg["channel_exp_dir"])
        aggregate_rrmse, operating_rrmse, _n_band = _channel_rrmse_aggregate_and_band(X_val, Y_val, adapter, band)
        seed_evms = np.asarray(seed_evms)

        points.append({
            "model": channel_run.get("model", "unknown"),
            "dist": channel_run.get("distribution") or "none",
            "num_params": channel_run["num_params"],
            "aggregate_rrmse": aggregate_rrmse,
            "operating_rrmse": operating_rrmse,
            "median_evm": float(np.median(seed_evms)),
            "seed_evms": seed_evms,
        })

    fig, axes = plt.subplots(1, 2, figsize=(_fw2, _fh), sharey=True, constrained_layout=True)

    panel_specs = [
        ("aggregate_rrmse", "Per-Burst RRMSE, all sent powers (%)", "(a)"),
        ("operating_rrmse",
         f"Per-Burst RRMSE, encoder band [{band[0]:.2f}, {band[1]:.2f}] (%)", "(b)"),
    ]

    for ax, (x_key, x_label, panel_label) in zip(axes, panel_specs):
        drawn_types = set()
        for point in points:
            style = _get_style(point["model"], point["dist"])
            label = _model_type_label(point["model"], point["dist"])
            show_label = label not in drawn_types
            drawn_types.add(label)

            ax.plot(np.full(len(point["seed_evms"]), point[x_key]), point["seed_evms"],
                    linestyle="none", marker="o", markersize=SEED_DOT_SIZE,
                    markerfacecolor=style["color"], markeredgecolor="none",
                    alpha=SEED_DOT_ALPHA, zorder=2)
            ax.plot(point[x_key], point["median_evm"],
                    linestyle="none", marker=style["marker"], markersize=5,
                    color=style["color"], zorder=4,
                    label=label if show_label else None)

        x = np.array([point[x_key] for point in points])
        y = np.array([point["median_evm"] for point in points])
        pearson = np.corrcoef(x, y)[0, 1]
        spearman = np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1]

        ax.text(0.05, 0.95,
                f"{panel_label}  Spearman r={spearman:.2f}\nPearson r={pearson:.2f}",
                transform=ax.transAxes, fontsize=_FONT, va="top", ha="left")
        ax.set_xlabel(x_label)
        ax.grid(True)

    axes[0].set_ylabel("Experimental EVM (%)")
    axes[0].legend(fontsize=_SMALL, handlelength=1.5, labelspacing=0.3, borderpad=0.4)
    fig.suptitle(RRMSE_EVM_SCATTER_TITLE + _run_suffix(cfg), fontsize=_FONT)

    plt.savefig(PLOT_PATH / "rrmse_evm_scatter.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "rrmse_evm_scatter.png", bbox_inches="tight")
    plt.show()

    print(f"\n  Operating-power band from validated encoders: "
          f"[{band[0]:.3f}, {band[1]:.3f}] (mean squared amplitude)")
    print(f"  {'channel_run':<16}{'params':>8}{'aggR%':>9}{'bandR%':>9}{'medEVM%':>9}")
    for point in sorted(points, key=lambda p: p["median_evm"]):
        run_id = next(rid for rid, s in seeds_per_channel.items()
                      if np.array_equal(np.asarray(s), point["seed_evms"]))
        print(f"  {run_id:<16}{point['num_params']:>8}{point['aggregate_rrmse']:>9.2f}"
              f"{point['operating_rrmse']:>9.2f}{point['median_evm']:>9.2f}")

    for x_key, name in [("aggregate_rrmse", "aggregate"), ("operating_rrmse", "operating-band")]:
        x = np.array([point[x_key] for point in points])
        y = np.array([point["median_evm"] for point in points])
        spearman = np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1]
        print(f"  Spearman(EVM, {name} RRMSE) = {spearman:.3f}")


# PLOT 5: predicted vs actual actual EVM per validated E/D run

def _form_style(channel_form: str) -> dict:
    form_to_key = {
        "gmp": ("gmp", "none"),
        "nonprob TCN": ("tcn", "none"),
        "prob TCN": ("tcn", "gaussian"),
        "nonprob LRU": ("lru", "none"),
        "prob LRU": ("lru", "gaussian"),
    }
    return _get_style(*form_to_key.get(channel_form, (channel_form, "none")))


def _replay_predicted_trial_evms(cfg) -> dict[str, np.ndarray]:
    """Per validated run: predicted per-trial EVM% from replaying its validation
    sent bursts through encoder -> frozen channel model -> decoder."""
    from modules.models import TCN

    root = zarr.open_group(_resolve(cfg["dataset_path"]), mode="r")
    active_carrier_indices = np.array(root.attrs["active_carrier_indices"])
    cyclic_prefix_length = int(root.attrs["cyclic_prefix_length"])
    clip_threshold = 3.0

    validation = zarr.open_group(_resolve(cfg["ed_val_exp_dir"]) / "validation.zarr", mode="r")
    validation_rows = _read_jsonl(_resolve(cfg["ed_val_exp_dir"]) / "runs.jsonl")
    encoder_decoder_rows = {row["run_id"]: row for row in _read_jsonl(_resolve(cfg["ed_exp_dir"]) / "runs.jsonl")}
    channel_by_run_id = {row["run_id"]: row for row in load_channel_runs(cfg["channel_exp_dir"])}

    channel_cache: dict[str, object] = {}
    predicted_evm = {}
    for row in validation_rows:
        run_id, ed_run_id, channel_run_id = row["run_id"], row["model"], row.get("channel_run_id")
        if run_id not in validation or ed_run_id not in encoder_decoder_rows or channel_run_id not in channel_by_run_id:
            continue

        architecture = {key: encoder_decoder_rows[ed_run_id][key] for key in _ED_ARCH_KEYS}
        encoder = TCN(**architecture).to(DEVICE)
        decoder = TCN(**architecture).to(DEVICE)
        checkpoint = torch.load(_resolve(cfg["ed_exp_dir"]) / "runs" / ed_run_id / "model.pt",
                                map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        encoder.eval()
        decoder.eval()

        if channel_run_id not in channel_cache:
            channel_cache[channel_run_id] = load_adapter(channel_by_run_id[channel_run_id], cfg["channel_exp_dir"])
        channel = channel_cache[channel_run_id]

        symbol = torch.tensor(validation[run_id]["sent_time"][:].astype(np.float32), device=DEVICE)
        symbol_length = symbol.shape[1]
        burst = torch.hstack([symbol[:, -cyclic_prefix_length:], symbol])  # [CP | symbol]
        with torch.no_grad():
            encoded = encoder(burst).clamp(-clip_threshold, clip_threshold)
            channel_output = channel.predict(encoded)
            if isinstance(channel_output, tuple):   # prob: noisy forward
                channel_output = channel_output[0]
            decoded = decoder(channel_output)[:, -symbol_length:].cpu().numpy()

        sent_spectrum = np.fft.fft(symbol.cpu().numpy().astype(np.float64), norm="ortho", axis=1)[:, active_carrier_indices]
        decoded_spectrum = np.fft.fft(decoded.astype(np.float64), norm="ortho", axis=1)[:, active_carrier_indices]
        signal_power = np.mean(np.abs(sent_spectrum) ** 2, axis=1)
        predicted_evm[run_id] = np.sqrt(np.mean(np.abs(decoded_spectrum - sent_spectrum) ** 2, axis=1) / signal_power) * 100
    return predicted_evm


def plot_predicted_vs_actual_evm(run_configs: list[dict]) -> None:
    n = len(run_configs)
    fig, axes_flat, needs_ylabel, needs_xlabel = _panel_grid(n)

    for panel_index, cfg in enumerate(run_configs):
        ax = axes_flat[panel_index]
        if "ed_exp_dir" not in cfg:
            print(f"[predicted_vs_actual] {cfg['label']}: no ed_exp_dir; skipping")
            continue
        _, _, active_carrier_indices, _ = load_dataset(cfg["dataset_path"])
        predicted = _replay_predicted_trial_evms(cfg)
        actual = per_trial_evm(cfg["ed_val_exp_dir"], active_carrier_indices)
        validation_rows = _read_jsonl(_resolve(cfg["ed_val_exp_dir"]) / "runs.jsonl")

        seen_forms = set()
        axis_limits = [np.inf, -np.inf]
        for row in validation_rows:
            run_id, form = row["run_id"], row["channel_form"]
            if run_id not in predicted or run_id not in actual:
                continue
            predicted_evm, actual_evm = float(np.nanmean(predicted[run_id])), float(np.nanmean(actual[run_id]))
            style = _form_style(form)
            ax.scatter([predicted_evm], [actual_evm], s=14, color=style["color"], marker=style["marker"],
                       label=form if form not in seen_forms else None, zorder=3)
            seen_forms.add(form)
            axis_limits = [min(axis_limits[0], predicted_evm, actual_evm),
                           max(axis_limits[1], predicted_evm, actual_evm)]

        pad = 0.08 * (axis_limits[1] - axis_limits[0] + 1e-9)
        lo, hi = axis_limits[0] - pad, axis_limits[1] + pad
        ax.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=0.8, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{cfg['dc_ma']} mA", fontsize=_FONT)
        ax.grid(True)
        ax.text(0.05, 0.95, _PANEL_LABELS[panel_index], transform=ax.transAxes,
                fontsize=_FONT, va="top", ha="left")
        ax.tick_params(labelbottom=True)
        if needs_ylabel[panel_index]:
            ax.set_ylabel("Actual EVM (%)")

        if needs_xlabel[panel_index]:
            ax.set_xlabel("Predicted EVM (%)")

    fig.suptitle(PRED_VS_ACTUAL_TITLE + _run_suffix(run_configs[0]), fontsize=_FONT)

    axes_flat[0].legend(fontsize=_SMALL, handlelength=1.2, markerscale=0.9,
                        labelspacing=0.3, borderpad=0.4)

    plt.savefig(PLOT_PATH / "predicted_vs_actual_evm.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "predicted_vs_actual_evm.png", bbox_inches="tight")
    plt.show()

# PLOT 6: best E/D end-to-end residual EVM vs estimated additive noise floor.
def noise_floor_evm_vs_frequency(ed_val_exp_dir, validation_run_id) -> np.ndarray:
    group = zarr.open_group(_resolve(ed_val_exp_dir) / "validation.zarr", mode="r")[validation_run_id]
    replays = group["noise_floor_replays"][:]        # (num_replays, num_carriers) complex
    reference = group["noise_floor_reference"][:]    # (num_carriers,) complex
    noise_power = np.var(replays, axis=0, ddof=1)
    reference_power = np.abs(reference) ** 2
    return np.sqrt(noise_power / (reference_power + 1e-12)) * 100.0


def plot_best_ed_noise_floor(run_configs: list[dict]) -> None:
    n = len(run_configs)

    fig, axes_flat, needs_ylabel, needs_xlabel = _panel_grid(n, sharex=False)

    y_max = 0.0
    for panel_index, cfg in enumerate(run_configs):
        ax = axes_flat[panel_index]
        _, _, active_carrier_indices, _ = load_dataset(cfg["dataset_path"])
        validation_rows = _read_jsonl(_resolve(cfg["ed_val_exp_dir"]) / "runs.jsonl")
        best = _best_by(validation_rows, "evm_pct")

        frequencies_mhz = subcarrier_frequencies_mhz(cfg["dataset_path"], active_carrier_indices)
        residual_evm = validation_run_evm_vs_frequency(cfg["ed_val_exp_dir"], best["run_id"],
                                                       active_carrier_indices)
        noise_floor_evm = noise_floor_evm_vs_frequency(cfg["ed_val_exp_dir"], best["run_id"])

        residual_mean = float(np.nanmean(residual_evm))
        noise_floor_mean = float(np.nanmean(noise_floor_evm))
        y_max = max(y_max, np.nanmax(residual_evm), np.nanmax(noise_floor_evm))

        ax.plot(frequencies_mhz, residual_evm, color="#0072B2", marker="o", markersize=2.5,
                linewidth=1.0, label="End-to-end residual")
        ax.plot(frequencies_mhz, noise_floor_evm, color="#D55E00", marker="^", markersize=2.5,
                linewidth=1.0, linestyle="--", label="Estimated noise floor")

        ax.set_title(f"{cfg['dc_ma']} mA", fontsize=_FONT)
        ax.grid(True)
        ax.text(0.05, 0.95,
                f"{_PANEL_LABELS[panel_index]}  residual mean EVM% = {residual_mean:.1f}\n"
                f"noise floor mean EVM% = {noise_floor_mean:.1f}",
                transform=ax.transAxes, fontsize=_FONT, va="top", ha="left")
        if needs_ylabel[panel_index]:
            ax.set_ylabel("EVM (%)")

        if needs_xlabel[panel_index]:
            ax.set_xlabel("Frequency (MHz)")

    axes_flat[0].set_ylim(0, y_max * 1.05)

    fig.suptitle(NOISE_FLOOR_TITLE + _run_suffix(run_configs[0]), fontsize=_FONT)

    axes_flat[0].legend(fontsize=_SMALL, handlelength=2, labelspacing=0.3, borderpad=0.4)

    plt.savefig(PLOT_PATH / "best_ed_noise_floor.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "best_ed_noise_floor.png", bbox_inches="tight")
    plt.show()


# PLOT 7: TCN predicted response to a Gaussian wave packet (lowest DC bias)
def plot_packet_response(run_configs: list[dict]) -> None:
    cfg = min(run_configs, key=lambda c: c["dc_ma"])

    runs = load_channel_runs(cfg["channel_exp_dir"])
    prob_runs = [r for r in runs if r["is_prob"] and r["is_tcn"]]
    if not prob_runs:
        print(f"[plot_packet_response] No prob TCN found for {cfg['label']}; skipping.")
        return
    best = _best_by(prob_runs, "val_per_burst_rrmse_pct", "val_rrmse_pct")

    adapter = load_adapter(best, cfg["channel_exp_dir"])
    model = adapter.model
    model.eval()

    # infer baseband sampling rate from dataset
    root = zarr.open_group(_resolve(cfg["dataset_path"]), mode="r")
    attrs = dict(root.attrs)
    active_carrier_indices = np.array(attrs["active_carrier_indices"])
    f_min_hz = float(attrs.get("f_min_hz", 300e3))
    subcarrier_spacing_hz = f_min_hz / active_carrier_indices[0]   # e.g. 300e3 / 30 = 10 kHz
    if "sent_burst" in root:
        preamble_length = int(attrs.get("preamble_length", 0))
        cyclic_prefix_length = int(attrs["cyclic_prefix_length"])
        symbol_length = root["sent_burst"].shape[1] - preamble_length - cyclic_prefix_length
    else:
        symbol_length = root["sent_baseband"].shape[1]
    sample_rate_hz = symbol_length * subcarrier_spacing_hz

    receptive_field = int(model.receptive_field)
    carriers = np.arange(symbol_length // 2 + 1)
    first_carrier, last_carrier = int(active_carrier_indices.min()), int(active_carrier_indices.max())
    center_carrier = 0.5 * (first_carrier + last_carrier)
    carrier_sigma = (last_carrier - first_carrier) / 6.0   # 3 sigma reaches each band edge
    in_band = (carriers >= first_carrier) & (carriers <= last_carrier)
    packet_spectrum = np.zeros(symbol_length // 2 + 1, dtype=complex)
    packet_spectrum[in_band] = np.exp(-0.5 * ((carriers[in_band] - center_carrier) / carrier_sigma) ** 2)
    packet = np.fft.irfft(packet_spectrum, n=symbol_length)
    packet = np.roll(packet, symbol_length // 2)   # centre the packet within each period
    packet *= 3.0 / np.abs(packet).max()           # scale peak to the +/-3 training range

    repetitions = max(1, int(np.ceil(receptive_field / symbol_length))) + 2   # enough periods for full RF context
    num_samples = repetitions * symbol_length
    packet_input = torch.tensor(np.tile(packet, repetitions), dtype=torch.float32).unsqueeze(0)

    device = next(model.parameters()).device
    with torch.no_grad():
        _, mean, std, nu = model(packet_input.to(device))

    # zoom to a window around one packet (past the receptive field), packet centre at t=0
    ns_per_sample = 1e9 / sample_rate_hz
    context_periods = max(1, int(np.ceil(receptive_field / symbol_length)))
    center = context_periods * symbol_length + symbol_length // 2
    envelope_width_samples = symbol_length / (2 * np.pi * carrier_sigma)
    half_window = int(np.ceil(6 * envelope_width_samples))   # +/-6 sigma around the packet
    window_start, window_end = center - half_window, center + half_window
    time_ns = (np.arange(num_samples) - center) * ns_per_sample

    predicted_mean = mean.squeeze(0).cpu().numpy()
    predicted_std = std.squeeze(0).cpu().numpy()
    predicted_nu = nu.squeeze(0).cpu().numpy()
    input_signal = packet_input.squeeze(0).numpy()
    is_gaussian = best.get("distribution", "gaussian") == "gaussian"

    cmap = _CMAP
    fig, ax1 = plt.subplots(figsize=(_fw, _fh))

    if not is_gaussian:
        lo, hi = stats.t.interval(0.997, predicted_nu[window_start:window_end],
                                  loc=predicted_mean[window_start:window_end],
                                  scale=predicted_std[window_start:window_end])
        ax1.fill_between(time_ns[window_start:window_end], lo, hi, color=cmap(0.9), alpha=0.8,
                         label="99.7% CI Student's-t")

    ax1.fill_between(time_ns[window_start:window_end],
                     predicted_mean[window_start:window_end] - 3 * predicted_std[window_start:window_end],
                     predicted_mean[window_start:window_end] + 3 * predicted_std[window_start:window_end],
                     color=cmap(0.5), alpha=0.8, label="±3 Std Dev")
    ax1.plot(time_ns[window_start:window_end], predicted_mean[window_start:window_end],
             color=cmap(0.1), label="Predicted Mean")

    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Predicted Received Amplitude", color=cmap(0.1))
    ax1.tick_params(axis="y", labelcolor=cmap(0.1))
    ax1.grid(False)
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)

    ax2 = ax1.twinx()
    ax2.set_zorder(0)
    ax2.grid(True, alpha=0.3)
    ax2.plot(time_ns[window_start:window_end], input_signal[window_start:window_end], "--",
             color="black", alpha=0.8, label="Input Signal")
    ax2.set_ylabel("Input Signal Amplitude", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    left_handles, left_labels = ax1.get_legend_handles_labels()
    right_handles, right_labels = ax2.get_legend_handles_labels()
    ax1.legend(left_handles + right_handles, left_labels + right_labels, loc="upper right",
               fontsize=_SMALL, frameon=True, framealpha=1)
    ax1.set_title(f"{PACKET_TITLE} ({cfg['dc_ma']} mA)" + _run_suffix(cfg), fontsize=_FONT)

    plt.tight_layout()
    plt.savefig(PLOT_PATH / "packet_response.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "packet_response.png", bbox_inches="tight")
    plt.show()

# PLOT 9: EVM% (and SNR dB) vs frequency from the fixed preamble
def evm_percent_to_snr_db(evm_percent):
    evm_fraction = np.asarray(evm_percent) / 100.0
    return -20.0 * np.log10(evm_fraction + 1e-12)


def load_fixed_preamble_and_received_preambles(dataset_path, device=DEVICE):
    root = zarr.open_group(_resolve(dataset_path), mode="r")
    attrs = dict(root.attrs)
    active_carrier_indices = np.array(attrs["active_carrier_indices"])
    preamble_length = int(attrs["preamble_length"])
    cyclic_prefix_length = int(attrs["cyclic_prefix_length"])
    subcarrier_spacing_hz = float(attrs["f_min_hz"]) / active_carrier_indices[0]
    symbol_length = root["sent_burst"].shape[1] - preamble_length - cyclic_prefix_length
    baseband_sampling_rate_hz = symbol_length * subcarrier_spacing_hz
    band_min_hz = float(attrs["f_min_hz"])
    band_max_hz = float(attrs.get("f_max_hz", active_carrier_indices[-1] * subcarrier_spacing_hz))
    sent_preamble = torch.tensor(root["sent_burst"][0, :preamble_length], dtype=torch.float32, device=device)
    received_preambles = np.array(root["received_burst"][:, :preamble_length])
    return (sent_preamble, received_preambles, preamble_length,
            baseband_sampling_rate_hz, band_min_hz, band_max_hz)


def preamble_in_band_frequencies(preamble_length, baseband_sampling_rate_hz, band_min_hz, band_max_hz):
    preamble_frequencies_hz = np.fft.rfftfreq(preamble_length, d=1.0 / baseband_sampling_rate_hz)
    in_band_mask = (preamble_frequencies_hz >= band_min_hz) & (preamble_frequencies_hz <= band_max_hz)
    return preamble_frequencies_hz[in_band_mask], in_band_mask


def measured_preamble_evm_vs_frequency(received_preambles, in_band_mask):
    received_spectrum = np.fft.rfft(received_preambles, norm="ortho", axis=-1)[:, in_band_mask]
    mean_signal_spectrum = received_spectrum.mean(axis=0)
    residual_noise_power = np.mean(np.abs(received_spectrum - mean_signal_spectrum[None, :]) ** 2, axis=0)
    return np.sqrt(residual_noise_power) / (np.abs(mean_signal_spectrum) + 1e-12) * 100.0


def predicted_preamble_evm_vs_frequency(adapter, sent_preamble, in_band_mask, is_gaussian,
                                        num_stochastic_samples=5000):
    model = adapter.model
    model.eval()
    device = next(model.parameters()).device
    preamble_batch = sent_preamble.unsqueeze(0).to(device)
    with torch.no_grad():
        _, predicted_mean, predicted_std, predicted_nu = model(preamble_batch)
        if is_gaussian:
            predicted_distribution = torch.distributions.Normal(predicted_mean, predicted_std)
        else:
            predicted_distribution = torch.distributions.StudentT(predicted_nu, predicted_mean, predicted_std)
        stochastic_samples = predicted_distribution.sample((num_stochastic_samples,)).squeeze(1).cpu().numpy()
    mean_response = predicted_mean.squeeze(0).cpu().numpy()
    mean_signal_spectrum = np.fft.rfft(mean_response, norm="ortho")[in_band_mask]
    sample_spectra = np.fft.rfft(stochastic_samples, norm="ortho", axis=-1)[:, in_band_mask]
    residual_noise_power = np.mean(np.abs(sample_spectra - mean_signal_spectrum[None, :]) ** 2, axis=0)
    return np.sqrt(residual_noise_power) / (np.abs(mean_signal_spectrum) + 1e-12) * 100.0


def best_probabilistic_channel_runs(channel_exp_dir):
    channel_runs = load_channel_runs(channel_exp_dir)
    probabilistic_runs = [run for run in channel_runs
                          if (run.get("distribution") or "none") in ("gaussian", "students_t")]
    selected_runs = []
    for model_name in dict.fromkeys(run["model"] for run in probabilistic_runs):
        architecture_runs = [run for run in probabilistic_runs if run["model"] == model_name]
        best_run = _best_by(architecture_runs, "val_nll", "val_per_burst_rrmse_pct")
        selected_runs.append((model_name, best_run.get("distribution") or "none", best_run))
    selected_runs.sort(key=lambda selected: _model_type_sort_key((selected[0], selected[1])))
    return selected_runs


# PLOT 10: EVM% vs frequency from empirical E/D validation waveforms
def subcarrier_frequencies_mhz(dataset_path, active_carrier_indices):
    attrs = dict(zarr.open_group(_resolve(dataset_path), mode="r").attrs)
    subcarrier_spacing_hz = float(attrs["f_min_hz"]) / active_carrier_indices[0]
    return active_carrier_indices * subcarrier_spacing_hz / 1e6


def validation_run_evm_vs_frequency(ed_val_exp_dir, validation_run_id, active_carrier_indices):
    validation_group = zarr.open_group(_resolve(ed_val_exp_dir) / "validation.zarr", mode="r")[validation_run_id]
    sent_spectrum = np.fft.rfft(validation_group["sent_time"][:], norm="ortho", axis=-1)[:, active_carrier_indices]
    received_spectrum = np.fft.rfft(validation_group["received_time"][:], norm="ortho", axis=-1)[:, active_carrier_indices]
    residual_power = np.mean(np.abs(sent_spectrum - received_spectrum) ** 2, axis=0)
    signal_power = np.mean(np.abs(sent_spectrum) ** 2, axis=0)
    return np.sqrt(residual_power / (signal_power + 1e-12)) * 100.0


def best_validation_run_for_channel(ed_val_exp_dir, channel_run_id, active_carrier_indices):
    validation_runs = _read_jsonl(_resolve(ed_val_exp_dir) / "runs.jsonl")
    matching_validation_runs = [run for run in validation_runs if run.get("channel_run_id") == channel_run_id]
    if not matching_validation_runs:
        return None
    return min(matching_validation_runs,
               key=lambda run: float(np.nanmean(validation_run_evm_vs_frequency(
                   ed_val_exp_dir, run["run_id"], active_carrier_indices))))["run_id"]




# MAIN
if __name__ == "__main__":
    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    run_configs = sorted(RUN_CONFIGS, key=lambda c: c["dc_ma"])

    # print("Plot 1 - Val RRMSE vs Sent Power (previous, unused) ...")
    # plot_val_rrmse_vs_power(run_configs)

    print("Plot 2 - Channel Val RRMSE% vs parameter count ...")
    plot_val_rrmse_vs_params(run_configs)

    print("Plot 3 - Pareto: channel params vs experimental EVM% ...")
    plot_pareto_evm(run_configs)

    # print("Plot 4 - Channel RRMSE (aggregate vs operating band) vs paired-encoder EVM ...")
    # plot_rrmse_evm_scatter(run_configs)

    # print("Plot 5 - Predicted vs actual actual EVM ...")
    # plot_predicted_vs_actual_evm(run_configs)

    print("Plot 6 - Best E/D end-to-end residual vs estimated noise floor ...")
    plot_best_ed_noise_floor(run_configs)

    # print("Plot 7 - Gaussian wave packet TCN response ...")
    # plot_packet_response(run_configs)

    # print("\nBest GMP ERR term ranking per DC bias ...")
    # print_best_gmp_err(run_configs)

    print(f"\nDone. Figures saved to {PLOT_PATH}/")
