'''
Base class that the automatic grid searches extend.

It handles the bookkeeping shared by every sweep:
  - Run naming: readable adj_noun names, plus a deterministic run_id (hash of
    the point's config) so a given config always maps to the same folder.
  - An experiment directory per sweep, cataloging the config manifest
    (git hash, seed, host, timestamp, grid), per-run training metrics, and plots.
  - Resumable runs: points that already have metrics.json are skipped on re-run,
    so a crashed or interrupted sweep continues where it left off.
  - Reproducibility: seeds random/numpy/torch and records the git hash + seed.
  - Aggregation: one row per run in runs.jsonl, ranked into leaderboard.csv.

To extend it, a subclass supplies `self.points` and overrides `_run_point()`
(train + eval + save one point, returning a metrics dict); `_prepare()` is
optional for loading shared data once, and `self.rank_by` sets the leaderboard
sort key.
'''

import csv
import hashlib
import json
import random
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from matplotlib.figure import Figure

import numpy as np
import torch
import yaml

_REPO_DIR = Path(__file__).resolve().parents[2]

_ADJECTIVES = [
    "amber", "azure", "bold", "brisk", "calm", "crisp", "dawn", "deep",
    "deft", "dim", "dry", "dusk", "fair", "fast", "firm", "fleet", "free",
    "fresh", "full", "glad", "gold", "grey", "high", "keen", "kind", "late",
    "lean", "light", "long", "low", "mild", "mint", "mist", "mute", "neat",
    "new", "nice", "nifty", "noble", "north", "old", "open", "pale", "plain",
    "prime", "pure", "quick", "quiet", "raw", "rich", "ripe", "rough", "round",
    "royal", "safe", "sage", "sharp", "sleek", "slim", "slow", "small", "smart",
    "soft", "solar", "solid", "spare", "stark", "steady", "still", "stone",
    "swift", "tall", "tame", "taut", "thin", "tidy", "tiny", "true", "vast",
    "warm", "wide", "wild", "wise", "young",
]
_NOUNS = [
    "arc", "bay", "beam", "birch", "brook", "cliff", "cloud", "coast", "comet",
    "creek", "crest", "dawn", "dell", "drift", "dune", "dust", "echo", "elm",
    "fern", "field", "fjord", "flame", "flare", "fleet", "flint", "flood",
    "flow", "foam", "fold", "forge", "frost", "gale", "glade", "glen", "glow",
    "gorge", "grove", "gulf", "haze", "heath", "hill", "isle", "lake", "lane",
    "leaf", "ledge", "mast", "mesa", "mist", "moon", "moor", "moss", "peak",
    "pine", "plain", "pond", "pulse", "rain", "reef", "ridge", "rift", "ring",
    "rise", "river", "rock", "root", "rush", "sand", "sea", "shade", "shore",
    "slope", "snow", "sol", "spark", "spire", "spring", "star", "stem", "stone",
    "storm", "stream", "summit", "sun", "surf", "tide", "trail", "vale", "vault",
    "wave", "wind", "wood",
]


def generate_run_name() -> str:
    """Return a short random name in the style of wandb run names (adj_noun)."""
    return f"{random.choice(_ADJECTIVES)}_{random.choice(_NOUNS)}"


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def run_id(point: dict) -> str:
    digest = hashlib.md5(json.dumps(point, sort_keys=True).encode()).hexdigest()[:8]
    return f"{point['model']}_{digest}"


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class GridSearchBase:
    def __init__(self, points, grid_manifest, shared_params, experiments_dir,
                 device, seed, experiment_name, extra_manifest=None, run_prefix=None):
        self.points = points
        self.grid_manifest = grid_manifest
        self.shared_params = shared_params
        self.device = device
        self.seed = seed
        self.extra_manifest = extra_manifest or {}
        self.all_metrics = []

        base = Path(experiments_dir) if experiments_dir else _REPO_DIR / "data" / "experiments"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        prefix = f"{run_prefix}_" if run_prefix else ""
        self.exp_dir = base / f"{prefix}{experiment_name}_{timestamp}"
        self.runs_dir = self.exp_dir / "runs"
        self.summary_dir = self.exp_dir / "summary"
        self.runs_jsonl = self.exp_dir / "runs.jsonl"

    # ------------------------------------------------------------------ setup
    def setup(self):
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "experiment_dir": str(self.exp_dir),
            "git_hash": _git_hash(),
            "seed": self.seed,
            "device": self.device,
            "host": socket.gethostname(),
            "created": datetime.now().isoformat(timespec="seconds"),
            "n_points": len(self.points),
            "grid": self.grid_manifest,
            **self.extra_manifest,
        }
        with open(self.exp_dir / "experiment_config.yaml", "w") as f:
            yaml.safe_dump(manifest, f, sort_keys=False)
        print(f"[grid] experiment dir: {self.exp_dir}  ({len(self.points)} points)")
        return self

    # ----------------------------------------------------------------- helpers
    def _seed_everything(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _write_history(self, run_dir: Path, history: dict):
        if not history:
            return
        n = len(next(iter(history.values())))
        with open(run_dir / "history.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", *history.keys()])
            for i in range(n):
                w.writerow([i, *(history[k][i] for k in history)])
        self._plot_history(run_dir, history)
        if "lr" in history:
            self._plot_lr(run_dir, history["lr"])

    def _plot_history(self, run_dir: Path, history: dict):
        '''Plots the ML training plots for the models'''
        labels = {"loss": "train", "val_loss": "validation", "train_nll": "train (true NLL)"}
        series = {k: v for k, v in history.items() if k.endswith("loss") or k == "train_nll"}
        if not series:
            return
        epochs = range(len(next(iter(series.values()))))
        fig = Figure(figsize=(7, 4))
        ax = fig.subplots()

        if "train_nll" in series:
            labels["val_loss"] = "validation (true NLL)"
            handles = []
            for key in ("train_nll", "val_loss"):
                if key in series:
                    handles += ax.plot(epochs, series[key], label=labels[key], marker=".", ms=3)
            ax.set_ylabel("true NLL")
            ax_beta = ax.twinx()
            handles += ax_beta.plot(epochs, series["loss"], label="train (β-NLL objective)",
                                    color="C2", ls="--", marker=".", ms=3)
            ax_beta.set_ylabel("β-NLL")
            ax.legend(handles=handles)
        else:
            for key, values in series.items():
                ax.plot(epochs, values, label=labels.get(key, key), marker=".", ms=3)
            ax.set_ylabel("loss")
            ax.legend()

        ax.set_xlabel("epoch")
        ax.set_title(run_dir.name)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plots_dir / "loss.png", dpi=120)

    def _plot_lr(self, run_dir: Path, lr_values: list):
        fig = Figure(figsize=(7, 3))
        ax = fig.subplots()
        ax.plot(range(len(lr_values)), lr_values, marker=".", ms=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.set_title(run_dir.name)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
        fig.savefig(run_dir / "plots" / "lr.png", dpi=120)

    def _append_run_record(self, rid, point, metrics):
        record = {"run_id": rid, "model": point["model"], **point["params"], **metrics}
        with open(self.runs_jsonl, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _write_leaderboard(self):
        if not self.runs_jsonl.exists():
            return
        rows = [json.loads(line) for line in self.runs_jsonl.read_text().splitlines() if line.strip()]
        if not rows:
            return
        rank_by = getattr(self, "rank_by", None)
        if rank_by:
            rows.sort(key=lambda r: r.get(rank_by, float("inf")))
        cols = list(dict.fromkeys(k for r in rows for k in r))  # union, order-preserving
        with open(self.summary_dir / "leaderboard.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

        # md table
        rows.sort(key=lambda r: r.get("rrmse_pct", float("inf")))
        with open(self.summary_dir / "tableRanking.md", "w") as file:
            file.write("\n" + "| ")
            for header in cols:
                file.write(header[:3] + " | ")
            file.write("\n" + "| " + ("--- | ")*len(cols) + "\n")

            for row in rows:
                values = [str(row.get(header, "")) for header in cols]
                file.write("| " + " | ".join(values) + " |\n")

            file.write("\n\nLong:\n\n")

            file.write("\n" + "| ")
            for header in cols:
                file.write(header + " | ")
            file.write("\n" + "| " + ("--- | ")*len(cols) + "\n")

            for row in rows:
                values = [str(row.get(header, "")) for header in cols]
                file.write("| " + " | ".join(values) + " |\n")


    # ------------------------------------------------------------- subclass hooks
    def _prepare(self, **kwargs):
        '''Load whatever data/models every grid point needs, once.'''
        return None

    def _run_point(self, point, run_dir, context):
        '''Train + evaluate + save one grid point. Must return a metrics dict.'''
        raise NotImplementedError

    # --------------------------------------------------------------------- run
    def run(self, **prepare_kwargs):
        self.setup()
        context = self._prepare(**prepare_kwargs)

        point_seconds = []  # per-point wall time for points run this session, for the ETA
        for i, point in enumerate(self.points):
            rid = run_id(point)
            run_dir = self.runs_dir / rid
            metrics_path = run_dir / "metrics.json"
            progress = f"({i + 1}/{len(self.points)})"
            if metrics_path.exists():
                stored = json.loads(metrics_path.read_text()).get("train_seconds")
                took = f"  {format_duration(stored)}" if stored is not None else ""
                print(f"[grid] {progress} skip {rid} (done{took})")
                continue

            (run_dir / "plots").mkdir(parents=True, exist_ok=True)
            with open(run_dir / "config.yaml", "w") as f:
                yaml.safe_dump({**point, "shared": self.shared_params}, f, sort_keys=False)

            self._seed_everything()
            t0 = time.time()
            metrics = self._run_point(point, run_dir, context)
            metrics["train_seconds"] = round(time.time() - t0, 3)

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            self._append_run_record(rid, point, metrics)

            self.all_metrics.append(metrics)
            point_seconds.append(metrics["train_seconds"])
            elapsed = format_duration(sum(point_seconds))
            eta = format_duration(np.mean(point_seconds) * (len(self.points) - i - 1))
            print(f"[grid] {progress} done {rid}  {format_duration(metrics['train_seconds'])}"
                  f"  [session {elapsed}, eta {eta}]")

        self._write_leaderboard()
        return self.exp_dir
