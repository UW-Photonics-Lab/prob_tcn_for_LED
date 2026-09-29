'''
Synthetic end-to-end test of the channel model and encoder/decoder grid searches.
No hardware, no zarr dataset — all data is generated here.  

Run from the repo root:
    python experiments/test_real_gridsearch.py
'''
import csv
import os
import sys
from pathlib import Path
import zarr
import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Ensure repo root is on the path regardless of launch location
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

from modules.constellation_diagram import get_constellation
from modules.grid_search import (
    ChannelModelGridSearch, EncoderDecoderGridSearch, select_channel_models,
    MODEL_REGISTRY,
)
from modules.models import TCN
from modules.utils import OFDMConfig
from modules.experimental_blocks import ModulateDataOFDM

from pyflux.core.block import Signal

# ── Experiment output ────────────────────────────────────────────────────────
EXP_DIR   = Path("data/experiments/test_real_gridsearch")
DEVICE    = "cuda"
SEED      = 42

# ── Experiment Dataset ────────────────────────────────────────────────────────
SENT_BURST = "data/dc0.052A_fmin300000_fmax7.6e+06_20260630_1743.zarr/sent_burst"
RECIEVED_BURST = "data/dc0.052A_fmin300000_fmax7.6e+06_20260630_1743.zarr/received_burst"

# ── Synthetic OFDM geometry ──────────────────────────────────────────────────
K_MIN            = 3
K_MAX            = 76
SUBCARRIER_SPACING_HZ = 10e4


ofdm_modulator = ModulateDataOFDM(
    constellation=get_constellation("qpsk"),
    f_min=K_MIN * SUBCARRIER_SPACING_HZ,
    f_max=K_MAX * SUBCARRIER_SPACING_HZ,
    subcarrier_spacing=SUBCARRIER_SPACING_HZ,
    preamble_method="zadoff_chu",
    awg_table_fraction=0.5,
    cyclic_prefix_fraction=0.125,
    upsample_factor=4,
    preamble_length=256
)

OFDM_CONFIG = OFDMConfig.from_modulator(ofdm_modulator)

def read_dataset(n_frames: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    sent_arr = zarr.open(SENT_BURST, mode='r')
    recieved_arr = zarr.open(RECIEVED_BURST, mode='r')
    """
    Shape: (5000, 940)
    Data Type: float32
    Chunks: (64, 940)
    Subset shape: (100, 940)
    """
    sent_time = torch.from_numpy(sent_arr[:n_frames, :]).float()
    recv_time = torch.from_numpy(recieved_arr[:n_frames, :]).float()

    return sent_time, recv_time

    

CHANNEL_GRID = {
    "VAL_FRACTION": 0.2,
    "models": [
        {
            "model": "tcn",
            "params": {
                "nlayers":          3,
                "dilation_base":    2,
                "kernel_size":      5,
                "hidden_channels":  16,
                "learn_noise":      True,
                "gaussian":         True,
                "epochs":           100,
                "lr":               1e-3,
                "batch_size":       16,
                "activation":       "relu",
            },
        },
    ]
}

ENCODER_DECODER_GRID = {
    "constellation":      "qpsk",
    "preamble_amplitude": 3.0,
    "Mix-Match_Archs": False,
    "models": [
        # {
        #     "model": "tcn_ae",
        #     "params": {
        #             "nlayers":         3,
        #             "dilation_base":   2,
        #             "kernel_size":     5,
        #             "hidden_channels": 8,
        #             "epochs":          1000,
        #             "lr":              1e-3,
        #             "weight_decay":    1e-5,
        #             "batch_size":      8,
        #             "activation":       "relu",
        #             "quantization":     None
        #     },
        # },
        {
            "model": "Qxx_tcn",
            "params": {
                    "nlayers":         2,
                    "dilation_base":   2,
                    "kernel_size":     3,
                    "hidden_channels": 4,
                    "epochs":          1000,
                    "lr":              1e-3,
                    "weight_decay":    1e-5,
                    "batch_size":      8,
                    "activation":       "relu",
                    "quantization":     [{"frac_bits": 8, "data_width": 16}, {"frac_bits": 7, "data_width": 14}, {"frac_bits": 7, "data_width": 13}, {"frac_bits": 8, "data_width": 12}, {"frac_bits": 6, "data_width": 11}, {"frac_bits": 5, "data_width": 10}, {"frac_bits": 5, "data_width": 9}, {"frac_bits": 4, "data_width": 8}, {"frac_bits": 3, "data_width": 7}, {"frac_bits": 3, "data_width": 6}, {"frac_bits": 2, "data_width": 5}, {"frac_bits": 2, "data_width": 4}],
            },
        },
        {
            "model": "adder_tcn",
            "params": {
                    "nlayers":         2,
                    "dilation_base":   2,
                    "kernel_size":     3,
                    "hidden_channels": 4,
                    "epochs":          1000,
                    "lr":              0.1, #cosine aneealing start if cosine anealing enabled
                    "weight_decay":    1e-5,
                    "batch_size":      8,
                    "activation":       "relu",
                    "quantization":     [{"frac_bits": 8, "data_width": 16}, {"frac_bits": 7, "data_width": 14}, {"frac_bits": 7, "data_width": 13}, {"frac_bits": 8, "data_width": 12}, {"frac_bits": 6, "data_width": 11}, {"frac_bits": 5, "data_width": 10}, {"frac_bits": 5, "data_width": 9}, {"frac_bits": 4, "data_width": 8}, {"frac_bits": 3, "data_width": 7}, {"frac_bits": 3, "data_width": 6}, {"frac_bits": 2, "data_width": 5}, {"frac_bits": 2, "data_width": 4}],
                    "cosineAnnealing_min_lr": 1e-6,
            },
        },
        # {
        #     "model": "shift_tcn",
        #     "params": {
        #             "nlayers":         3,
        #             "dilation_base":   2,
        #             "kernel_size":     5,
        #             "hidden_channels": 8,
        #             "epochs":          1000,
        #             "lr":              1e-3,
        #             "weight_decay":    1e-5,
        #             "batch_size":      8,
        #             "activation":       "relu",
        #             "quantization":     [{"N": 1, "B": 4, "data_width": 4}, {"N": 2, "B": 4, "data_width": 8}],
        #     },
        # },
    ]
}


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 1: channel model grid search")
    print("=" * 60)

    X, Y = read_dataset(n_frames=5000)
    print(f"Real dataset: X{tuple(X.shape)}  Y{tuple(Y.shape)}")

    channel_gs = ChannelModelGridSearch(
        CHANNEL_GRID,
        dataset_path="data\dc0.052A_fmin300000_fmax7.6e+06_20260630_1743.zarr", 
        experiments_dir=EXP_DIR,
        device=DEVICE,
        seed=SEED,
        experiment_name="channel_models",
    )
    channel_exp_dir = channel_gs.run(data=(X, Y))
    print(f"\nChannel grid done → {channel_exp_dir}")

    print("\n" + "=" * 60)
    print("Stage 2: select channel models (best per architecture, by val loss)")
    print("=" * 60)

    # SELECTED_RUN_IDS = ["tcn_xxxxxxxx"]  # pin specific runs instead of best-per-family
    SELECTED_RUN_IDS = None
    best_channels = select_channel_models(channel_exp_dir, mode="best", run_ids=SELECTED_RUN_IDS)
    all_channels  = select_channel_models(channel_exp_dir, mode="all")
    channel_models_by_id = {cm["run_id"]: cm for cm in all_channels}

    for cm in best_channels:
        print(f"  {cm['model']:4s}  run={cm['run_id']}")
    print(f"\n  {len(all_channels)} total channel runs available; using best per arch for E/D stage.")

    print("\n" + "=" * 60)
    print("Stage 3: encoder/decoder grid search")
    print("=" * 60)

    ed_gs = EncoderDecoderGridSearch(
        ENCODER_DECODER_GRID,
        channel_models=best_channels,
        dataset_path="data\dc0.052A_fmin300000_fmax7.6e+06_20260630_1743.zarr",
        experiments_dir=EXP_DIR,
        device=DEVICE,
        seed=SEED,
        experiment_name="encoder_decoder",
    )
    ed_exp_dir = ed_gs.run(ofdm_config=OFDM_CONFIG)
    print(f"\nEncoder/decoder grid done → {ed_exp_dir}")

    print("\n" + "=" * 60)
    print("Results summary")
    print("=" * 60)

    for label, summary_dir in [("channel", channel_exp_dir), ("encoder/decoder", ed_exp_dir)]:
        lb = Path(summary_dir) / "summary" / "leaderboard.csv"
        rows = list(csv.DictReader(open(lb)))
        metric = next((m for m in ("evm_pct", "per_burst_rrmse_pct", "rrmse_pct")
                        if rows and m in rows[0]), "per_burst_rrmse_pct")  # channel ranks by rrmse, E/D by EVM
        print(f"\n  {label} leaderboard (sorted by {metric}):")
        for row in sorted(rows, key=lambda r: float(r[metric])):
            extra = f"  ber={float(row['ber']):.4f}" if row.get("ber") else ""
            print(f"    {row['run_id']}  {metric}={float(row[metric]):.6f}{extra}  "
                    f"params={row['num_params']}  t={row['train_seconds']}s")
