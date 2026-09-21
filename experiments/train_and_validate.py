'''
This is the primary experiment for:

1. Gathering data at a chosen DC offset
2. Training channel models
3. Training encoders/decoders (E/Ds) on chosen channel models 
4. Validating trained E/Ds in experiment 

The exact behavior of this experiment is set by modifyign the parameters in the train_and_validate.yml file
'''
import csv
import random
import time
from pathlib import Path

import numpy as np

from pyflux.core.experiment import ExperimentalContext
from pyflux.core.block import Signal
from modules.experimental_blocks import *
from modules.constellation_diagram import get_constellation
from modules.grid_search import (
    ChannelModelGridSearch, EncoderDecoderGridSearch, select_channel_models,
    select_encoder_decoders, EncoderDecoderValidation,
)
from modules.grid_search.base import generate_run_name, format_duration
from modules.utils import OFDMConfig

HERE = Path(__file__).resolve().parent

CONFIG_FILE = HERE / "train_and_validate.yml"
EXP_DIR = HERE.parent / "data/experiments/train_and_validate"
LOG_DIR = HERE.parent / "data/logs"

SELECTED_CHANNEL_RUN_IDS = None  # If none, pick automatiicaly 
SELECTED_ED_RUN_IDS = None  

VALIDATION_TRIALS = 40           # The number of validation trials for each E/D in experiment to get a better estimate of the performance over many OFDM symbols
MAX_SYNC_RETRIES = 3             # re-collect a capture this many times if sync_outlier is flagged

if __name__ == "__main__":
    RUN_NAME = generate_run_name()

    with ExperimentalContext(CONFIG_FILE=CONFIG_FILE,
                             create_log_file=True,
                             run_name=RUN_NAME,
                             log_dir=LOG_DIR) as Exp:
        cfg = Exp.config.DATA_COLLECTION
        device = Exp.config.RUNTIME.DEVICE
        seed = int(Exp.config.RUNTIME.SEED)
        Exp.log(f"Run name: {RUN_NAME}")

        awg = Exp.Agilent_33250A
        osc = Exp.Tektronix_TDS2000
        pwr_supply = Exp.HP_EE3631A

        awg.set_output_load("INF")
        osc.display_channel(ch=1)
        osc.display_channel(ch=3)
        osc.set_record_length(200_000)
        osc.set_trigger(ch=1, voltage_level=0)
        osc.set_probe_gain(ch=1, gain=1)
        osc.set_probe_gain(ch=3, gain=1)
        osc.configure_channel(ch=3, scale=cfg.OSC_SCALE, offset=0)
        osc.configure_channel(ch=1, scale=1, offset=0)
        osc.set_coupling(ch=1, coupling="AC")
        osc.set_coupling(ch=3, coupling="AC")

        # single DC offset per dataset
        dc_offset_A = float(cfg.DC_OFFSETS[0])
        assert dc_offset_A < 0.4, f"DC offset {dc_offset_A} A exceeds safe range for the LED driver"
        min_freq = float(cfg.F_MINS[0])
        max_freq = float(cfg.F_MAXS[0])
        osc_fs = float(cfg.OSC_SAMPLE_RATES[0])
        subcarrier_spacing = float(cfg.SUBCARRIER_SPACING)

        pwr_supply.set_25V(voltage=4, current=dc_offset_A)  # current-limited; never reaches 4 V
        pwr_supply.enable_output()
        Exp.log(f"DC offset set to {dc_offset_A:.3f} A")
        check_channel = CheckChannel(awg_driver=awg, osc_driver=osc, data_channel=3)

        mod_ofdm = ModulateDataOFDM(
            constellation=get_constellation(cfg.MODULATION_FORMAT),
            f_min=min_freq, f_max=max_freq,
            subcarrier_spacing=subcarrier_spacing,
            preamble_method="zadoff_chu",
            awg_table_fraction=cfg.AWG_TABLE_FRACTION,
            cyclic_prefix_fraction=cfg.CP_LENGTH_FRACTION,
            upsample_factor=cfg.UPSAMPLE_FACTOR,
            preamble_length=cfg.PREAMBLE_LENGTH,
            power_min=getattr(cfg, 'POWER_MIN', None),
            power_max=getattr(cfg, 'POWER_MAX', None),
            jitter_power=getattr(cfg, 'JITTER_POWER', 0.0),
            clip_threshold=float(cfg.CLIP_THRESHOLD),
        )

        f_AWG = mod_ofdm.awg_frequency
        Exp.log(f"AWG frequency {f_AWG:.2f} Hz")
        osc.set_horizontal_scale(0.2 / f_AWG)
        send_waveform = SendWaveform(
            fs=mod_ofdm.fs_out, awg_driver=awg, freq=f_AWG, amplitude=18, offset=0)
        measure_waveform = MeasureWaveform(
            fs_in=mod_ofdm.fs_out, fs_out=osc_fs, osc_driver=osc,
            input_signal_frequency=f_AWG, trigger_channel=1, data_channel=3, debug=False)
        resample_waveform = ResampleMeasuredWaveform(
            fs_in=osc_fs, fs_out=mod_ofdm.baseband_sampling_rate, debug=False)
        fractional_sync = FractionalSync(
            fs=mod_ofdm.baseband_sampling_rate, f_min=min_freq, f_max=max_freq, debug=False)
        demod_ofdm = DemodulateDataOFDM(
            constellation=get_constellation(cfg.MODULATION_FORMAT),
            f_min=min_freq, f_max=max_freq, subcarrier_spacing=subcarrier_spacing,
            preamble_method="zadoff_chu",
            baseband_fft_length=mod_ofdm.baseband_fft_length,
            cyclic_prefix_length=mod_ofdm.cyclic_prefix_length,
            upsample_factor=cfg.UPSAMPLE_FACTOR, debug=False)

        def _restore_ofdm_scope():
            osc.set_record_length(200_000)
            osc.set_horizontal_scale(0.2 / f_AWG)
            osc.configure_channel(ch=3, scale=cfg.OSC_SCALE, offset=0)

        check_channel.run("start")
        _restore_ofdm_scope()

        # 1. gather the dataset
        dataset_path = getattr(cfg, 'DATASET_PATH', None) or None
        if dataset_path is None:
            append_to_dataset = AppendToDataset(
                fs_in=mod_ofdm.baseband_sampling_rate, dc_offset=dc_offset_A,
                f_min=min_freq, f_max=max_freq,
                active_carrier_indices=mod_ofdm.subcarrier_indicies,
                cyclic_prefix_length=mod_ofdm.cyclic_prefix_length,
                preamble_length=mod_ofdm.preamble_length,
                modulation_format=cfg.MODULATION_FORMAT, dataset_path=None,
                run_name=RUN_NAME,
                power_min=getattr(cfg, 'POWER_MIN', None),
                power_max=getattr(cfg, 'POWER_MAX', None),
                jitter_power=getattr(cfg, 'JITTER_POWER', None),
                clip_threshold=getattr(cfg, 'CLIP_THRESHOLD', None))
            dataset_path = append_to_dataset.dataset_path
            send_and_receive = SendAndReceiveOFDM(
                mod_ofdm, send_waveform, measure_waveform, resample_waveform, demod_ofdm,
                fractional_sync_block=fractional_sync)
            collection_start = time.time()
            sync_outliers = 0
            for i in range(cfg.N):
                point_start = time.time()

                for attempt in range(MAX_SYNC_RETRIES + 1):
                    x = send_and_receive.run(Signal(data=np.zeros(1), sampling_rate=mod_ofdm.fs_out))
                    if not x.artifact_container.get("sync_outlier"):
                        break
                    sync_outliers += 1
                    residual = x.artifact_container["residual_delay_samples"]
                    retries_left = MAX_SYNC_RETRIES - attempt
                    if retries_left > 0:
                        Exp.log(f"sync outlier on point {i + 1} (residual {residual:+.3f} samples), "
                                f"re-collecting ({retries_left} left)")
                else:
                    Exp.log(f"sync retries exhausted on point {i + 1} (residual {residual:+.3f}), "
                            f"passing over this point")
                    continue

                x = append_to_dataset.run(x)
                point_seconds = time.time() - point_start
                elapsed = time.time() - collection_start
                eta = elapsed / (i + 1) * (cfg.N - i - 1)
                Exp.log(f"gathered {i + 1}/{cfg.N}  frame={x.data.shape}  "
                        f"{format_duration(point_seconds)}  "
                        f"[elapsed {format_duration(elapsed)}, eta {format_duration(eta)}]")
            Exp.log(f"dataset saved to {dataset_path}")
            if sync_outliers:
                Exp.log(f"sync outliers detected and re-collected during collection: {sync_outliers}")
        else:
            Exp.log(f"reusing existing dataset at {dataset_path}")

        check_channel.run("post-collection")

        # 2. channel-model grid search
        channel_exp_dir_override = getattr(Exp.config, 'CHANNEL_EXP_DIR', None)
        if channel_exp_dir_override:
            channel_exp_dir = Path(channel_exp_dir_override)
            Exp.log(f"reusing channel models from {channel_exp_dir}")
        else:
            channel_gs = ChannelModelGridSearch.from_experiment_config(
                CONFIG_FILE, experiment_name="channel_models", experiments_dir=EXP_DIR,
                dataset_path=dataset_path, run_prefix=RUN_NAME)
            channel_exp_dir = channel_gs.run()

        # 3. select channel models: 
        # SELECTION_MODE settings: 
        # "all" 
        # "best" (one winner per channel form)
        # "best_per_size" (best per int(log_2(param_count)) bin)

        channel_gs_config = Exp.config.CHANNEL_GRID_SEARCH
        selection_mode = getattr(channel_gs_config, "SELECTION_MODE")
        prob_selection_key = getattr(channel_gs_config, "PROB_SELECTION_KEY")

        best_channels = select_channel_models(
            channel_exp_dir, mode=selection_mode, prob_metric=prob_selection_key,
            run_ids=SELECTED_CHANNEL_RUN_IDS)
        for cm in best_channels:
            dist = cm["params"].get("distribution", "none")
            print(f"  channel {cm['model']:4s}  dist={dist:12s}  run={cm['run_id']}")

        # 4. encoder/decoder grid search
        ed_exp_dir_override = getattr(Exp.config, 'ENCODER_DECODER_EXP_DIR', None)
        if ed_exp_dir_override:
            ed_exp_dir = Path(ed_exp_dir_override)
            Exp.log(f"reusing E/D models from {ed_exp_dir}")
        else:
            ed_gs = EncoderDecoderGridSearch.from_experiment_config(
                CONFIG_FILE, channel_models=best_channels,
                experiment_name="encoder_decoder", experiments_dir=EXP_DIR,
                dataset_path=dataset_path, run_prefix=RUN_NAME)
            ed_exp_dir = ed_gs.run(ofdm_config=OFDMConfig.from_modulator(mod_ofdm))

        # 5. select E/D models and validate on the live channel. Validation order is
        # shuffled so slow channel drift over a multi-hour sweep does not
        # correlate with grid order

        ed_models = select_encoder_decoders(
            ed_exp_dir, channel_exp_dir=channel_exp_dir, run_ids=SELECTED_ED_RUN_IDS)
        random.Random(seed).shuffle(ed_models)
        for m in ed_models:
            print(f"  E/D {m['run_id']}  channel_form={m['channel_form']}")

        check_channel.run("pre-validation")
        _restore_ofdm_scope()

        # Validation transmits clean symbols with its own constellation
        val_constellation = get_constellation(Exp.config.ENCODER_DECODER_VALIDATION.CONSTELLATION)
        mod_ofdm_val = ModulateDataOFDM(
            constellation=val_constellation,
            f_min=min_freq, f_max=max_freq,
            subcarrier_spacing=subcarrier_spacing,
            preamble_method="zadoff_chu",
            awg_table_fraction=cfg.AWG_TABLE_FRACTION,
            cyclic_prefix_fraction=cfg.CP_LENGTH_FRACTION,
            upsample_factor=cfg.UPSAMPLE_FACTOR,
            preamble_length=cfg.PREAMBLE_LENGTH,
            power_min=None, power_max=None, jitter_power=0.0,
            clip_threshold=float(cfg.CLIP_THRESHOLD),
        )
        demod_ofdm_val = DemodulateDataOFDM(
            constellation=val_constellation,
            f_min=min_freq, f_max=max_freq, subcarrier_spacing=subcarrier_spacing,
            preamble_method="zadoff_chu",
            baseband_fft_length=mod_ofdm_val.baseband_fft_length,
            cyclic_prefix_length=mod_ofdm_val.cyclic_prefix_length,
            upsample_factor=cfg.UPSAMPLE_FACTOR, debug=False)

        validation = EncoderDecoderValidation(
            ed_models,
            (mod_ofdm_val, send_waveform, measure_waveform, resample_waveform,
             fractional_sync, demod_ofdm_val),
            num_trials=VALIDATION_TRIALS,
            constellation=val_constellation,
            clip_value=float(cfg.CLIP_THRESHOLD),
            noise_floor_points=int(getattr(Exp.config.ENCODER_DECODER_VALIDATION, "NOISE_FLOOR_POINTS", 0)),
            device=device, seed=seed, experiments_dir=EXP_DIR, experiment_name="ed_validation",
            run_prefix=RUN_NAME, debug=False)

        validation_exp_dir_override = getattr(Exp.config, 'VALIDATION_EXP_DIR', None)
        if validation_exp_dir_override:
            validation.exp_dir = Path(validation_exp_dir_override)
            validation.runs_dir = validation.exp_dir / "runs"
            validation.summary_dir = validation.exp_dir / "summary"
            validation.runs_jsonl = validation.exp_dir / "runs.jsonl"
            Exp.log(f"resuming validation in {validation.exp_dir}")

        val_exp_dir = validation.run()
        check_channel.run("post-validation")



        # Final printout of results
        print("\n" + "=" * 60)
        print(f"Results summary: DC offset {dc_offset_A:.3f} A")
        print("=" * 60)
        for label, summary_dir in [("channel", channel_exp_dir), ("encoder/decoder", ed_exp_dir), ("validation", val_exp_dir)]:
            
            lb = Path(summary_dir) / "summary" / "leaderboard.csv"
            rows = list(csv.DictReader(open(lb)))
            metric = next((m for m in ("evm_pct", "evm", "per_burst_rrmse_pct", "rrmse_pct") if rows and m in rows[0]), "per_burst_rrmse_pct")
            
            print(f"\n  {label} leaderboard (sorted by {metric}):")
            for row in sorted(rows, key=lambda r: float(r[metric])):

                extra = f"  ber={float(row['ber']):.4f}" if row.get("ber") else ""
                dist = row.get("distribution") or row.get("channel_distribution") or row.get("channel_form") or "?"
                clip = ""

                if row.get("clip_penalty_weight") not in (None, ""):
                    clip = f"  clip_w={float(row['clip_penalty_weight']):g} rho={float(row['clip_penalty_rho']):g}"
                print(f"    {row['run_id']}  {metric}={float(row[metric]):.6f}{extra}  "
                      f"dist={dist}  params={row['num_params']}  t={row['train_seconds']}s")
