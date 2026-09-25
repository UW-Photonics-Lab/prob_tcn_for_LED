'''
Achievable-rate estimate of the best encoder/decoder against conventional one-tap OFDM
equalization, on the live channel at each DC bias listed in rate_estimate.yml.

For each bias:
1. take the lowest-EVM E/D from a previous hardware validation and revalidate it;
   validation.zarr stores the sent and received time-domain symbols and the noise-floor
   replays of one fixed encoded symbol
2. send plain QPSK OFDM with no encoder/decoder at the E/D's measured mean drive power,
   first as one-tap channel-estimation symbols and then as held-out evaluation symbols
3. plot_rate_estimate.py turns these into per-carrier EVM, SNR, achievable rate and
   spectral efficiency for the one-tap baseline, the E/D and the E/D noise floor

    python rate_estimate.py [config.yml]
'''
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import zarr

from pyflux.core.block import Signal
from pyflux.core.chain import Chain
from pyflux.core.experiment import ExperimentalContext
from modules.experimental_blocks import *
from modules.constellation_diagram import get_constellation
from modules.grid_search import EncoderDecoderValidation
from modules.grid_search.base import generate_run_name
from plot_rate_estimate import summarize_rates

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / (sys.argv[1] if len(sys.argv) > 1 else "rate_estimate.yml")
EXP_DIR = HERE.parent / "data/experiments/train_and_validate"
LOG_DIR = HERE.parent / "data/logs"

MAX_SYNC_RETRIES = 3


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def best_encoder_decoder(previous_validation_dir):
    '''The lowest-EVM E/D of a previous validation, rebuilt from that run's config.yaml so
    it is revalidated with exactly the same checkpoint and channel labels.'''
    best_row = min(read_jsonl(previous_validation_dir / "runs.jsonl"), key=lambda row: row["evm_pct"])
    run_config = yaml.safe_load((previous_validation_dir / "runs" / best_row["run_id"] / "config.yaml").read_text())
    ed_model = {"run_id": run_config["model"],
                "arch": run_config["params"],
                "checkpoint": run_config["checkpoint"],
                "channel_form": run_config["channel_form"],
                "channel_run_id": run_config["channel_run_id"],
                "channel_receptive_field": run_config["channel_receptive_field"],
                "channel_distribution": run_config["channel_distribution"]}
    return best_row, ed_model


def send_without_encoder_decoder(chain, sampling_rate, num_symbols):
    '''Send plain OFDM symbols through the live channel. Returns the sent and received
    time-domain symbols (CP and preamble removed), the same format as validation.zarr.'''
    sent_time, received_time = [], []
    for _ in range(num_symbols):
        for _ in range(MAX_SYNC_RETRIES + 1):
            x = chain.run(Signal(data=np.zeros(1), sampling_rate=sampling_rate))
            if not x.artifact_container["sync_outlier"]:
                break

        sent_time.append(x.artifact_container["sent_baseband"])
        received_time.append(x.data)
    return np.stack(sent_time).astype("float32"), np.stack(received_time).astype("float32")


if __name__ == "__main__":
    RUN_NAME = generate_run_name()

    with ExperimentalContext(CONFIG_FILE=CONFIG_FILE, create_log_file=True, run_name=RUN_NAME,
                             log_dir=LOG_DIR) as Exp:
        cfg = Exp.config.RATE_ESTIMATE
        device = Exp.config.RUNTIME.DEVICE
        seed = int(Exp.config.RUNTIME.SEED)

        constellation = get_constellation(cfg.CONSTELLATION)
        subcarrier_spacing = float(cfg.SUBCARRIER_SPACING)
        osc_fs = float(cfg.OSC_SAMPLE_RATE)
        clip_threshold = float(cfg.CLIP_THRESHOLD)
        num_estimation_symbols = int(cfg.ONE_TAP_ESTIMATION_SYMBOLS)
        num_evaluation_symbols = int(cfg.ONE_TAP_EVALUATION_SYMBOLS)

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

        check_channel = CheckChannel(awg_driver=awg, osc_driver=osc, data_channel=3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        parent_dir = EXP_DIR / f"{RUN_NAME}_rate_estimate_{timestamp}"
        parent_dir.mkdir(parents=True, exist_ok=True)
        no_eq_root = zarr.open_group(parent_dir / "no_equalization.zarr", mode="a")
        Exp.log(f"rate_estimate directory: {parent_dir}")

        manifest = {"subcarrier_spacing_hz": subcarrier_spacing, "biases": []}

        for previous_validation_name in cfg.PREVIOUS_VALIDATION_DIRS:
            previous_validation_dir = EXP_DIR / previous_validation_name
            previous_best, ed_model = best_encoder_decoder(previous_validation_dir)

            ed_exp_dir = Path(ed_model["checkpoint"]).parents[2]
            dataset_path = yaml.safe_load((ed_exp_dir / "experiment_config.yaml").read_text())["dataset_path"]
            dataset_attrs = dict(zarr.open_group(dataset_path, mode="r").attrs)

            dc_offset = float(dataset_attrs["dc_offset_A"])
            dc_ma = int(round(dc_offset * 1000))
            f_min = float(dataset_attrs["f_min_hz"])
            f_max = float(dataset_attrs["f_max_hz"])
            preamble_length = int(dataset_attrs["preamble_length"])
            Exp.log(f"Starting {dc_ma} mA ({f_min / 1e6:.1f} to {f_max / 1e6:.1f} MHz): "
                    f"{ed_model['run_id']} ({ed_model['channel_form']}), previous EVM "
                    f"{previous_best['evm_pct']:.2f}% in {previous_validation_name}")

            assert dc_offset < 0.4, f"DC offset {dc_offset} A exceeds safe range for the LED driver"
            pwr_supply.set_25V(voltage=4, current=dc_offset)  # current-limited; never reaches 4 V
            pwr_supply.enable_output()

            def build_modulator(drive_power):
                return ModulateDataOFDM(
                    constellation=constellation, f_min=f_min, f_max=f_max,
                    subcarrier_spacing=subcarrier_spacing, preamble_method="zadoff_chu",
                    awg_table_fraction=cfg.AWG_TABLE_FRACTION,
                    cyclic_prefix_fraction=cfg.CP_LENGTH_FRACTION,
                    upsample_factor=cfg.UPSAMPLE_FACTOR, preamble_length=preamble_length,
                    power_min=drive_power, power_max=drive_power, jitter_power=0.0,
                    clip_threshold=clip_threshold)

            modulate = build_modulator(drive_power=None)
            assert modulate.cyclic_prefix_length == dataset_attrs["cyclic_prefix_length"], \
                "CP length differs from the dataset the E/D was trained on"
            assert modulate.subcarrier_indicies.tolist() == dataset_attrs["active_carrier_indices"], \
                "active carriers differ from the dataset the E/D was trained on"

            demodulate = DemodulateDataOFDM(
                constellation=constellation, f_min=f_min, f_max=f_max,
                subcarrier_spacing=subcarrier_spacing, preamble_method="zadoff_chu",
                baseband_fft_length=modulate.baseband_fft_length,
                cyclic_prefix_length=modulate.cyclic_prefix_length,
                upsample_factor=cfg.UPSAMPLE_FACTOR, debug=False)

            f_AWG = modulate.awg_frequency
            send_waveform = SendWaveform(fs=modulate.fs_out, awg_driver=awg, freq=f_AWG, amplitude=18, offset=0)
            measure_waveform = MeasureWaveform(
                fs_in=modulate.fs_out, fs_out=osc_fs, osc_driver=osc,
                input_signal_frequency=f_AWG, trigger_channel=1, data_channel=3, debug=False)
            resample_waveform = ResampleMeasuredWaveform(
                fs_in=osc_fs, fs_out=modulate.baseband_sampling_rate, debug=False)
            fractional_sync = FractionalSync(
                fs=modulate.baseband_sampling_rate, f_min=f_min, f_max=f_max, debug=False)

            check_channel.run(f"{dc_ma} mA pre-validation")
            osc.set_record_length(200_000)
            osc.set_horizontal_scale(0.2 / f_AWG)
            osc.configure_channel(ch=3, scale=cfg.OSC_SCALE, offset=0)

            validation = EncoderDecoderValidation(
                [ed_model],
                (modulate, send_waveform, measure_waveform, resample_waveform, fractional_sync, demodulate),
                num_trials=int(cfg.VALIDATION_TRIALS), constellation=constellation,
                clip_value=clip_threshold, noise_floor_points=int(cfg.NOISE_FLOOR_POINTS),
                device=device, seed=seed, experiments_dir=parent_dir,
                experiment_name=f"ed_val_{dc_ma}mA", debug=False)
            validation_exp_dir = validation.run()
            validated = read_jsonl(validation_exp_dir / "runs.jsonl")[0]

            # Send the baseline at the drive power the E/D was just measured at, so both
            # cases share the same DC bias and average electrical drive power
            drive_power = float(validated["encoder_power_mean"])
            Exp.log(f"{dc_ma} mA revalidated EVM {validated['evm_pct']:.2f}%, "
                    f"mean drive power {drive_power:.4f}")

            chain = Chain([build_modulator(drive_power), send_waveform, measure_waveform,
                           resample_waveform, fractional_sync, demodulate])
            estimation_sent, estimation_received = send_without_encoder_decoder(
                chain, modulate.fs_out, num_estimation_symbols)
            evaluation_sent, evaluation_received = send_without_encoder_decoder(
                chain, modulate.fs_out, num_evaluation_symbols)

            no_eq_group = no_eq_root.require_group(f"{dc_ma}mA")
            no_eq_group.create_array("estimation_sent_time", data=estimation_sent, overwrite=True)
            no_eq_group.create_array("estimation_received_time", data=estimation_received, overwrite=True)
            no_eq_group.create_array("evaluation_sent_time", data=evaluation_sent, overwrite=True)
            no_eq_group.create_array("evaluation_received_time", data=evaluation_received, overwrite=True)
            no_eq_group.attrs["drive_power"] = drive_power

            check_channel.run(f"{dc_ma} mA post-validation")

            manifest["biases"].append({"dc_ma": dc_ma,
                                       "dataset_path": str(dataset_path),
                                       "previous_validation_dir": str(previous_validation_dir),
                                       "previous_evm_pct": float(previous_best["evm_pct"]),
                                       "validation_exp_dir": str(validation_exp_dir),
                                       "validation_run_id": validated["run_id"],
                                       "drive_power": drive_power,
                                       "no_eq_group": f"{dc_ma}mA"})
            (parent_dir / "rate_estimate.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

        summarize_rates(parent_dir)
        Exp.log(f"rate_estimate complete: {parent_dir}")
        Exp.log(f"replot: python {HERE / 'plot_rate_estimate.py'} {parent_dir}")
