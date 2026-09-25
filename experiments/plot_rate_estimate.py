'''
Per-carrier EVM, SNR, achievable rate and spectral efficiency for a rate_estimate
experiment, and the figure of EVM% vs frequency for each DC bias.

Every EVM is data-aided against the known sent carriers, with error and signal power
averaged over symbols per carrier. Reading SNR_k = 1 / EVM_k^2 as the SNR of an
independent Gaussian subchannel, the achievable rate is

    R = sum_k log2(1 + SNR_k) / T_symbol,    T_symbol = (1 + CP fraction) / subcarrier spacing

and the spectral efficiency is R divided by the used band, num_carriers * spacing.
The preamble is treated as frame overhead and left out of T_symbol.

Cases per bias:
    one_tap_baseline              plain QPSK at the E/D's drive power, least-squares one-tap
                                  equalizer estimated on separate symbols; residual noise,
                                  ISI and nonlinear distortion all count against it
    encoder_decoder               best E/D, decoder output scored directly
    encoder_decoder_noise_floor   one encoded symbol replayed through channel and decoder;
                                  the per-carrier spread is additive noise only, so this is
                                  the noise-limited ceiling of the E/D link

Outputs:
    rate_summary.csv      raw EVM, SNR, rate and spectral efficiency per bias and case
    rate_comparison.csv   per bias: the raw rates and spectral efficiencies side by side,
                          the E/D gain over the one-tap baseline, and each rate as a
                          percentage of the noise-floor ceiling
    evm_vs_frequency.png/.svg

    python plot_rate_estimate.py <rate_estimate experiment dir>
'''
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import zarr

CASE_STYLES = {
    "one_tap_baseline": {"label": "One-tap equalization", "color": "#009E73", "marker": "s", "linestyle": "-"},
    "encoder_decoder": {"label": "E/D end-to-end residual", "color": "#0072B2", "marker": "o", "linestyle": "-"},
    "encoder_decoder_noise_floor": {"label": "E/D estimated noise floor", "color": "#D55E00", "marker": "^", "linestyle": "--"},
}


def carrier_spectrum(time_symbols, active_carrier_indices):
    return np.fft.rfft(np.asarray(time_symbols, dtype=np.float64), norm="ortho", axis=-1)[:, active_carrier_indices]


def evm_pct_per_carrier(sent_spectrum, received_spectrum):
    residual_power = np.mean(np.abs(sent_spectrum - received_spectrum) ** 2, axis=0)
    signal_power = np.mean(np.abs(sent_spectrum) ** 2, axis=0)
    return np.sqrt(residual_power / signal_power) * 100


def one_tap_evm(no_eq_group, active_carrier_indices):
    '''Least-squares one-tap channel estimate per carrier from the estimation symbols,
    applied to the held-out evaluation symbols.'''
    estimation_sent = carrier_spectrum(no_eq_group["estimation_sent_time"][:], active_carrier_indices)
    estimation_received = carrier_spectrum(no_eq_group["estimation_received_time"][:], active_carrier_indices)
    channel_estimate = (np.sum(estimation_received * np.conj(estimation_sent), axis=0)
                        / np.sum(np.abs(estimation_sent) ** 2, axis=0))

    evaluation_sent = carrier_spectrum(no_eq_group["evaluation_sent_time"][:], active_carrier_indices)
    evaluation_received = carrier_spectrum(no_eq_group["evaluation_received_time"][:], active_carrier_indices)
    return evm_pct_per_carrier(evaluation_sent, evaluation_received / channel_estimate)


def encoder_decoder_evm(validation_group, active_carrier_indices):
    sent_spectrum = carrier_spectrum(validation_group["sent_time"][:], active_carrier_indices)
    received_spectrum = carrier_spectrum(validation_group["received_time"][:], active_carrier_indices)
    return evm_pct_per_carrier(sent_spectrum, received_spectrum)


def encoder_decoder_noise_floor_evm(validation_group):
    replays = validation_group["noise_floor_replays"][:]
    reference = validation_group["noise_floor_reference"][:]
    noise_power = np.var(replays, axis=0, ddof=1)
    return np.sqrt(noise_power / np.abs(reference) ** 2) * 100


def achievable_rate_bps(evm_pct, subcarrier_spacing_hz, cyclic_prefix_fraction):
    snr = 1 / (evm_pct / 100) ** 2
    symbol_duration_s = (1 + cyclic_prefix_fraction) / subcarrier_spacing_hz
    return float(np.sum(np.log2(1 + snr)) / symbol_duration_s)


def compare_rates(summary):
    '''One row per bias with the raw rates and spectral efficiencies of every case, the
    E/D gain over the one-tap baseline, and each rate relative to the noise-floor ceiling.'''
    rows = []
    for dc_ma, bias_rows in summary.groupby("dc_ma"):
        by_case = bias_rows.set_index("case")
        baseline = by_case.loc["one_tap_baseline"]
        encoder_decoder = by_case.loc["encoder_decoder"]
        ceiling = by_case.loc["encoder_decoder_noise_floor"]

        rows.append({
            "dc_ma": int(dc_ma),
            "used_band_mhz": float(baseline["used_band_mhz"]),
            "one_tap_rate_mbps": baseline["rate_mbps"],
            "one_tap_spectral_efficiency_bps_per_hz": baseline["spectral_efficiency_bps_per_hz"],
            "ed_rate_mbps": encoder_decoder["rate_mbps"],
            "ed_spectral_efficiency_bps_per_hz": encoder_decoder["spectral_efficiency_bps_per_hz"],
            "noise_floor_rate_mbps": ceiling["rate_mbps"],
            "noise_floor_spectral_efficiency_bps_per_hz": ceiling["spectral_efficiency_bps_per_hz"],
            "ed_rate_gain_mbps": encoder_decoder["rate_mbps"] - baseline["rate_mbps"],
            "ed_spectral_efficiency_gain_bps_per_hz": (encoder_decoder["spectral_efficiency_bps_per_hz"]
                                                       - baseline["spectral_efficiency_bps_per_hz"]),
            "ed_rate_gain_pct": 100 * (encoder_decoder["rate_mbps"] / baseline["rate_mbps"] - 1),
            "one_tap_pct_of_noise_floor_rate": 100 * baseline["rate_mbps"] / ceiling["rate_mbps"],
            "ed_pct_of_noise_floor_rate": 100 * encoder_decoder["rate_mbps"] / ceiling["rate_mbps"],
            "ed_pct_of_gap_to_noise_floor_closed": (100 * (encoder_decoder["rate_mbps"] - baseline["rate_mbps"])
                                                    / (ceiling["rate_mbps"] - baseline["rate_mbps"])),
        })
    return pd.DataFrame.from_records(rows)


def summarize_rates(experiment_dir):
    experiment_dir = Path(experiment_dir)
    manifest = yaml.safe_load((experiment_dir / "rate_estimate.yaml").read_text())
    subcarrier_spacing_hz = float(manifest["subcarrier_spacing_hz"])
    no_eq_root = zarr.open_group(experiment_dir / "no_equalization.zarr", mode="r")

    biases = manifest["biases"]
    fig, axes = plt.subplots(1, len(biases), figsize=(5 * len(biases), 3.8), squeeze=False)

    rows = []
    for ax, bias in zip(axes[0], biases):
        dataset_attrs = dict(zarr.open_group(bias["dataset_path"], mode="r").attrs)
        active_carrier_indices = np.asarray(dataset_attrs["active_carrier_indices"])
        frequencies_mhz = active_carrier_indices * subcarrier_spacing_hz / 1e6
        used_band_hz = len(active_carrier_indices) * subcarrier_spacing_hz

        validation_root = zarr.open_group(Path(bias["validation_exp_dir"]) / "validation.zarr", mode="r")
        validation_group = validation_root[bias["validation_run_id"]]
        fft_length = validation_group["sent_time"].shape[1]
        cyclic_prefix_fraction = int(dataset_attrs["cyclic_prefix_length"]) / fft_length

        evm_by_case = {
            "one_tap_baseline": one_tap_evm(no_eq_root[bias["no_eq_group"]], active_carrier_indices),
            "encoder_decoder": encoder_decoder_evm(validation_group, active_carrier_indices),
            "encoder_decoder_noise_floor": encoder_decoder_noise_floor_evm(validation_group),
        }

        for case_name, evm_pct in evm_by_case.items():
            rate_bps = achievable_rate_bps(evm_pct, subcarrier_spacing_hz, cyclic_prefix_fraction)
            rows.append({"dc_ma": bias["dc_ma"],
                         "case": case_name,
                         "used_band_mhz": used_band_hz / 1e6,
                         "rms_evm_pct": float(np.sqrt(np.mean(evm_pct ** 2))),
                         "mean_snr_db": float(np.mean(-20 * np.log10(evm_pct / 100))),
                         "rate_mbps": rate_bps / 1e6,
                         "spectral_efficiency_bps_per_hz": rate_bps / used_band_hz})

            style = CASE_STYLES[case_name]
            ax.plot(frequencies_mhz, evm_pct, color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], markersize=2.5, linewidth=1.0, label=style["label"])

        ax.set_title(f"{bias['dc_ma']} mA")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("EVM (%)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle("Per-Carrier EVM%: One-Tap Baseline, E/D Residual and E/D Noise Floor")
    fig.tight_layout()
    fig.savefig(experiment_dir / "evm_vs_frequency.png", dpi=300, bbox_inches="tight")
    fig.savefig(experiment_dir / "evm_vs_frequency.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame.from_records(rows)
    summary.to_csv(experiment_dir / "rate_summary.csv", index=False)
    comparison = compare_rates(summary)
    comparison.to_csv(experiment_dir / "rate_comparison.csv", index=False)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print()
    print(comparison.set_index("dc_ma").T.to_string(float_format=lambda value: f"{value:.3f}"))
    return summary, comparison


if __name__ == "__main__":
    summarize_rates(sys.argv[1])
