"""
Equalization-comparison figure: the constellation as it moves Sent -> Encoded -> Received ->
Decoded, with the encoder/decoder and without (raw OFDM, no ML in the loop).

The symbols are saved to an .npz first and the figure is drawn from that file, so it can be
restyled later (for example with a different font size) without rerunning the hardware.
"""
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from modules.figure_fonts import FigureFonts
from modules.utils import evm_pct

STAGE_NAMES = ["Sent", "Encoded", "Received", "Decoded"]
DEFAULT_FONTS = FigureFonts.from_base(9)


def save_equalization_constellations(constellation_path: Path, channel_form: str,
                                     ed_stages: dict[str, list[np.ndarray]], no_ed_sent: list[np.ndarray],
                                     no_ed_received: list[np.ndarray], carrier_frequencies_hz: np.ndarray) -> None:
    """Each stage is stored as [trial, carrier] complex symbols."""
    np.savez(constellation_path,
             channel_form=np.array(channel_form),
             carrier_frequencies_hz=np.asarray(carrier_frequencies_hz),
             ed_sent=np.stack(ed_stages["Sent"]),
             ed_encoded=np.stack(ed_stages["Encoded"]),
             ed_received=np.stack(ed_stages["Received"]),
             ed_decoded=np.stack(ed_stages["Decoded"]),
             no_ed_sent=np.stack(no_ed_sent),
             no_ed_received=np.stack(no_ed_received))


def unit_power(symbols: np.ndarray) -> np.ndarray:
    flattened = symbols.ravel()
    return flattened / (np.sqrt(np.mean(np.abs(flattened) ** 2)) + 1e-12)


def plot_equalization_comparison(constellation_path: Path, output_stem: Path, fonts: FigureFonts) -> None:
    """Writes output_stem.png and output_stem.svg. Each panel is normalized to unit average
    power so the shape is visible; the Decoded panel overlays the ideal reference and its EVM."""
    data = np.load(constellation_path)
    channel_form = str(data["channel_form"])
    carrier_frequencies_hz = data["carrier_frequencies_hz"]

    ed_stages = {"Sent": data["ed_sent"], "Encoded": data["ed_encoded"],
                 "Received": data["ed_received"], "Decoded": data["ed_decoded"]}
    # Without an E/D, encoding is the identity and there is no decoder, so the Encoded panel
    # repeats Sent and the Decoded panel repeats the raw Received
    no_ed_stages = {"Sent": data["no_ed_sent"], "Encoded": data["no_ed_sent"],
                    "Received": data["no_ed_received"], "Decoded": data["no_ed_received"]}
    rows = [("With Encoder/Decoder", ed_stages), ("Without Encoder/Decoder", no_ed_stages)]

    # Constrained layout makes room for the labels and colorbar at any font size
    fig = Figure(figsize=(14, 7.5), layout="constrained")
    axes = fig.subplots(2, 4)
    for row_index, (row_label, stages) in enumerate(rows):
        row_evm = float(evm_pct(stages["Sent"].ravel(), stages["Decoded"].ravel()))
        reference = unit_power(stages["Sent"])
        num_trials = stages["Sent"].shape[0]
        color = np.tile(carrier_frequencies_hz, num_trials)

        for column_index, stage_name in enumerate(STAGE_NAMES):
            ax = axes[row_index][column_index]
            symbols = unit_power(stages[stage_name])
            scatter = ax.scatter(symbols.real, symbols.imag, s=8, c=color, cmap="viridis")

            if stage_name == "Decoded":
                ax.scatter(reference.real, reference.imag, s=45, marker="x", c="red", linewidth=1.3, zorder=5)
                ax.text(0.04, 0.96, f"received EVM = {row_evm:.1f}%", transform=ax.transAxes,
                        ha="left", va="top", fontsize=fonts.tick, weight="bold",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"))

            if row_index == 0:
                ax.set_title(stage_name, fontsize=fonts.title)
            if column_index == 0:
                ax.set_ylabel(f"{row_label}\n\nQuadrature", fontsize=fonts.label)
            ax.set_xlabel("In-Phase", fontsize=fonts.label)
            ax.tick_params(labelsize=fonts.tick)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", "box")

    colorbar = fig.colorbar(scatter, ax=axes, fraction=0.02, pad=0.02)
    colorbar.set_label("Carrier Frequency (Hz)", fontsize=fonts.label)
    colorbar.ax.tick_params(labelsize=fonts.tick)
    fig.suptitle("Comparison of Encoder Decoder Equalization to No Equalization "
                 f"(Trained on {channel_form} Channel Model)", fontsize=fonts.suptitle)
    # fig.supxlabel("Each panel normalized to unit average power. "
    #               "Red x marks the ideal reference constellation.", fontsize=fonts.annotation)

    fig.savefig(output_stem.with_suffix(".png"), dpi=130, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), format="svg", bbox_inches="tight")
