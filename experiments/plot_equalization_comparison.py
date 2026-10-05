'''
Redraws an equalization-comparison figure from the constellations a validation run saved,
for example with a different font size. Writes equalization_comparison.png and .svg next to
the .npz, replacing the ones the validation drew.

    python plot_equalization_comparison.py <run dir>/plots/equalization_constellations.npz [--font-size N]

--font-size sets the axis label size; titles, ticks and annotations scale from it.
'''
import argparse
from pathlib import Path

from modules.equalization_comparison import plot_equalization_comparison
from modules.figure_fonts import FigureFonts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redraw an equalization-comparison figure from its saved constellations")
    parser.add_argument("constellation_path", type=Path)
    parser.add_argument("--font-size", type=float, default=9.0, help="axis label size in points; other text scales from it")
    arguments = parser.parse_args()

    output_stem = arguments.constellation_path.parent / "equalization_comparison"
    plot_equalization_comparison(arguments.constellation_path, output_stem, FigureFonts.from_base(arguments.font_size))
    print(f"wrote {output_stem.with_suffix('.png')} and {output_stem.with_suffix('.svg')}")
