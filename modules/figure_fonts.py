"""Font sizes for the experiment figures, so one base size restyles a whole figure."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FigureFonts:
    suptitle: float
    title: float
    label: float
    tick: float
    legend: float
    annotation: float

    @classmethod
    def from_base(cls, base_size: float) -> "FigureFonts":
        """Axis labels at base_size, titles larger, ticks and legend slightly smaller."""
        return cls(suptitle=1.4 * base_size,
                   title=1.2 * base_size,
                   label=base_size,
                   tick=0.9 * base_size,
                   legend=0.9 * base_size,
                   annotation=base_size)
