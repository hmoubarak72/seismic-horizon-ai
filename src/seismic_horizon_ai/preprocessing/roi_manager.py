"""Region of interest (ROI) in survey units, converted to index slices."""

from __future__ import annotations

import dataclasses

import numpy as np

from ..seismic_io.survey_geometry import SurveyGeometry


@dataclasses.dataclass(frozen=True)
class ROI:
    """Inclusive ranges in line numbers and milliseconds. ``None`` means full extent."""

    inline_range: tuple[int, int] | None = None
    crossline_range: tuple[int, int] | None = None
    time_range_ms: tuple[float, float] | None = None

    def to_slices(self, g: SurveyGeometry) -> tuple[slice, slice, slice]:
        def axis_slice(axis: np.ndarray, rng: tuple[float, float] | None, name: str) -> slice:
            if rng is None:
                return slice(0, axis.size)
            lo, hi = sorted(rng)
            idx = np.nonzero((axis >= lo) & (axis <= hi))[0]
            if idx.size == 0:
                raise ValueError(f"ROI {name} range {rng} does not intersect the survey ({axis[0]}..{axis[-1]}).")
            return slice(int(idx[0]), int(idx[-1]) + 1)

        return (
            axis_slice(g.ilines, self.inline_range, "inline"),
            axis_slice(g.xlines, self.crossline_range, "crossline"),
            axis_slice(g.samples_ms, self.time_range_ms, "time"),
        )
