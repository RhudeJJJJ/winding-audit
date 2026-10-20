"""Windowed winding steps along the grid lines of a surface.

Integrating edge crossings over the whole surface accumulates every small
bias of the predicted sheet normals, and a non-integrable defect spreads
into a dipole over the entire map. A sheet switch, in contrast, is local:
somewhere along a grid line the surface climbs onto the neighbouring winding
within a short distance. So the detector only integrates over short windows.

For each grid line (rows use horizontal edges, columns vertical edges) and
half-window h (in edges), the net crossing over the window centred on
vertex k is

    D_h(k) = sum of edge crossings from k - h to k + h,

computed only where the window has no missing edge. A switch at k0 gives
|D_h| close to the calibrated value of one winding for every window that
contains k0, while a bias b per edge gives only 2 h b. Each vertex keeps the
strongest step of either direction at each scale, with its sign and the
direction in which it was seen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .edges import EdgeField


@dataclass
class StepMap:
    scales: tuple  # half-windows in edges
    step: np.ndarray  # (S, R, C) signed net crossing, strongest of both directions
    direction: np.ndarray  # (S, R, C) 0 = along rows (horizontal edges), 1 = along columns


def _window_sums(cross: np.ndarray, ok: np.ndarray, h: int) -> np.ndarray:
    """Net crossing over [k - h, k + h) edges around each vertex k along axis 1.

    ``cross`` has shape (R, C - 1) (edge k joins vertex k and k + 1). Returns
    (R, C) with NaN where the window leaves the line or meets a missing edge.
    """
    R, E = cross.shape
    C = E + 1
    c = np.where(ok, cross, 0.0)
    bad = (~ok).astype(np.int64)
    S = np.zeros((R, E + 1))
    S[:, 1:] = np.cumsum(c, axis=1)
    B = np.zeros((R, E + 1), dtype=np.int64)
    B[:, 1:] = np.cumsum(bad, axis=1)
    out = np.full((R, C), np.nan)
    k = np.arange(C)
    lo = k - h
    hi = k + h
    good = (lo >= 0) & (hi <= E)
    kk = k[good]
    s = S[:, hi[good]] - S[:, lo[good]]
    nb = B[:, hi[good]] - B[:, lo[good]]
    s = np.where(nb == 0, s, np.nan)
    out[:, kk] = s
    return out


def step_map(eh: EdgeField, ev: EdgeField, scales=(4, 8, 16)) -> StepMap:
    R = ev.valid.shape[0] + 1
    C = eh.valid.shape[1] + 1
    steps = np.full((len(scales), R, C), np.nan)
    dirs = np.full((len(scales), R, C), -1, dtype=np.int8)
    for i, h in enumerate(scales):
        a = _window_sums(eh.crossing, eh.valid, h)  # along rows
        b = _window_sums(ev.crossing.T, ev.valid.T, h).T  # along columns
        aa = np.where(np.isfinite(a), np.abs(a), -1.0)
        bb = np.where(np.isfinite(b), np.abs(b), -1.0)
        use_b = bb > aa
        steps[i] = np.where(use_b, b, a)
        dirs[i] = np.where(use_b, 1, np.where(aa >= 0, 0, -1))
    return StepMap(tuple(scales), steps, dirs)
