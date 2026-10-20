"""Load tifxyz quad surfaces (x.tif, y.tif, z.tif, optional mask.tif, meta.json).

A vertex is valid when all three coordinates are finite, not the -1 sentinel,
z > 0 (as volume-cartographer's loader treats z <= 0 cells as invalid), and,
if mask.tif exists, its mask pixel is non-zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import tifffile

from .zarr_io import ChunkCache, join, read_bytes, s3_to_https


@dataclass
class Surface:
    points: np.ndarray  # (rows, cols, 3) float64, x y z in base voxels
    valid: np.ndarray  # (rows, cols) bool
    meta: dict
    source: str

    @property
    def shape(self):
        return self.valid.shape


def _read_tif(url: str, cache: ChunkCache | None) -> np.ndarray | None:
    data = read_bytes(url, cache)
    if data is None:
        return None
    import io

    return tifffile.imread(io.BytesIO(data))


def load_tifxyz(path: str, cache: ChunkCache | None = None) -> Surface:
    path = s3_to_https(path).rstrip("/")
    x = _read_tif(join(path, "x.tif"), cache)
    y = _read_tif(join(path, "y.tif"), cache)
    z = _read_tif(join(path, "z.tif"), cache)
    if x is None or y is None or z is None:
        raise FileNotFoundError(f"{path}: x.tif/y.tif/z.tif missing")
    pts = np.stack([x, y, z], axis=-1).astype(np.float64)
    valid = np.isfinite(pts).all(axis=-1)
    valid &= ~((pts[..., 0] == -1) & (pts[..., 1] == -1) & (pts[..., 2] == -1))
    valid &= pts[..., 2] > 0
    mask = _read_tif(join(path, "mask.tif"), cache)
    if mask is not None:
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != valid.shape:
            raise ValueError(f"{path}: mask shape {mask.shape} differs from surface {valid.shape}")
        valid &= mask != 0
    meta_raw = read_bytes(join(path, "meta.json"), cache)
    meta = json.loads(meta_raw) if meta_raw else {}
    pts[~valid] = np.nan
    return Surface(points=pts, valid=valid, meta=meta, source=path)


def grid_normals(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-vertex unit normal from grid central differences (one-sided at borders).

    The orientation is the grid's own: cross(d/dcol, d/drow). It is consistent
    across the surface wherever the parametrisation does not fold.
    """
    p = points
    def diff(axis):
        fwd = np.roll(p, -1, axis=axis) - p
        bwd = p - np.roll(p, 1, axis=axis)
        vf = valid & np.roll(valid, -1, axis=axis)
        vb = valid & np.roll(valid, 1, axis=axis)
        if axis == 0:
            vf[-1, :] = False
            vb[0, :] = False
        else:
            vf[:, -1] = False
            vb[:, 0] = False
        d = np.where((vf & vb)[..., None], 0.5 * (fwd + bwd), np.where(vf[..., None], fwd, bwd))
        ok = vf | vb
        return d, ok

    du, oku = diff(1)
    dv, okv = diff(0)
    n = np.cross(du, dv)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    ok = oku & okv & (nn[..., 0] > 1e-9)
    n = np.where(ok[..., None], n / np.where(nn > 0, nn, 1.0), np.nan)
    return n
