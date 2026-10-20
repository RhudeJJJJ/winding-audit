"""Independent check of a flagged site against the raw CT.

The Lasagna fields are a model's reading of the scan; the CT is the scan.
For each flagged site this module renders the CT on a plane spanned by the
direction across the suspected switch (in the surface) and the sheet normal,
and marks where the surface cuts that plane. A real sheet switch shows the
surface leaving one bright papyrus layer, crossing the dark gap, and
continuing on the neighbouring layer; a false alarm shows the surface staying
on one layer.
"""

from __future__ import annotations

import numpy as np

from .lasagna import SparseChannel
from .zarr_io import ChunkCache, ZarrArray, join, read_bytes, s3_to_https


class CTVolume:
    def __init__(self, volume_url: str, level: int = 0, cache: ChunkCache | None = None,
                 cache_chunks: bool = False):
        self.url = s3_to_https(volume_url).rstrip("/")
        self.level = level
        self.scale = float(1 << level)
        # CT chunks are large (2 MB at 128^3) and rarely re-read: by default
        # only the array metadata goes through the disk cache
        arr = ZarrArray(join(self.url, str(level)), cache, workers=8)
        if not cache_chunks:
            arr.cache = None
        if arr.dtype != np.uint8:
            raise ValueError("CT volume level must be uint8")
        self.channel = SparseChannel(arr, self.scale)

    def sample(self, pts_xyz: np.ndarray) -> np.ndarray:
        """Trilinear CT intensity; NaN outside, bounded decoded-chunk memory.

        Group all corner requests by chunk and fetch at most eight chunks
        concurrently. No CT chunk is retained after this call. Unlike a dense
        bounding box or an accumulating SparseChannel, curved-surface reads
        cannot retain gigabytes of unused CT in memory.
        """
        from concurrent.futures import ThreadPoolExecutor

        pts_xyz = np.asarray(pts_xyz, dtype=float)
        if pts_xyz.ndim != 2 or pts_xyz.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        out = np.full(len(pts_xyz), np.nan)
        arr = self.channel.array
        q = pts_xyz / self.scale
        finite = np.isfinite(q).all(axis=1)
        safe = np.where(finite[:, None], q, 0.0)
        xyz0 = np.floor(safe).astype(np.int64)
        shape_xyz = np.array(arr.shape[::-1])
        inside = finite & (xyz0 >= 0).all(axis=1) & (xyz0 + 1 < shape_xyz).all(axis=1)
        ids = np.flatnonzero(inside)
        if not len(ids):
            return out
        base = xyz0[ids]
        frac = safe[ids] - base
        coords, weights, indices = [], [], []
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    corner = np.array([dx, dy, dz])
                    weight = np.prod(np.where(corner, frac, 1.0 - frac), axis=1)
                    use = weight > 0
                    coords.append((base[use] + corner)[:, ::-1])
                    weights.append(weight[use])
                    indices.append(ids[use])
        coords = np.concatenate(coords)
        weights = np.concatenate(weights)
        indices = np.concatenate(indices)
        chunk_shape = np.array(arr.chunks)
        keys, group = np.unique(coords // chunk_shape, axis=0, return_inverse=True)
        order = np.argsort(group, kind="stable")
        boundaries = np.searchsorted(group[order], np.arange(len(keys) + 1))
        out[ids] = 0.0

        def fetch(k):
            key = tuple(int(v) for v in keys[k])
            return k, arr._decode(read_bytes(arr._chunk_url(key), arr.cache))

        with ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0, len(keys), 8):
                # Bounded batches avoid executor.map queuing/retaining a full
                # surface worth of completed chunks behind a slow request.
                for k, block in pool.map(fetch, range(start, min(start + 8, len(keys)))):
                    ii = order[boundaries[k]:boundaries[k + 1]]
                    local = coords[ii] % chunk_shape
                    values = block[local[:, 0], local[:, 1], local[:, 2]]
                    np.add.at(out, indices[ii], weights[ii] * values)
        return out


def plane_frame(center: np.ndarray, across: np.ndarray, normal: np.ndarray):
    """Orthonormal (u, v, w): u along ``across`` made orthogonal to the normal, v = normal."""
    v = normal / np.linalg.norm(normal)
    u = across - np.dot(across, v) * v
    u /= np.linalg.norm(u)
    w = np.cross(u, v)
    return u, v, w


def render_plane(ct: CTVolume, center, u, v, half_u: float, half_v: float, step: float = 1.0):
    us = np.arange(-half_u, half_u + 1e-9, step)
    vs = np.arange(-half_v, half_v + 1e-9, step)
    U, V = np.meshgrid(us, vs)
    pts = center[None, :] + U.reshape(-1, 1) * u[None, :] + V.reshape(-1, 1) * v[None, :]
    img = ct.sample(pts).reshape(U.shape)
    return img, us, vs


def straightened_section(ct: CTVolume, points: np.ndarray, normals: np.ndarray, half_depth: float,
                         depth_step: float = 1.0, sub: int = 4):
    """CT along a grid line, resampled along the surface normal at every point.

    ``points``/``normals`` are the vertices of one grid line (N, 3). The line
    is subdivided ``sub`` times per edge (normals blended and renormalised),
    and at every position the CT is sampled at offsets t in
    [-half_depth, half_depth] along the normal. Returns the image (depth x
    positions), the fractional vertex index of every column, and the offsets.
    A surface that stays on one sheet keeps a bright layer at offset 0; a
    sheet switch shifts the whole layer pattern by one spacing.
    """
    ok = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
    idx = np.arange(len(points), dtype=float)
    pos, nrm, frac = [], [], []
    for i in range(len(points) - 1):
        if not (ok[i] and ok[i + 1]):
            continue
        n0, n1 = normals[i], normals[i + 1]
        if np.dot(n0, n1) < 0:
            n1 = -n1
        for j in range(sub):
            a = j / sub
            pos.append((1 - a) * points[i] + a * points[i + 1])
            n = (1 - a) * n0 + a * n1
            nrm.append(n / np.linalg.norm(n))
            frac.append(idx[i] + a)
    pos, nrm, frac = np.array(pos), np.array(nrm), np.array(frac)
    ts = np.arange(-half_depth, half_depth + 1e-9, depth_step)
    pts = pos[None, :, :] + ts[:, None, None] * nrm[None, :, :]
    img = ct.sample(pts.reshape(-1, 3)).reshape(len(ts), len(pos))
    return img, frac, ts


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return np.nan
    a = a[ok] - a[ok].mean()
    b = b[ok] - b[ok].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def _best_shift(ref: np.ndarray, mov: np.ndarray, max_shift: int, core: int):
    """Shift s (samples) maximising NCC of ref[t] and mov[t + s] over a central window."""
    n = len(ref)
    c0, c1 = n // 2 - core, n // 2 + core
    best = (np.nan, -np.inf)
    scores = {}
    for s in range(-max_shift, max_shift + 1):
        if c0 + s < 0 or c1 + s > n:
            continue
        v = _ncc(ref[c0:c1], mov[c0 + s:c1 + s])
        scores[s] = v
        if np.isfinite(v) and v > best[1]:
            best = (s, v)
    return best[0], best[1], scores


def layer_drift(img: np.ndarray, ts: np.ndarray, max_step: int = 6, core: float | None = None):
    """Cumulative displacement of the mesh relative to the CT layer stack.

    Consecutive columns of a straightened section are registered by the
    depth shift that maximises their normalised cross-correlation (|shift| <=
    ``max_step`` samples between neighbouring columns, which keeps the
    estimate unambiguous in a periodic layer stack when columns are closely
    spaced). The running sum is the drift of the surface across the layers in
    voxels: ~0 for a surface that stays on its sheet, about one sheet spacing
    across a sheet switch. Returns drift per column (voxels) and the per-step
    correlation.
    """
    step = float(ts[1] - ts[0])
    core_n = int((core if core is not None else 0.4 * (ts[-1] - ts[0])) / step / 2)
    drift = np.zeros(img.shape[1])
    corr = np.full(img.shape[1], np.nan)
    acc = 0.0
    for j in range(1, img.shape[1]):
        # b[t + s] ~ a[t]: the layers near the surface moved by s samples
        s, v, _ = _best_shift(img[:, j - 1], img[:, j], max_step, core_n)
        if not np.isfinite(v):
            drift[j:] = np.nan
            break
        acc += s * step
        drift[j] = acc
        corr[j] = v
    return drift, corr


def direct_shift(img: np.ndarray, ts: np.ndarray, left: slice, right: slice, max_shift: float,
                 core: float = 60.0):
    """Depth shift of the CT layer stack between two stretches of a section.

    The mean depth profiles of the columns in ``left`` and ``right`` are
    compared by normalised cross-correlation over a window of +-``core``
    voxels around the surface; the shift maximising it (refined to sub-sample
    by a parabola through the peak) is how far the layers moved relative to
    the surface between the two stretches. Returns shift (voxels), its
    correlation, and the best correlation at least ``0.5 * max_shift`` away
    from it (a second, competing alignment indicates ambiguity).
    """
    step = float(ts[1] - ts[0])
    a = np.nanmean(img[:, left], axis=1)
    b = np.nanmean(img[:, right], axis=1)
    n = len(a)
    c = int(core / step)
    m = int(max_shift / step)
    mid = n // 2
    shifts = np.arange(-m, m + 1)
    scores = np.full(len(shifts), np.nan)
    for i, s in enumerate(shifts):
        lo, hi = mid - c, mid + c
        if lo + s < 0 or hi + s > n:
            continue
        scores[i] = _ncc(a[lo:hi], b[lo + s:hi + s])
    if not np.isfinite(scores).any():
        return np.nan, np.nan, np.nan
    i = int(np.nanargmax(scores))
    s = float(shifts[i])
    if 0 < i < len(shifts) - 1 and np.isfinite(scores[i - 1]) and np.isfinite(scores[i + 1]):
        y0, y1, y2 = scores[i - 1], scores[i], scores[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            s += 0.5 * (y0 - y2) / den
    far = np.abs(shifts - shifts[i]) >= 0.5 * m
    rival = float(np.nanmax(scores[far])) if np.isfinite(scores[far]).any() else np.nan
    return s * step, float(scores[i]), rival


def surface_cut(points: np.ndarray, valid: np.ndarray, center, u, v, w, half_u, half_v, slab: float):
    """Surface vertices within ``slab`` of the plane, in plane coordinates (u, v)."""
    p = points[valid] - center[None, :]
    du, dv, dw = p @ u, p @ v, p @ w
    sel = (np.abs(dw) <= slab) & (np.abs(du) <= half_u) & (np.abs(dv) <= half_v)
    return du[sel], dv[sel]
