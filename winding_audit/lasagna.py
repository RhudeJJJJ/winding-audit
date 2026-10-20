"""Read published Lasagna winding-density and normal channels.

Conventions follow VC3D (volume-cartographer/core/src/lasagna) and the spiral
fitter (spiral-fitting/losses.py) exactly:

* a channel group's grid spacing is 2**scaledown * source_to_base base voxels,
  and a base-voxel point p maps to channel index p / spacing (no half-voxel
  offset), sampled trilinearly;
* winding density (windings per base voxel) = u8 * grad_mag_factor /
  grad_mag_encode_scale; a zero byte means "no prediction";
* a normal is decoded from compact (nx, ny) bytes as (raw - 128) / 127 with
  nz = sqrt(max(0, 1 - nx^2 - ny^2)), then normalised; both bytes zero means
  "no prediction". Normals are sign-free (nz >= 0 is only a storage convention).

Only the chunks that the requested points touch are fetched (a curved surface
touches a small fraction of its bounding box), and sampling is vectorised.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .zarr_io import ChunkCache, ZarrArray, join, read_bytes, read_json, s3_to_https


@dataclass
class SparseChannel:
    """Chunks of one uint8 channel held in a stack, addressed through a lookup grid."""

    array: ZarrArray
    spacing: float
    lut: np.ndarray = field(init=False)
    stack: list = field(default_factory=list)

    def __post_init__(self):
        grid = tuple(-(-s // c) for s, c in zip(self.array.shape, self.array.chunks))
        self.lut = np.full(grid, -1, dtype=np.int32)
        # slot 0 is the fill chunk for keys outside the array
        self.stack = [np.full(self.array.chunks, self.array.fill_value, dtype=np.uint8)]
        self._stacked = None

    def ensure(self, keys: np.ndarray, workers: int = 8) -> None:
        keys = np.unique(keys, axis=0)
        grid = np.array(self.lut.shape)
        inside = ((keys >= 0) & (keys < grid)).all(axis=1)
        keys = keys[inside]
        keys = keys[self.lut[keys[:, 0], keys[:, 1], keys[:, 2]] < 0]
        if len(keys) == 0:
            return
        arr = self.array

        def fetch(k):
            key = tuple(int(v) for v in k)
            return key, arr._decode(read_bytes(arr._chunk_url(key), arr.cache))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for key, chunk in ex.map(fetch, keys):
                self.lut[key] = len(self.stack)
                self.stack.append(np.ascontiguousarray(chunk, dtype=np.uint8))
        self._stacked = None

    def stacked(self) -> np.ndarray:
        if self._stacked is None or len(self._stacked) != len(self.stack):
            self._stacked = np.stack(self.stack)
        return self._stacked

    def gather(self, z: np.ndarray, y: np.ndarray, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Bytes at integer indices; entries outside ``mask`` are returned as 0 unread."""
        if mask is not None:
            out = np.zeros(z.shape, dtype=np.uint8)
            if mask.any():
                out[mask] = self.gather(z[mask], y[mask], x[mask])
            return out
        cz, cy, cx = self.array.chunks
        kz, ky, kx = z // cz, y // cy, x // cx
        grid = self.lut.shape
        inside = (kz >= 0) & (ky >= 0) & (kx >= 0) & (kz < grid[0]) & (ky < grid[1]) & (kx < grid[2])
        slot = np.zeros(z.shape, dtype=np.int64)
        slot[inside] = self.lut[kz[inside], ky[inside], kx[inside]]
        if (slot < 0).any():
            # fetch what the caller did not prefetch, then look the slots up again
            miss = slot < 0
            self.ensure(np.stack([kz[miss], ky[miss], kx[miss]], axis=1))
            slot[inside] = self.lut[kz[inside], ky[inside], kx[inside]]
        chunk_bytes = int(np.prod(self.array.chunks))
        if chunk_bytes * len(self.stack) <= 512 * 2**20:
            return self.stacked()[slot, z % cz, y % cy, x % cx]
        # large chunk sets (CT): read chunk by chunk instead of stacking them
        flat = slot.ravel()
        order = np.argsort(flat, kind="stable")
        bounds = np.flatnonzero(np.diff(flat[order])) + 1
        out = np.empty(flat.shape, dtype=np.uint8)
        lz, ly, lx = (z % cz).ravel(), (y % cy).ravel(), (x % cx).ravel()
        for grp in np.split(order, bounds):
            if len(grp):
                out[grp] = self.stack[flat[grp[0]]][lz[grp], ly[grp], lx[grp]]
        return out.reshape(z.shape)


class LasagnaFields:
    def __init__(self, manifest_url: str, cache: ChunkCache | None = None):
        self.manifest_url = s3_to_https(manifest_url)
        self.manifest = read_json(self.manifest_url, cache)
        base = self.manifest_url.rsplit("/", 1)[0]
        m = self.manifest
        self.source_to_base = float(m.get("source_to_base", 1.0))
        self.encode_scale = float(m["grad_mag_encode_scale"])
        self.grad_mag_factor = float(m["grad_mag_factor"])
        self.channels: dict[str, SparseChannel] = {}
        for name, g in m["groups"].items():
            chans = g.get("channels", [name])
            if len(chans) != 1:
                raise ValueError("multi-channel lasagna groups are not supported")
            arr = ZarrArray(join(base, g["zarr"]), cache)
            if arr.dtype != np.uint8 or len(arr.shape) != 3:
                raise ValueError(f"channel {chans[0]!r} must be a 3D uint8 zarr")
            spacing = float(1 << int(g.get("scaledown", 0))) * self.source_to_base
            self.channels[chans[0]] = SparseChannel(arr, spacing)
        for need in ("grad_mag", "nx", "ny"):
            if need not in self.channels:
                raise ValueError(f"lasagna manifest lacks channel {need!r}")
        sp = {self.channels[c].spacing for c in ("grad_mag", "nx", "ny")}
        if len(sp) != 1:
            raise ValueError("grad_mag, nx and ny must share one sampling grid")
        self.spacing = sp.pop()
        self.density_per_byte = self.grad_mag_factor / self.encode_scale

    # -- chunk management --------------------------------------------------
    def _corner_index(self, pts_xyz: np.ndarray, spacing: float):
        q = pts_xyz / spacing
        return np.floor(q[:, 2]).astype(np.int64), np.floor(q[:, 1]).astype(np.int64), np.floor(q[:, 0]).astype(np.int64), q

    def prefetch(self, pts_xyz: np.ndarray, channels=("grad_mag", "nx", "ny")) -> int:
        pts_xyz = pts_xyz[np.isfinite(pts_xyz).all(axis=1)]
        n = 0
        for c in channels:
            ch = self.channels[c]
            z0, y0, x0, _ = self._corner_index(pts_xyz, ch.spacing)
            cz, cy, cx = ch.array.chunks
            keys = []
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        keys.append(np.stack([(z0 + dz) // cz, (y0 + dy) // cy, (x0 + dx) // cx], axis=1))
            keys = np.unique(np.concatenate(keys), axis=0)
            ch.ensure(keys)
            n += len(keys)
        return n

    # -- sampling ------------------------------------------------------------
    def _cube(self, pts_xyz: np.ndarray, spacing: float, shape):
        z0, y0, x0, q = self._corner_index(pts_xyz, spacing)
        inside = (
            np.isfinite(q).all(axis=1)
            & (z0 >= 0) & (y0 >= 0) & (x0 >= 0)
            & (z0 + 1 < shape[0]) & (y0 + 1 < shape[1]) & (x0 + 1 < shape[2])
        )
        z0 = np.where(inside, z0, 0)
        y0 = np.where(inside, y0, 0)
        x0 = np.where(inside, x0, 0)
        fz = np.where(inside, q[:, 2] - z0, 0.0)
        fy = np.where(inside, q[:, 1] - y0, 0.0)
        fx = np.where(inside, q[:, 0] - x0, 0.0)
        return (z0, y0, x0), (fz, fy, fx), inside

    def sample(self, pts_xyz: np.ndarray, orient_xyz: np.ndarray | None = None):
        """Density (windings/base voxel), unit normal (x, y, z) and validity at points.

        With ``orient_xyz`` the eight corner normals are flipped to agree with
        that reference (e.g. the surface normal) before the trilinear average,
        so the result carries the reference's orientation. Without one, the
        principal eigenvector of the interpolated axis tensor n n^T is used,
        as VC3D does. A point is valid when all eight corners carry
        predictions in every channel.
        """
        gch, xch, ych = self.channels["grad_mag"], self.channels["nx"], self.channels["ny"]
        (z0, y0, x0), (fz, fy, fx), inside = self._cube(pts_xyz, self.spacing, gch.array.shape)
        n_pts = len(pts_xyz)
        dens = np.zeros(n_pts)
        acc = np.zeros((n_pts, 3))
        tensor = np.zeros((n_pts, 3, 3)) if orient_xyz is None else None
        valid = inside.copy()
        for dz in (0, 1):
            wz = fz if dz else 1.0 - fz
            for dy in (0, 1):
                wy = fy if dy else 1.0 - fy
                for dx in (0, 1):
                    wx = fx if dx else 1.0 - fx
                    w = wz * wy * wx
                    zz, yy, xx = z0 + dz, y0 + dy, x0 + dx
                    gb = gch.gather(zz, yy, xx, inside).astype(np.float64)
                    rx = xch.gather(zz, yy, xx, inside).astype(np.float64)
                    ry = ych.gather(zz, yy, xx, inside).astype(np.float64)
                    valid &= (gb > 0) & ((rx != 0) | (ry != 0))
                    dens += w * gb
                    nx = (rx - 128.0) / 127.0
                    ny = (ry - 128.0) / 127.0
                    nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, None))
                    n = np.stack([nx, ny, nz], axis=1)
                    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
                    if orient_xyz is not None:
                        s = np.sign(np.einsum("ij,ij->i", n, orient_xyz))
                        s[s == 0] = 1.0
                        acc += (w * s)[:, None] * n
                    else:
                        tensor += w[:, None, None] * n[:, :, None] * n[:, None, :]
        dens *= self.density_per_byte
        if orient_xyz is None:
            _, vecs = np.linalg.eigh(tensor)
            normal = vecs[:, :, -1]
        else:
            normal = acc / np.linalg.norm(acc, axis=1, keepdims=True).clip(1e-12)
        return dens, normal, valid

    def sample_channel(self, name: str, pts_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Trilinear sample of any single uint8 channel (raw bytes), with validity."""
        ch = self.channels[name]
        (z0, y0, x0), (fz, fy, fx), inside = self._cube(pts_xyz, ch.spacing, ch.array.shape)
        out = np.zeros(len(pts_xyz))
        for dz in (0, 1):
            wz = fz if dz else 1.0 - fz
            for dy in (0, 1):
                wy = fy if dy else 1.0 - fy
                for dx in (0, 1):
                    wx = fx if dx else 1.0 - fx
                    out += wz * wy * wx * ch.gather(z0 + dz, y0 + dy, x0 + dx, inside)
        out[~inside] = np.nan
        return out, inside
