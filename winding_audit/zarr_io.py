"""Minimal read-only Zarr v2 access for OME-Zarr levels on S3/HTTPS or local disk.

Only what the audit needs: read a dense axis-aligned box of a uint8 array,
fetching every chunk the box touches. Chunks are cached on disk by their
exact URL so a rerun reads nothing from the network. Missing chunks (HTTP 404
or absent files) are the array's fill value, as Zarr v2 specifies.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np
import requests
from urllib3.util.retry import Retry

_SESSION_LOCAL = threading.local()


def _session() -> requests.Session:
    s = getattr(_SESSION_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        retry = Retry(total=4, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=("GET",))
        adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=retry)
        s.mount("https://", adapter)
        _SESSION_LOCAL.session = s
    return s


def is_remote(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def s3_to_https(path: str) -> str:
    """Map s3://bucket/key to the public HTTPS endpoint (anonymous read)."""
    if path.startswith("s3://"):
        bucket, _, key = path[5:].partition("/")
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return path


def join(base: str, *parts: str) -> str:
    out = base.rstrip("/")
    for p in parts:
        out = out + "/" + p.strip("/")
    return out


class ChunkCache:
    def __init__(self, root: str | os.PathLike | None):
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, url: str) -> Path | None:
        if not self.root:
            return None
        h = hashlib.sha256(url.encode()).hexdigest()
        return self.root / h[:2] / h

    def get(self, url: str) -> bytes | None:
        """Return cached bytes, b"" for a cached miss, or None if not cached."""
        p = self._path(url)
        if p is None:
            return None
        if p.exists():
            return p.read_bytes()
        miss = p.with_suffix(".missing")
        if miss.exists():
            return b""
        return None

    def put(self, url: str, data: bytes | None) -> None:
        p = self._path(url)
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        if data is None:
            p.with_suffix(".missing").touch()
            return
        tmp = p.with_suffix(".tmp%d" % threading.get_ident())
        tmp.write_bytes(data)
        os.replace(tmp, p)


def read_bytes(url: str, cache: ChunkCache | None = None) -> bytes | None:
    """Read a whole object; None means it does not exist."""
    if cache is not None:
        hit = cache.get(url)
        if hit is not None:
            return hit if hit != b"" else None
    if is_remote(url):
        # requests may fail while consuming the body, after urllib3's
        # header/status retries have finished. Never cache a partial object.
        for attempt in range(3):
            try:
                r = _session().get(url, timeout=120)
                break
            except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout):
                if attempt == 2:
                    raise
                time.sleep(0.25 * 2**attempt)
        # Access denied is not an absent Zarr chunk. Silently filling a 403
        # with zeros can manufacture apparent masked-volume findings.
        if r.status_code == 404:
            data = None
        else:
            r.raise_for_status()
            data = r.content
    else:
        p = Path(url)
        data = p.read_bytes() if p.exists() else None
    if cache is not None:
        cache.put(url, data)
    return data


def read_json(url: str, cache: ChunkCache | None = None) -> dict:
    data = read_bytes(url, cache)
    if data is None:
        raise FileNotFoundError(url)
    return json.loads(data)


class ZarrArray:
    """A Zarr v2 array: shape/chunks/dtype from .zarray, dense box reads."""

    def __init__(self, url: str, cache: ChunkCache | None = None, workers: int = 16):
        self.url = s3_to_https(url).rstrip("/")
        self.cache = cache
        self.workers = workers
        meta = read_json(join(self.url, ".zarray"), cache)
        if meta.get("zarr_format") != 2:
            raise ValueError(f"{self.url}: only zarr_format 2 is supported")
        self.shape = tuple(int(v) for v in meta["shape"])
        self.chunks = tuple(int(v) for v in meta["chunks"])
        self.dtype = np.dtype(meta["dtype"])
        if len(self.shape) != 3 or len(self.chunks) != 3:
            raise ValueError(f"{self.url}: only 3D arrays are supported")
        self.fill_value = meta.get("fill_value") or 0
        self.order = meta.get("order", "C")
        self.sep = meta.get("dimension_separator", ".")
        comp = meta.get("compressor")
        self.codec = numcodecs.get_codec(comp) if comp else None
        if meta.get("filters"):
            raise ValueError(f"{self.url}: zarr filters are not supported")

    def _chunk_url(self, idx: tuple[int, ...]) -> str:
        return join(self.url, self.sep.join(str(i) for i in idx))

    def _decode(self, raw: bytes | None) -> np.ndarray:
        if raw is None:
            return np.full(self.chunks, self.fill_value, dtype=self.dtype)
        buf = self.codec.decode(raw) if self.codec else raw
        arr = np.frombuffer(buf, dtype=self.dtype)
        return arr.reshape(self.chunks, order=self.order)

    def read_box(self, lo: tuple[int, ...], hi: tuple[int, ...]) -> np.ndarray:
        """Dense read of [lo, hi) in array index order (z, y, x), clipped to shape."""
        lo = tuple(max(0, int(a)) for a in lo)
        hi = tuple(min(s, int(b)) for s, b in zip(self.shape, hi))
        out_shape = tuple(max(0, b - a) for a, b in zip(lo, hi))
        out = np.full(out_shape, self.fill_value, dtype=self.dtype)
        if min(out_shape) == 0:
            return out
        ranges = [range(a // c, (b - 1) // c + 1) for a, b, c in zip(lo, hi, self.chunks)]
        keys = [(i, j, k) for i in ranges[0] for j in ranges[1] for k in ranges[2]]

        def fetch(key):
            return key, self._decode(read_bytes(self._chunk_url(key), self.cache))

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for key, chunk in ex.map(fetch, keys):
                c0 = [k * c for k, c in zip(key, self.chunks)]
                src = tuple(
                    slice(max(a, s) - s, min(b, s + c) - s)
                    for a, b, s, c in zip(lo, hi, c0, self.chunks)
                )
                dst = tuple(
                    slice(max(a, s) - a, min(b, s + c) - a)
                    for a, b, s, c in zip(lo, hi, c0, self.chunks)
                )
                out[dst] = chunk[src]
        return out
