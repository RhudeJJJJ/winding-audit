"""Shared helpers for the reproduction scripts (catalog lookups, per-winding sets)."""

from __future__ import annotations

import re

from winding_audit.catalog import BUCKET_HTTPS, lasagna_manifests, load_catalog, segment_surface_on_volume
from winding_audit.zarr_io import ChunkCache


def open_catalog(cache_dir: str):
    cache = ChunkCache(cache_dir) if cache_dir else None
    return cache, load_catalog(cache)


def manifest(cat, sid: str, vid: str) -> str:
    urls = lasagna_manifests(cat, sid, vid)
    if len(urls) != 1:
        raise RuntimeError(f"{sid}/{vid}: expected one Lasagna manifest, found {urls}")
    return urls[0]


def volume_url(cat, sid: str, vid: str) -> str:
    v = cat["samples"][sid]["volumes"][vid]
    for e in v["data"]:
        if e["type"] == "ome-zarr":
            return f"{BUCKET_HTTPS}/{e['origins'][0]['path']}"
    raise RuntimeError(f"{sid}/{vid}: no OME-Zarr")


def per_winding(cat, sid: str, vid: str) -> dict[int, tuple[str, str]]:
    """Winding number -> (segment id, long id) for segments named w<NN>_..."""
    out = {}
    for segid, seg in sorted(cat["samples"][sid]["segments"].items()):
        m = re.match(r"^\d+-w0?(\d+)_", seg["long_id"])
        if m and "merged" not in seg["long_id"] and segment_surface_on_volume(cat, sid, segid, vid):
            out.setdefault(int(m.group(1)), (segid, seg["long_id"]))
    return out


def surface_url(cat, sid, segid, vid):
    return segment_surface_on_volume(cat, sid, segid, vid)
