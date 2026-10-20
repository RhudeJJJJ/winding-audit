"""Resolve segments and Lasagna manifests from the open-data catalog (metadata.json)."""

from __future__ import annotations

import gzip
import json
import re

from .zarr_io import ChunkCache, read_bytes

BUCKET_HTTPS = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"


def load_catalog(cache: ChunkCache | None = None) -> dict:
    raw = read_bytes(f"{BUCKET_HTTPS}/metadata.json", cache)
    if raw is None:
        raise FileNotFoundError("metadata.json")
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def list_prefix(prefix: str) -> list[str]:
    """Keys directly under an S3 prefix (no caching: listings change)."""
    import requests

    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix.rstrip("/") + "/", "delimiter": "/"}
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET_HTTPS + "/", params=params, timeout=60)
        r.raise_for_status()
        keys += re.findall(r"<Key>([^<]*)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", r.text)
        if not m:
            return keys
        token = m.group(1)


def lasagna_manifests(catalog: dict, sample_id: str, volume_id: str) -> list[str]:
    """HTTPS URLs of Lasagna (density + normal) manifests for one volume."""
    out = []
    vol = catalog["samples"][sample_id]["volumes"][volume_id]
    for e in vol.get("data") or []:
        if e["type"] != "lasagna":
            continue
        path = e["origins"][0]["path"]
        if "/lasagna/" not in path:
            continue  # fibre predictions share the type but not the channels
        for key in list_prefix(path):
            if key.endswith(".lasagna.json"):
                out.append(f"{BUCKET_HTTPS}/{key}")
    return out


def segment_surface_on_volume(catalog: dict, sample_id: str, segment_id: str, volume_id: str) -> str | None:
    seg = catalog["samples"][sample_id]["segments"][segment_id]
    for e in seg["data"]:
        if e["type"] == "tifxyz-transformed":
            p = e["origins"][0]["path"]
            if f"-on-{volume_id}-" in p:
                return f"{BUCKET_HTTPS}/{p}"
    return None
