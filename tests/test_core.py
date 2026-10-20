"""Unit tests of the audit arithmetic on a synthetic layered field.

The field is written to disk as real Zarr v2 / blosc arrays with a Lasagna
manifest, so every test runs the same reader, decoder and sampler as an
audit of scroll data. Sheets are the planes x = const with a spacing of
``SPACING`` voxels, i.e. density 1 / SPACING and normal (1, 0, 0).
"""

from __future__ import annotations

import json
from pathlib import Path

import numcodecs
import numpy as np
import pytest
import tifffile

from winding_audit.audit import audit_surface
from winding_audit.edges import edge_fields
from winding_audit.lasagna import LasagnaFields
from winding_audit.report import write_point_collection
from winding_audit.steps import _window_sums
from winding_audit.tifxyz import load_tifxyz

SPACING = 40.0
SHAPE = (64, 64, 64)  # z, y, x at 16 base voxels per sample -> 1024^3 base voxels


def _write_zarr(root: Path, arr: np.ndarray, chunks=(16, 16, 16)):
    root.mkdir(parents=True, exist_ok=True)
    codec = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=1)
    meta = {"zarr_format": 2, "shape": list(arr.shape), "chunks": list(chunks), "dtype": "|u1",
            "fill_value": 0, "order": "C", "filters": None, "dimension_separator": "/",
            "compressor": codec.get_config()}
    (root / ".zarray").write_text(json.dumps(meta))
    for i in range(0, arr.shape[0], chunks[0]):
        for j in range(0, arr.shape[1], chunks[1]):
            for k in range(0, arr.shape[2], chunks[2]):
                block = np.zeros(chunks, np.uint8)
                sub = arr[i:i + chunks[0], j:j + chunks[1], k:k + chunks[2]]
                block[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
                p = root / str(i // chunks[0]) / str(j // chunks[1])
                p.mkdir(parents=True, exist_ok=True)
                (p / str(k // chunks[2])).write_bytes(codec.encode(block))


@pytest.fixture(scope="module")
def field_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("lasagna")
    # density byte = density * encode_scale / factor = (1/SPACING) * 1000 / 0.25
    g = np.full(SHAPE, int(round(1000.0 / 0.25 / SPACING)), np.uint8)
    nx = np.full(SHAPE, 128 + 127, np.uint8)  # nx = 1
    ny = np.full(SHAPE, 128, np.uint8)  # ny = 0
    x = (np.arange(SHAPE[2]) * 16.0)[None, None, :]
    cos = np.broadcast_to((255 * (0.5 + 0.5 * np.cos(2 * np.pi * x / SPACING))).astype(np.uint8), SHAPE)
    for name, arr in (("grad_mag", g), ("nx", nx), ("ny", ny)):
        _write_zarr(d / f"{name}.zarr" / "4", arr)
    _write_zarr(d / "cos.zarr" / "4", np.ascontiguousarray(cos))
    manifest = {"version": 2, "source_to_base": 1.0, "grad_mag_encode_scale": 1000.0, "grad_mag_factor": 0.25,
                "groups": {n: {"zarr": f"{n}.zarr/4", "scaledown": 4, "channels": [n]}
                           for n in ("grad_mag", "nx", "ny", "cos")}}
    (d / "test.lasagna.json").write_text(json.dumps(manifest))
    return d


def _write_tifxyz(d: Path, pts: np.ndarray, mask: np.ndarray | None = None):
    d.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate("xyz"):
        tifffile.imwrite(d / f"{n}.tif", pts[..., i].astype(np.float32))
    if mask is not None:
        tifffile.imwrite(d / "mask.tif", mask.astype(np.uint8) * 255)
    (d / "meta.json").write_text(json.dumps({"format": "tifxyz", "scale": [0.05, 0.05]}))


def _plane(rows=40, cols=40, step=20.0, x0=300.0, slope=0.0):
    r, c = np.mgrid[0:rows, 0:cols].astype(float)
    y = 200.0 + c * step
    z = 200.0 + r * step
    x = x0 + slope * (c * step)
    return np.stack([x, y, z], axis=-1)


def test_parallel_surface_crosses_nothing(field_dir, tmp_path):
    _write_tifxyz(tmp_path / "s", _plane())
    f = LasagnaFields(str(field_dir / "test.lasagna.json"))
    eh, ev = edge_fields(load_tifxyz(str(tmp_path / "s")), f)
    assert eh.valid.all() and ev.valid.all()
    assert np.abs(eh.crossing).max() < 1e-6
    assert np.abs(ev.crossing).max() < 1e-6


def test_tilted_surface_crosses_known_windings(field_dir, tmp_path):
    slope = 0.1  # x rises 2 voxels per 20-voxel column edge -> 0.05 windings per edge
    _write_tifxyz(tmp_path / "s", _plane(slope=slope))
    f = LasagnaFields(str(field_dir / "test.lasagna.json"))
    eh, _ = edge_fields(load_tifxyz(str(tmp_path / "s")), f)
    expected = slope * 20.0 / SPACING
    # density is quantised to whole bytes: 100 -> exactly 1/40
    assert np.allclose(np.abs(eh.crossing[eh.valid]), expected, rtol=1e-3)


def test_step_is_found_and_counted(field_dir, tmp_path):
    pts = _plane(rows=60, cols=80, step=10.0)  # stays inside the 1024-voxel field
    pts[:, 40:, 0] += SPACING  # right half moved onto the next sheet: one winding
    _write_tifxyz(tmp_path / "s", pts)
    f = LasagnaFields(str(field_dir / "test.lasagna.json"))
    res = audit_surface(load_tifxyz(str(tmp_path / "s")), f)
    s4 = res.steps.step[1]
    # every row sees exactly one winding across column 40 at every scale
    assert np.allclose(np.abs(s4[10:50, 38:42]), 1.0, atol=1e-6)
    assert np.nanmax(np.abs(s4[:, :30])) < 1e-6
    assert len(res.sites) == 1 and res.sites[0]["direction"] == "rows"
    assert abs(abs(res.sites[0]["median_step_small"]) - 1.0) < 1e-6


def test_mask_and_sentinels_invalidate(field_dir, tmp_path):
    pts = _plane()
    pts[5, 5] = -1
    mask = np.ones(pts.shape[:2], bool)
    mask[10, 10] = False
    _write_tifxyz(tmp_path / "s", pts, mask)
    s = load_tifxyz(str(tmp_path / "s"))
    assert not s.valid[5, 5] and not s.valid[10, 10] and s.valid.sum() == pts.shape[0] * pts.shape[1] - 2


def test_window_sums_match_brute_force():
    rng = np.random.default_rng(0)
    c = rng.normal(size=(3, 30))
    ok = rng.random((3, 30)) > 0.1
    h = 4
    got = _window_sums(c, ok, h)
    for r in range(3):
        for k in range(31):
            lo, hi = k - h, k + h
            if lo < 0 or hi > 30 or not ok[r, lo:hi].all():
                assert np.isnan(got[r, k])
            else:
                assert np.isclose(got[r, k], c[r, lo:hi].sum())


def test_point_collection_schema(tmp_path):
    p = tmp_path / "pc.json"
    write_point_collection(p, {"a": [{"peak_xyz": [1.0, 2.0, 3.0]}]}, {"a": (1, 0, 0)})
    j = json.loads(p.read_text())
    assert j["vc_pointcollections_json_version"] == "1"
    col = next(iter(j["collections"].values()))
    assert col["name"] == "a" and col["metadata"]["winding_is_absolute"] is False
    assert next(iter(col["points"].values()))["p"] == [1.0, 2.0, 3.0]


def test_chain_verdicts_need_field_and_geometry():
    from winding_audit.chain import chain_verdicts

    base = {"coincide_fraction": 0.05, "n": 100, "windings_median": 1.0, "distance_median_vox": 60.0}
    pairs = {f"p{i}": dict(base) for i in range(6)}
    pairs["dup"] = dict(base, windings_median=0.2, distance_median_vox=15.0)
    pairs["same"] = dict(base, coincide_fraction=0.8, windings_median=0.0, distance_median_vox=0.0)
    pairs["gap"] = dict(base, windings_median=2.1, distance_median_vox=120.0)
    pairs["field_low"] = dict(base, windings_median=0.4, distance_median_vox=62.0)
    pairs["tight"] = dict(base, distance_median_vox=20.0)
    assert chain_verdicts(pairs) == 60.0
    got = {k: r["verdict"] for k, r in pairs.items() if not k.startswith("p")}
    assert got == {"dup": "duplicate", "same": "duplicate", "gap": "gap", "field_low": "check", "tight": "check"}
    assert all(pairs[f"p{i}"]["verdict"] == "ok" for i in range(6))


def test_chain_on_synthetic_field(field_dir, tmp_path):
    from winding_audit.chain import chain_verdicts, pair_check

    f = LasagnaFields(str(field_dir / "test.lasagna.json"))
    surfs = []
    for i, x0 in enumerate((300.0, 340.0, 380.0, 420.0, 500.0)):  # 500: the sheet at 460 is missing
        _write_tifxyz(tmp_path / f"s{i}", _plane(rows=20, cols=20, x0=x0))
        surfs.append(load_tifxyz(str(tmp_path / f"s{i}")))
    pairs = {i: pair_check(surfs[i], surfs[i + 1], f, samples=200, seed=i) for i in range(4)}
    assert [round(pairs[i]["windings_median"], 3) for i in range(4)] == [1.0, 1.0, 1.0, 2.0]
    assert pairs[3]["verdict"] == "check"  # field alone: two windings, no geometry yet
    assert chain_verdicts(pairs) == SPACING
    assert [pairs[i]["verdict"] for i in range(4)] == ["ok", "ok", "ok", "gap"]


def test_placement_classes(field_dir, tmp_path):
    from winding_audit.ct import CTVolume
    from winding_audit.placement import placement_check

    n = 128
    x = np.arange(n)
    row = np.where(x < 110, 30, 0).astype(np.uint8)  # x >= 110 masked away
    for xs in (20, 60, 100):
        row[np.abs(x - xs) <= 2] = 200
    _write_zarr(tmp_path / "ct.zarr" / "0", np.ascontiguousarray(np.broadcast_to(row, (n, n, n))), chunks=(32, 32, 32))
    ct = CTVolume(str(tmp_path / "ct.zarr"), level=0)
    r, c = np.mgrid[0:9, 0:9].astype(float)
    f = LasagnaFields(str(field_dir / "test.lasagna.json"))
    got = {}
    for x0 in (60.0, 40.0, 118.0, 300.0):
        _write_tifxyz(tmp_path / f"p{x0:.0f}", np.stack([np.full_like(r, x0), 24 + 10 * c, 24 + 10 * r], axis=-1))
        got[x0] = placement_check(load_tifxyz(str(tmp_path / f"p{x0:.0f}")), ct, (n, n, n), samples=50, fields=f)
    assert {k: v["verdict"] for k, v in got.items()} == {
        60.0: "on_papyrus", 40.0: "no_layer_at_surface", 118.0: "on_empty_voxels", 300.0: "outside_volume"}
    assert got[60.0]["field_tilt_median_deg"] < 0.5  # the planes are parallel to the field's sheets
    # a plane tilted by 30 degrees about the z axis
    t = np.radians(30.0)
    _write_tifxyz(tmp_path / "tilted", np.stack([60 + 10 * c * np.sin(t), 24 + 10 * c * np.cos(t), 24 + 10 * r], axis=-1))
    tilted = placement_check(load_tifxyz(str(tmp_path / "tilted")), ct, (n, n, n), samples=50, fields=f)
    assert abs(tilted["field_tilt_median_deg"] - 30.0) < 0.5
