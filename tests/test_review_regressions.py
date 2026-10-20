"""Adversarial cases found during the independent submission review."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import requests
from winding_audit.ct import CTVolume
from winding_audit.placement import placement_check
from winding_audit.tifxyz import Surface, load_tifxyz
from winding_audit.chain import verdict
from winding_audit.calibrate import integrate_segment
from winding_audit import zarr_io
from test_core import _write_zarr, _write_tifxyz


def test_access_denied_is_not_missing_or_cached(tmp_path, monkeypatch):
    response = requests.Response()
    response.status_code = 403
    response.url = 'https://example.invalid/chunk'
    monkeypatch.setattr(zarr_io, '_session', lambda: SimpleNamespace(get=lambda *a, **k: response))
    cache = zarr_io.ChunkCache(tmp_path)
    with pytest.raises(requests.HTTPError):
        zarr_io.read_bytes(response.url, cache)
    assert cache.get(response.url) is None


def test_404_keeps_zarr_fill_semantics(monkeypatch):
    response = requests.Response()
    response.status_code = 404
    monkeypatch.setattr(zarr_io, '_session', lambda: SimpleNamespace(get=lambda *a, **k: response))
    assert zarr_io.read_bytes('https://example.invalid/absent') is None


def test_off_center_layer_does_not_pass():
    class OffCenterCT:
        def sample(self, p):
            return 30.0 + 170.0 * np.exp(-((p[:, 0] - 68.0) / 1.5)**2)
    r, c = np.mgrid[:8, :8]
    points = np.stack([np.full_like(r, 60), 20+c*2, 20+r*2], axis=-1).astype(float)
    surf = Surface(points, np.ones((8, 8), bool), {}, 'synthetic')
    got = placement_check(surf, OffCenterCT(), (128, 128, 128))
    assert got['profile_peak_to_edge'] > 1.04  # the old rule accepted this
    assert got['profile_peak_offset_vox'] == 8
    assert got['verdict'] == 'layer_offset'


def test_empty_surface_is_insufficient_not_a_crash():
    surf = Surface(np.full((3, 3, 3), np.nan), np.zeros((3, 3), bool), {}, '')
    assert placement_check(surf, None, (100, 100, 100))['verdict'] == 'insufficient_data'
    w, raw, ok = integrate_segment(None, np.empty((0, 3)), np.empty((0, 3)))
    assert len(w) == len(raw) == len(ok) == 0


def test_one_valid_path_cannot_certify_a_chain_pair():
    r = dict(coincide_fraction=0.0, n=1, windings_median=1.0, distance_median_vox=40.0)
    assert verdict(r, spacing=40.0) == 'check'


def test_ct_sampler_is_exact_across_chunks_and_does_not_retain_them(tmp_path):
    z, y, x = np.mgrid[:12, :12, :12]
    a = (x + 2*y + 3*z).astype(np.uint8)
    _write_zarr(tmp_path/'ct'/'0', a, chunks=(4, 4, 4))
    ct = CTVolume(str(tmp_path/'ct'))
    p = np.array([[3.7, 4.2, 7.8], [0, 0, 0], [9.1, 8.6, 5.2], [-1, 0, 0], [np.nan, 0, 0]])
    got = ct.sample(p)
    assert np.allclose(got[:3], p[:3] @ [1, 2, 3])
    assert np.isnan(got[3:]).all()
    assert len(ct.channel.stack) == 1


def test_mismatched_mask_is_rejected(tmp_path):
    p = np.ones((4, 4, 3))*20
    _write_tifxyz(tmp_path/'s', p, np.ones((3, 3), bool))
    with pytest.raises(ValueError, match='mask shape'):
        load_tifxyz(str(tmp_path/'s'))


def test_real_world_lzw_tiff_is_supported(tmp_path):
    import tifffile
    r,c=np.mgrid[:4,:4]
    p=np.stack([20+c,20+r,np.full_like(r,20)],axis=-1).astype(np.float32)
    d=tmp_path/'compressed';d.mkdir()
    for j,n in enumerate('xyz'):
        tifffile.imwrite(d/f'{n}.tif',p[...,j],compression='lzw')
    got=load_tifxyz(str(d))
    assert got.valid.all() and np.array_equal(got.points,p)


def test_missing_numeric_report_values_are_portable_json():
    from winding_audit.json_io import dumps
    text=dumps({'outside':np.nan,'offset':np.inf,'samples':np.int64(0)})
    parsed=json.loads(text,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    assert parsed=={'outside':None,'offset':None,'samples':0}
