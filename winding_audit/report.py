"""Write an audit's outputs: JSON report, maps, preview image, VC3D point collection."""

from __future__ import annotations

from .json_io import dumps
import time
import uuid
from pathlib import Path

import numpy as np
import tifffile

from .audit import AuditResult, summarize


def write_point_collection(path: Path, groups: dict[str, list], colors: dict[str, tuple]):
    """VC3D PointCollections JSON (vc_pointcollections_json_version 1)."""
    now = int(time.time())
    collections = {}
    pid = 1
    for cid, (name, sites) in enumerate(groups.items(), start=1):
        points = {}
        for s in sites:
            points[str(pid)] = {"p": s["peak_xyz"], "creation_time": now, "wind_a": None}
            pid += 1
        collections[str(cid)] = {
            "uuid": str(uuid.uuid4()),
            "name": name,
            "points": points,
            "metadata": {"winding_is_absolute": False},
            "color": list(colors.get(name, (1.0, 0.2, 0.2))),
        }
    path.write_text(dumps({"vc_pointcollections_json_version": "1", "collections": collections}, indent=1))


def write_outputs(res: AuditResult, out_dir: str | Path, meta: dict) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(res)
    summary.update(meta)
    summary["step_sites_list"] = res.sites
    summary["disagreement_zones_list"] = res.disagreements
    (out / "report.json").write_text(dumps(summary, indent=1))

    maps = {
        "tilt_deg": res.tilt_deg,
        "step_mask": res.step_mask.astype(np.uint8),
        "disagreement_mask": res.disagree_mask.astype(np.uint8),
    }
    for i, h in enumerate(res.steps.scales):
        maps[f"net_crossing_h{h}"] = res.steps.step[i]
    if res.cos is not None:
        maps["cos_phase"] = res.cos
    for name, arr in maps.items():
        tifffile.imwrite(out / f"{name}.tif", np.asarray(arr, dtype=np.float32 if arr.dtype.kind == "f" else arr.dtype))

    write_point_collection(
        out / "sites_pointcollection.json",
        {"winding-audit step sites": res.sites, "winding-audit field/surface disagreement": res.disagreements},
        {"winding-audit step sites": (1.0, 0.15, 0.15), "winding-audit field/surface disagreement": (0.2, 0.4, 1.0)},
    )
    _preview(res, out / "overview.png", meta.get("title", ""))
    return summary


def _preview(res: AuditResult, path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = res.surface
    n = 4 if res.cos is not None else 3
    aspect = s.shape[1] / max(1, s.shape[0])
    w = 5.2 * n
    fig, ax = plt.subplots(1, n, figsize=(w, max(3.5, 5.2 / max(aspect, 0.4))))
    im = ax[0].imshow(res.tilt_deg, cmap="magma", vmin=0, vmax=45)
    ax[0].set_title("surface tilt to predicted sheets (deg)")
    plt.colorbar(im, ax=ax[0], fraction=0.046)
    k = 1 if len(res.steps.scales) > 2 else 0
    im = ax[1].imshow(res.steps.step[k], cmap="coolwarm", vmin=-1.5, vmax=1.5)
    ax[1].set_title(f"net windings crossed within ±{res.steps.scales[k]} edges")
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    j = 2
    if res.cos is not None:
        im = ax[2].imshow(res.cos, cmap="viridis", vmin=0, vmax=1)
        ax[2].set_title("predicted winding phase (cos)")
        plt.colorbar(im, ax=ax[2], fraction=0.046)
        j = 3
    ov = np.zeros(s.shape + (3,))
    ov[~s.valid] = 0.85
    ov[res.disagree_mask] = (0.2, 0.4, 1.0)
    ov[res.step_mask] = (0.95, 0.15, 0.15)
    ax[j].imshow(ov)
    ax[j].set_title("red: step sites  blue: field/surface disagreement")
    for st in res.sites[:12]:
        ax[j].annotate(st["id"].split("-")[1], (st["peak_rc"][1], st["peak_rc"][0]), color="black", fontsize=8,
                       ha="center", va="center", bbox=dict(boxstyle="round,pad=0.1", fc="yellow", alpha=0.7))
    for a in ax:
        a.set_xlabel("column")
        a.set_ylabel("row")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=70)
    plt.close(fig)
