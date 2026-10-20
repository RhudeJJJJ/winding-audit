"""Figures for the README from the files in results/ (each figure is skipped if its input is missing)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

R = Path("results")
F = R / "figures"
F.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#184f95", "#6da7ec", "#f0efec", "#ef8a7a", "#b52f2f"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "legend.labelcolor": INK2,
})


def save(fig, name):
    fig.savefig(F / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", F / name)


def fig_chain():
    files = [(R / "chain_PHerc0139.json", "PHerc. 139"), (R / "chain_PHerc1667.json", "PHerc. 1667")]
    files = [(p, t) for p, t in files if p.exists()]
    if not files:
        return
    fig, axes = plt.subplots(len(files), 1, figsize=(11, 3.8 * len(files)), squeeze=False,
                             gridspec_kw={"hspace": 0.62})
    status = {"duplicate": (CRITICAL, "duplicate"), "gap": (SERIOUS, "gap"), "overlap": (WARNING, "overlap"),
              "check": (MUTED, "check CT")}
    for ax, (path, name) in zip(axes[:, 0], files):
        d = json.loads(path.read_text())
        pairs = d["pairs"]
        keys = list(pairs)
        x = np.arange(len(keys))
        med = np.array([pairs[k].get("windings_median", np.nan) for k in keys])
        lo = np.array([pairs[k].get("windings_q25", np.nan) for k in keys])
        hi = np.array([pairs[k].get("windings_q75", np.nan) for k in keys])
        geo = np.array([pairs[k].get("spacing_ratio", np.nan) for k in keys])
        ax.axhline(1.0, color=INK2, lw=1.0, zorder=1)
        for i, k in enumerate(keys):
            v = pairs[k]["verdict"]
            if v in status:
                col, lab = status[v]
                ax.axvspan(i - 0.45, i + 0.45, color=col, alpha=0.16, lw=0, zorder=0)
                ax.text(i, 2.62, lab, ha="center", va="top", fontsize=8, color=INK, rotation=90 if len(lab) > 5 else 0)
        # the two measures sit side by side so that neither hides the other
        ax.vlines(x - 0.14, lo, hi, color=BLUE, lw=2, alpha=0.35, zorder=2)
        ax.scatter(x - 0.14, med, s=40, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4,
                   label="field: windings counted along the normals (median, whisker: quartiles)")
        ax.scatter(x + 0.14, geo, s=30, marker="D", color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=f"geometry: normal distance / typical spacing ({d.get('spacing_vox', float('nan')):.0f} voxels)")
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("->", "→") for k in keys], rotation=90, fontsize=7)
        ax.set_xlim(-0.6, len(keys) - 0.4)
        ax.set_ylabel("step to the next segment\n(windings)")
        ax.set_ylim(-0.1, 2.7)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{name}: {len(keys)} pairs of consecutive per-winding segments", loc="left")
        ax.legend(loc="upper left", fontsize=8, ncol=1, bbox_to_anchor=(0.0, 0.93))
    save(fig, "chain.png")


def fig_calibration():
    rows = []
    p = R / "chain_PHerc0139_samples.json"
    if p.exists():
        d = json.loads(p.read_text())
        good = [v for v in d.values() if v["verdict"] == "ok" and "distance" in v]
        dist = np.concatenate([np.array(v["distance"]) for v in good])
        wnd = np.concatenate([np.array(v["windings"]) for v in good])
        rows.append((f"PHerc. 139: neighbouring per-winding segments ({len(good)} pairs judged ok)", dist, wnd, BLUE))
    p = R / "selfcalib_PHercParis4.json"
    if p.exists():
        d = json.loads(p.read_text())
        dist = np.concatenate([np.array(v["distance"]) for v in d.values()])
        wnd = np.concatenate([np.array(v["windings"]) for v in d.values()])
        rows.append(("PHerc. Paris. 4: adjacent turns of multi-turn traces", dist, wnd, ORANGE))
    if not rows:
        return
    edges = np.arange(20, 201, 20)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.axhline(1.0, color=INK2, lw=1.0)
    for j, (lab, dist, wnd, col) in enumerate(rows):
        cx, m, q1, q3 = [], [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (dist >= a) & (dist < b)
            if sel.sum() >= 30:
                cx.append((a + b) / 2 + (j - 0.5) * 3)
                m.append(np.median(wnd[sel]))
                q1.append(np.percentile(wnd[sel], 25))
                q3.append(np.percentile(wnd[sel], 75))
        cx, m = np.array(cx), np.array(m)
        ax.vlines(cx, q1, q3, color=col, lw=2, alpha=0.35)
        ax.plot(cx, m, color=col, lw=2, marker="o", ms=6, mec=SURFACE, mew=2, label=lab)
    ax.set_xlabel("distance between the two traced windings along the surface normal (voxels, 2.4 µm)")
    ax.set_ylabel("windings counted by the field\n(median, whisker: quartiles)")
    ax.set_ylim(0, 2.2)
    ax.set_title("Field integrals on selected traced-neighbour paths (descriptive)", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "calibration.png")


def _winding(surface_url):
    seg = surface_url.rstrip("/").split("/")[-3]
    return int(seg.split("-w")[1][:3])


def fig_placement():
    groups = []
    for sid in ("PHerc1447", "PHerc0800"):
        p = R / f"placement_{sid}.json"
        if p.exists():
            for r in json.loads(p.read_text()):
                date = r["surface"].rstrip("/").split("/")[-3][:8]
                label = {"PHerc1447": "PHerc. 1447\ntraced " + ("2 May 2025" if date == "20250502" else "Jul–Nov 2025"),
                         "PHerc0800": "PHerc. 800\ntraced Oct 2025"}[sid]
                groups.append((label, r))
    if not groups:
        return
    labels = list(dict.fromkeys(g for g, _ in groups))
    panels = [("on_material", "vertices on material\n(not empty, masked or outside)", (0, 1.05)),
              ("profile_peak_to_edge", "maximum in normal profile\n(peak / edge; not a truth label)", None),
              ("field_tilt_median_deg", "angle to the predicted sheets\n(median, degrees)", (0, 90))]
    if not any(np.isfinite(r.get("field_tilt_median_deg", np.nan)) for _, r in groups):
        panels = panels[:2]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.9))
    rng = np.random.default_rng(0)
    for ax, (key, ylab, lim) in zip(axes, panels):
        for i, lab in enumerate(labels):
            for r in (r for g, r in groups if g == lab):
                v = (1 - r["outside_fraction"]) * (1 - r["empty_fraction"]) if key == "on_material" else r[key]
                good = r["verdict"] == "on_papyrus"
                ax.scatter(i + rng.uniform(-0.13, 0.13), v, s=42, color=GOOD if good else CRITICAL,
                           edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        if lim:
            ax.set_ylim(*lim)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
    # above the grid, which would otherwise hide a line drawn at a grid position
    axes[1].axhline(1.04, color=INK2, lw=1.0, zorder=2)
    axes[1].text(0.5, 1.041, "threshold 1.04", fontsize=8, color=INK2, ha="center", va="bottom")
    h1 = axes[0].scatter([], [], s=42, color=GOOD, label="legacy intensity screen passes")
    h2 = axes[0].scatter([], [], s=42, color=CRITICAL, label="legacy intensity screen flags")
    axes[0].legend(handles=[h1, h2], loc="lower right", fontsize=8)
    fig.suptitle("Placement screens in the catalog-assigned 8.64 µm volumes (original measurements)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "placement.png")


def fig_duplicate():
    p = R / "duplicate_w045_w046_maps.npz"
    q = R / "duplicate_ct_section.npz"
    if not (p.exists() and q.exists()):
        return
    d = np.load(p)
    s = np.load(q)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.42})
    im = axes[0].imshow(np.clip(d["d45"], 0, 80), cmap=LinearSegmentedColormap.from_list("b", BLUE_RAMP), vmin=0, vmax=80)
    axes[0].set_title("w046 → nearest w045 vertex\n(light: close sampled vertices)", loc="left")
    axes[0].set_xlabel("w046 column / 4")
    axes[0].set_ylabel("w046 row / 4")
    axes[0].grid(False)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046)
    cb.set_label("voxels", color=INK2)
    img, frac, ts, x0 = s["img"], s["frac"], s["ts"], float(s["x0"])
    axes[1].imshow(img, cmap="gray", aspect="auto", extent=[x0 + frac[0], x0 + frac[-1], ts[-1], ts[0]],
                   vmin=np.nanpercentile(img, 1), vmax=np.nanpercentile(img, 99.5))
    axes[1].axhline(0, color="#ffffff", lw=1.0, alpha=0.8)
    axes[1].scatter(s["x46"], s["y46"], s=22, color=BLUE, edgecolor="#ffffff", linewidth=1, label="w046 crosses here")
    axes[1].scatter(s["x47"], s["y47"], s=22, color=ORANGE, edgecolor="#ffffff", linewidth=1, label="w047 crosses here")
    axes[1].set_title("CT along w045 normals (row 600)\nw045 at 0; intervening layer visible locally", loc="left")
    axes[1].set_xlabel("w045 column")
    axes[1].set_ylabel("offset along the w045 normal (voxels)")
    axes[1].grid(False)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8)
    save(fig, "duplicate_w045_w046.png")


def fig_planted():
    p = R / "planted_control_maps.npz"
    j = R / "planted_control.json"
    if not (p.exists() and j.exists()):
        return
    d = np.load(p)
    meta = json.loads(j.read_text())
    R0, R1, C0, C1 = meta["crop"]
    br, bc = meta["block_rows"], meta["block_cols"]
    names = [("original", "unmodified w040"), ("planted_ramp0", "block moved onto w041, abrupt"),
             ("planted_ramp6", "block moved onto w041, 6-vertex ramp")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (k, title) in zip(axes, names):
        im = ax.imshow(d[f"{k}_step4"], cmap=DIVERGING, vmin=-1.5, vmax=1.5)
        if k != "original":
            ax.add_patch(plt.Rectangle((bc[0] - C0, br[0] - R0), bc[1] - bc[0], br[1] - br[0], fill=False, ec=INK, lw=1))
        v = meta["variants"][k]["plateau_criterion"]
        ax.set_title(f"{title}\ngrid lines across the block edge flagged: {100 * v['line_sensitivity']:.0f}%",
                     loc="left", fontsize=9)
        ax.grid(False)
        ax.set_xlabel("column (crop)")
        ax.set_ylabel("row (crop)")
    cb = fig.colorbar(im, ax=axes, fraction=0.02)
    cb.set_label("net windings crossed within ±4 edges", color=INK2)
    v = meta["variants"]["original"]["plateau_criterion"]
    fig.text(0.01, -0.02, f"Away from the block, the same {100 * v['vertices_far_flagged']:.1f}% of the vertices are flagged in all three "
             "(false alarms of the plateau criterion on this surface). White: windows that contain block vertices without a crossing of w041 (dropped).",
             fontsize=8, color=INK2, ha="left")
    save(fig, "planted_control.png")


def fig_masked():
    p = R / "masked_1667_axial.npz"
    j = R / "masked_1667.json"
    q = R / "placement_PHerc1667.json"
    if not (p.exists() and j.exists() and q.exists()):
        return
    d = np.load(p)
    meta = json.loads(j.read_text())
    rows = sorted(json.loads(q.read_text()), key=lambda r: _winding(r["surface"]))
    xs = d["xs"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axes[0]
    w = np.array([_winding(r["surface"]) for r in rows])
    e = np.array([r["empty_fraction"] for r in rows])
    good = np.array([r["verdict"] == "on_papyrus" for r in rows])
    ax.axhline(0.5, color=INK2, lw=1.0, zorder=1)
    ax.text(w.min(), 0.515, "half the in-volume samples", fontsize=8, color=INK2, va="bottom")
    ax.scatter(w[good], e[good], s=42, color=GOOD, edgecolor=SURFACE, linewidth=2, zorder=3, label="mostly nonempty CT samples")
    ax.scatter(w[~good], e[~good], s=42, color=CRITICAL, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="mostly on masked voxels")
    ax.set_xlabel("winding of the per-winding segment")
    ax.set_ylabel("vertices on masked (zero) voxels")
    ax.set_ylim(-0.03, 1.0)
    ax.set_title("Empty fraction among in-volume samples", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    for ax, zc in zip(axes[1:], ("15000", "31000")):
        img = d[f"img_{zc}"]
        ax.imshow(img, cmap="gray", extent=[xs[0], xs[-1], xs[-1], xs[0]], vmin=0, vmax=200)
        for k, col in (("w040", ORANGE), ("w041", BLUE)):
            P = d[f"{k}_{zc}"]
            frac = meta["slices"][f"{k} at z={zc}"]["on_masked_voxels"]
            ax.scatter(P[:, 0], P[:, 1], s=2, color=col, label=f"{k}: {100 * frac:.0f}% on masked voxels", rasterized=True)
        ax.set_title(f"Axial slice z = {int(zc):,}", loc="left")
        ax.set_xlabel("x (voxels)")
        ax.set_ylabel("y (voxels)")
        ax.grid(False)
        ax.legend(loc="lower left", fontsize=8, markerscale=6, labelcolor="#ffffff")
    fig.suptitle("PHerc. 1667, masked volume 20251217075048: outer meshes extend beyond CT support",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "masked_1667.png")


if __name__ == "__main__":
    fig_chain()
    fig_calibration()
    fig_placement()
    fig_duplicate()
    fig_planted()
    fig_masked()
