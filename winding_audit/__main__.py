"""winding-audit: check traced surfaces and per-winding segment sets against the
published Lasagna winding fields and the CT of the Vesuvius Challenge open data.

  audit      per-surface maps: tilt to the predicted sheets, net windings crossed
             within short windows, predicted phase; step sites and field/surface
             disagreement zones; optional CT sections through the strongest sites
  chain      do consecutive per-winding segments step out one winding at a time?
  placement  does a surface lie on papyrus in the volume it is attributed to?
  calibrate  field winding count between two traced windings (or turns)
"""

from __future__ import annotations

import argparse
from .json_io import dumps
import sys
from pathlib import Path

import numpy as np

from .audit import AuditParams, audit_surface
from .calibrate import integrate_segment, normal_hits, self_hits
from .lasagna import LasagnaFields
from .report import write_outputs
from .tifxyz import grid_normals, load_tifxyz
from .zarr_io import ChunkCache


def _ct_sections(res, ct_url: str, level: int, out: Path, top: int, half_len: int, depth: float, cache=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .ct import CTVolume, straightened_section

    ct = CTVolume(ct_url, level=level, cache=cache)
    s = res.surface
    N = grid_normals(s.points, s.valid)
    paths = []
    for site in res.sites[:top]:
        r, c = site["peak_rc"]
        if site["direction"] == "rows":
            a, b = max(0, c - half_len), min(s.shape[1], c + half_len + 1)
            P, M, lab = s.points[r, a:b], N[r, a:b], f"row {r}, columns {a}-{b - 1}"
        else:
            a, b = max(0, r - half_len), min(s.shape[0], r + half_len + 1)
            P, M, lab = s.points[a:b, c], N[a:b, c], f"column {c}, rows {a}-{b - 1}"
        img, frac, ts = straightened_section(ct, P, M, depth, depth_step=float(1 << level), sub=4)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(img, cmap="gray", aspect="auto", extent=[a + frac[0], a + frac[-1], ts[-1], ts[0]],
                  vmin=np.nanpercentile(img, 1), vmax=np.nanpercentile(img, 99.5))
        ax.axhline(0, color="red", lw=0.7, alpha=0.7)
        ax.axvline(c if site["direction"] == "rows" else r, color="yellow", lw=0.7, alpha=0.7)
        ax.set_title(f"{site['id']}: CT straightened along {lab} (surface at 0)")
        ax.set_xlabel("grid index along the line")
        ax.set_ylabel("offset along surface normal (voxels)")
        plt.tight_layout()
        p = out / f"ct_{site['id']}.png"
        plt.savefig(p, dpi=80)
        plt.close(fig)
        site["ct_section"] = p.name
        paths.append(p)
    return paths



def cmd_audit(args):
    cache = ChunkCache(args.cache) if args.cache else None
    surf = load_tifxyz(args.surface, cache)
    fields = LasagnaFields(args.lasagna, cache)
    params = AuditParams(step_lo=args.step_lo, step_hi=args.step_hi,
                         plateau_tol=args.plateau_tol, implausible=args.implausible,
                         min_site_vertices=args.min_site_vertices)
    res = audit_surface(surf, fields, params, with_cos=not args.no_cos)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.ct:
        _ct_sections(res, args.ct, args.ct_level, out, args.ct_top, args.ct_half_len, args.ct_depth, cache)
    meta = {"surface": args.surface, "lasagna": args.lasagna, "ct": args.ct, "title": args.title or Path(args.surface.rstrip("/")).name}
    summary = write_outputs(res, out, meta)
    print(dumps({k: summary[k] for k in ("shape", "valid_vertices", "step_sites", "disagreement_zones",
                                               "step_vertex_fraction", "disagreement_vertex_fraction")}, indent=1))


def cmd_calibrate(args):
    cache = ChunkCache(args.cache) if args.cache else None
    fields = LasagnaFields(args.lasagna, cache)
    a = load_tifxyz(args.surface_a, cache)
    if args.surface_b:
        b = load_tifxyz(args.surface_b, cache)
        pick, p, m, th = normal_hits(a, b, args.samples, args.max_dist, seed=args.seed)
        side = np.sign(np.median(th)) if len(th) else 0
        keep = np.sign(th) == side
        p, m, th = p[keep], m[keep], th[keep]
    else:
        pick, p, m, th = self_hits(a, args.samples, args.max_dist, args.min_grid_sep, seed=args.seed)
    w, raw, ok = integrate_segment(fields, p, p + th[:, None] * m)
    w, d = w[ok], np.abs(th[ok])
    if not len(w):
        out = {"pairs": 0, "verdict": "insufficient_data", "expected_windings": args.expected}
        print(dumps(out, indent=1))
        if args.out:
            Path(args.out).write_text(dumps(out, indent=1))
        return
    bins = [(0, 40), (40, 80), (80, 120), (120, 160), (160, 1e9)]
    out = {
        "pairs": int(ok.sum()),
        "expected_windings": args.expected,
        "median_windings": float(np.median(w)),
        "quartiles": [float(np.percentile(w, q)) for q in (25, 50, 75)],
        "median_distance_vox": float(np.median(d)),
        "by_distance": {f"{lo}-{hi if hi < 1e9 else 'inf'}": {"n": int(((d >= lo) & (d < hi)).sum()),
                                                                "median": float(np.median(w[(d >= lo) & (d < hi)])) if ((d >= lo) & (d < hi)).sum() else None}
                        for lo, hi in bins},
    }
    print(dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(dumps(out, indent=1))


def cmd_chain(args):
    from .chain import chain_verdicts, pair_check

    cache = ChunkCache(args.cache) if args.cache else None
    fields = LasagnaFields(args.lasagna, cache)
    surfaces = list(args.surfaces)
    pairs = {}
    prev = load_tifxyz(surfaces[0], cache)
    for i in range(1, len(surfaces)):
        cur = load_tifxyz(surfaces[i], cache)
        r = pair_check(prev, cur, fields, samples=args.samples, seed=i, one_winding=args.one_winding)
        r["a"], r["b"] = surfaces[i - 1], surfaces[i]
        pairs[f"{i - 1}->{i}"] = r
        print(f"{i - 1}->{i}: windings {r.get('windings_median', float('nan')):.3f} "
              f"distance {r.get('distance_median_vox', float('nan')):.0f} coincide {r['coincide_fraction']:.3f} n {r['n']}",
              flush=True)
        prev = cur
    spacing = chain_verdicts(pairs, args.one_winding, args.spacing)
    print(f"typical spacing {spacing:.1f} voxels" if spacing else "typical spacing unknown")
    for k, r in pairs.items():
        print(f"{k}: {r['verdict']:9s} windings {r.get('windings_median', float('nan')):.3f} "
              f"spacing ratio {r.get('spacing_ratio', float('nan')):.2f} coincide {r['coincide_fraction']:.3f}")
    if args.out:
        Path(args.out).write_text(dumps({"spacing_vox": spacing, "pairs": pairs}, indent=1))


def cmd_placement(args):
    from .ct import CTVolume
    from .placement import placement_check
    from .zarr_io import join, read_json, s3_to_https

    cache = ChunkCache(args.cache) if args.cache else None
    shape = read_json(join(s3_to_https(args.volume).rstrip("/"), "0", ".zarray"), cache)["shape"]
    fields = LasagnaFields(args.lasagna, cache) if args.lasagna else None
    rows = []
    for s in args.surfaces:
        # a fresh reader per surface keeps at most one surface's CT chunks in memory
        ct = CTVolume(args.volume, level=args.level, cache=cache)
        r = placement_check(load_tifxyz(s, cache), ct, shape, samples=args.samples, fields=fields,
                            peak_tolerance=args.peak_tolerance)
        r["surface"] = s
        r["level"] = args.level
        rows.append(r)
        if r["samples"] == 0:
            print(f"insufficient_data    {s}", flush=True)
            continue
        print(f"{r['verdict']:20s} CT median {r['ct_at_surface_median']:6.1f} outside {r['outside_fraction']:.2f} "
              f"empty {r['empty_fraction']:.2f} peak/edge {r['profile_peak_to_edge']:.2f} "
              f"tilt {r['field_tilt_median_deg']:.1f}  {s}", flush=True)
    if args.out:
        Path(args.out).write_text(dumps(rows, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="winding-audit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="audit a tifxyz surface against a Lasagna field")
    a.add_argument("surface", help="tifxyz directory (local path, https:// or s3:// URL)")
    a.add_argument("--lasagna", required=True, help="*.lasagna.json manifest of the same volume")
    a.add_argument("--out", required=True)
    a.add_argument("--cache", default=".winding_audit_cache", help="chunk cache directory ('' to disable)")
    a.add_argument("--ct", help="OME-Zarr CT volume root of the same volume, for straightened sections")
    a.add_argument("--ct-level", type=int, default=0)
    a.add_argument("--ct-top", type=int, default=12)
    a.add_argument("--ct-half-len", type=int, default=30)
    a.add_argument("--ct-depth", type=float, default=120.0)
    a.add_argument("--title")
    a.add_argument("--no-cos", action="store_true")
    a.add_argument("--step-lo", type=float, default=0.6)
    a.add_argument("--step-hi", type=float, default=1.8)
    a.add_argument("--plateau-tol", type=float, default=0.35)
    a.add_argument("--implausible", type=float, default=2.0)
    a.add_argument("--min-site-vertices", type=int, default=12)
    a.set_defaults(func=cmd_audit)
    c = sub.add_parser("calibrate", help="field winding count between traced windings")
    c.add_argument("surface_a")
    c.add_argument("surface_b", nargs="?", help="neighbouring winding; omit for a multi-turn surface")
    c.add_argument("--lasagna", required=True)
    c.add_argument("--expected", type=float, default=1.0)
    c.add_argument("--samples", type=int, default=3000)
    c.add_argument("--max-dist", type=float, default=200.0)
    c.add_argument("--min-grid-sep", type=int, default=150)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--cache", default=".winding_audit_cache")
    c.add_argument("--out")
    c.set_defaults(func=cmd_calibrate)
    h = sub.add_parser("chain", help="check that per-winding segments step out one winding at a time")
    h.add_argument("surfaces", nargs="+", help="tifxyz surfaces in winding order (inner to outer or reverse)")
    h.add_argument("--lasagna", required=True)
    h.add_argument("--samples", type=int, default=2000)
    h.add_argument("--one-winding", type=float, default=1.0,
                   help="field count of one winding used for the gap/duplicate verdicts")
    h.add_argument("--spacing", type=float, help="typical sheet spacing in voxels (default: median over the chain)")
    h.add_argument("--cache", default=".winding_audit_cache")
    h.add_argument("--out")
    h.set_defaults(func=cmd_chain)
    pl = sub.add_parser("placement", help="check that surfaces lie on papyrus in their volume")
    pl.add_argument("surfaces", nargs="+")
    pl.add_argument("--volume", required=True, help="OME-Zarr CT volume root the surfaces are attributed to")
    pl.add_argument("--level", type=int, default=0)
    pl.add_argument("--lasagna", help="Lasagna manifest of the volume: also report the tilt to the predicted sheets")
    pl.add_argument("--samples", type=int, default=1500)
    pl.add_argument("--peak-tolerance", type=float, default=3.0,
                    help="maximum normal offset of the CT profile peak, in base voxels")
    pl.add_argument("--cache", default=".winding_audit_cache")
    pl.add_argument("--out")
    pl.set_defaults(func=cmd_placement)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
