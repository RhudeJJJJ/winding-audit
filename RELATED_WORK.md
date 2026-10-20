# Related work and intended contribution

This is a source/documentation comparison, **not a comparative performance benchmark**. Checked against the upstream villa tree `d99d1e978fcc4bf6fb4681f20fe6348450898453` on 20 October 2026.

| Existing work | Overlap | Scope claimed here |
|---|---|---|
| `vc_tifxyz_winding` | Winding-number estimation on mesh geometry | Comparison of consecutive labelled meshes, model field counts and CT support; no claim of a new general winding estimator |
| `vc_tifxyz_selfcross` / windcheck | Mesh intersections and geometry QA | Cross-mesh label/coverage screening is a different input question, but the tools are complementary |
| `vc_lasagna_line_probe` | Uses the same Lasagna observations | Batch reports for existing per-winding mesh sets, with explicit source-volume support checks |
| `find_inconsistent_windings.py` | Existing consistency diagnostics | That tool propagates absolute/relative point-collection annotations through spiral-space patch links; this tool reads sampled mesh geometry and CT before fitting |
| Herculaneum Scroll Tools / `audit_ct_support` | CT support and masking diagnostics | Applying support checks to transformed traced vertices; **CT support auditing itself is not novel** |
| TIFXYZ Doctor / spiralcheck / tifxyz-repair | Mesh/metadata QC and controlled evaluation | No claim of first mesh QA, first synthetic switch control or superiority; the concrete per-winding cases and combined report are the proposed deliverable |

Primary sources:

* [Official community catalog](https://github.com/ScrollPrize/villa/blob/main/scrollprize.org/docs/20_community_projects.md).
* [Upstream app sources](https://github.com/ScrollPrize/villa/tree/main/volume-cartographer/apps/src).
* [Annotation consistency tool](https://github.com/ScrollPrize/villa/blob/main/spiral-fitting/find_inconsistent_windings.py).
* [Compact normal sampler](https://github.com/ScrollPrize/villa/blob/main/volume-cartographer/core/src/lasagna/ChannelSampler.cpp).

The novelty and actionable value of these exact data findings still require maintainer assessment. No claim is made that the cases were previously unknown to the annotation team. No matched accuracy/runtime comparison or downstream fitting improvement has been demonstrated.

The public [villa issue #1254](https://github.com/ScrollPrize/villa/issues/1254) also discusses predictions outside masked CT support and credits earlier mitigation work. That problem class predates winding-audit. The PHerc1667 result here concerns transformed mesh vertices and their chosen scan, so it should be presented as a specific measured support mismatch, not the discovery of CT-support QA.
