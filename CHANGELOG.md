# Changes

## 0.1.1 — 20 October 2026

* Stop on denied remote access instead of converting HTTP 403 into a fill-valued chunk. Retry bounded transient HTTP and interrupted-body failures without caching partial objects.
* Stream CT requests by chunk in bounded batches of eight; do not retain all CT chunks of a curved surface in memory. Interpolation arithmetic is tested across chunk boundaries.
* Require a CT profile peak near the sampled surface, report an off-centre `layer_offset`, expose CT and orientation screens separately, and retain the sampled normal profile and denominators.
* Add `check_orientation` for CT-pass cases with median model-normal deviation above 20°. This is a screening flag, not ground truth.
* Compute geometric chain distance before field-validity filtering, report path counts and directional coincidence denominators, and withhold field verdicts with fewer than 20 valid paths.
* Emit portable JSON null values for missing numeric observations rather than NaN/Infinity.
* Handle no-hit/empty inputs and reject mismatched masks explicitly.
* Declare `imagecodecs` for compressed real TIFF input.
* Remove shell evaluation of catalog-derived placement commands. Checkpoint long chain runs; avoid redundant TIFF loads.
* Add offline result consistency checks, descriptive pair-level field summaries, metadata snapshot, regression tests and submission evidence boundaries.
* Replace global absence/absolute identity claims with local, inspectable overlap and coverage observations. Distinguish vertices from area and screen passes from labelled negatives.

Original results are retained directly under `results/`; independent review runs are stored under `results/review/`. Original JSON verdicts do not retroactively become revised-code outputs.
