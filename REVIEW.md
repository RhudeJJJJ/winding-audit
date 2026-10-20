# Independent review of the supplied 0.1.0 package

20 October 2026. Original ZIP SHA256:
`ee9cdf3ad675f2354460bc3016c73fc1812134675cf5652280e8a744afa1dd89`.

This review was performed with OpenAI Codex after the initial Claude-assisted implementation. “Independently reproduced” below means a separate automated run from public source data in this session, not independent human anatomical adjudication or independence of the underlying CT-derived evidence.

## Evidence status

| Check | Result |
|---|---|
| Uploaded ZIP and the two initial text-file hashes | Match the supplied manifest |
| Original test suite | 9/9 passed |
| Revised engineering suite | 18/18 passed; includes real LZW TIFF decoding |
| Installed 0.1.1 wheel in a fresh virtual environment | 18/18 passed, from outside the source tree |
| w046→w045 geometry and CT strip | Reproduced exact original sampled overlap and median distance |
| Reverse overlap and w042/w041 geometry | Reproduced from raw TIFFs, with threshold curves and counts in `results/review/overlap_thresholds.json` |
| README's three-surface chain | Reproduced with revised code, 2,000 requested paths/pair and seeds 1/2. Verdicts: duplicate candidate and gap candidate; field median 2.129380, geometric distance 116 voxels for the latter |
| PHerc. 1667 selected masked-support slices | Reproduced, including support percentages and slice-based distance estimates |
| Planted control | Reproduced all original control statistics |
| PHerc. 1447 full 15-surface rerun | All 15 surfaces were rerun at the original 1,500-vertex density. Measurements agree within 1e-8; the 11 May candidates and 4 later screen passes remain. See `results/review/placement_PHerc1447.json` and `completion.json` |
| Full 51-pair original chains, Paris4 self-hit, PHerc800/814 exploratory audits | Code/result consistency inspected; not all independently rerun |

`results/offline_review.json` is an arithmetic check of supplied outputs. It must not be described as a new CT experiment. Stored path values were rounded in the original experiment, so arithmetic comparisons use a 0.00011 tolerance for medians.

## Defects corrected

1. **403 interpreted as absent data.** The original reader filled denied requests as if the chunk were missing. This can manufacture apparent empty-support regions. Only 404 now uses remote missing-chunk fill semantics; 403 fails. The original review reruns did not establish that this defect caused any supplied finding. Old cached misses cannot be retrospectively attributed to 403 versus 404 and should be cleared.
2. **Off-centre profile accepted.** The original `peak/edge` ratio used the maximum anywhere in ±12 voxels. An explicit counterexample has a peak eight voxels from the mesh but passes the old rule. Revised code reports `layer_offset` unless the mean-profile maximum is within the configurable three-voxel tolerance. This is a conservative heuristic and can still flag genuine curved or recto/verso-offset surfaces.
3. **CT intensity pass can coexist with a large orientation mismatch.** A 150-vertex rerun of one May mesh passes the intensity rule while the model-normal tilt is about 48°. The partial two-surface precheck is retained as `results/review/placement_150_vertex_precheck.json`; it predates the orientation-combination fix. With 1,500 vertices its original screen fails. Revised output keeps the intensity verdict and model angle separately and emits `check_orientation` above 20° for a CT-pass case. These observations restrict, rather than establish, anatomical truth.
4. **Geometry conditioned on field validity.** The original chain's distance median was restricted to paths with valid Lasagna predictions. Version 0.1.1 computes a geometric median over same-side hits before that filter and separately retains the field-valid median and sample counts. Fewer than 20 field-valid paths produce a review outcome rather than a field-based pass.
5. **Unbounded CT retention.** The original per-surface sampler accumulated every touched decoded CT chunk. It now fetches by sparse corner groups in batches of eight and retains none between calls. The real-data reruns confirm unchanged interpolation measurements for the tested support slices and full-sample placement cases.
6. **Missing compressed-TIFF dependency.** A later real PHerc1447 surface needs LZW decoding. `imagecodecs` was missing from the declared dependencies and is now required and tested.
7. **Failure/edge handling and reproducibility.** Bounded HTTP status/body retries, empty-input handling, shape validation, no shell evaluation of catalog URLs, long-chain checkpoints and explicit denominators were added. No resumed result is silently treated as a complete corpus run.

## Claims narrowed

* The 82.328% result is sampled directional **vertex proximity**, not 82.328% of physical surface area and not proof the entire mesh is winding 45. About 12.86% of the sampled vertices are at least 30 voxels away.
* A selected strip supports an intervening layer; global absence of winding 46 and absolute winding identities are not established.
* CT is the raw scan and Lasagna a model derived from it; these are not statistically independent references. Mesh tracing and model predictions may share provenance too.
* PHerc1667's zeros show lack of support in the selected masked export. Whether papyrus was masked away or a transform is wrong remains open. The reproduced distances are 2D distance-to-nonzero estimates on a 16-voxel grid, not exact anatomical distances.
* PHerc1447 outputs are screening inconsistencies, not proof that every vertex is off papyrus or that the source catalog is conclusively wrong. Date-like identifiers alone do not prove chronological provenance.
* The original assertion that all 27 README numbers had an automated checker was not supported by a bundled verifier. The added offline verifier checks the central stored-result counts and summaries; it does not certify every sentence, runtime estimate or CT interpretation.
* 47/51 chain screen passes are not 47 independently adjudicated negative controls. The six PHerc800 intensity controls originally include one borderline failure.
* The 56,695 field paths are correlated and selected using the method being described. Their pooled median is 1.0627, while the median of pair medians is 1.0483. Neither is an unbiased general recalibration. A count-versus-distance trend alone is not proof of a causal field defect.
* No reliable real-switch detection rate, benchmark superiority, automatic repair or downstream fitter/reading improvement is claimed.

## Submission assessment

The official Progress Prize category is a reasonable fit for a documented real-data diagnostic. The strongest submission is the concrete, inspectable label/support case set plus the reusable checks, not a claim of general sheet-switch detection. Existing projects already cover mesh QA, CT support, winding estimation and consistency checks; villa issue #1254 explicitly documents earlier CT-support mismatch observations and mitigations. The exact incremental contribution and whether the cases are new to maintainers remain open; see `RELATED_WORK.md`.

The 20,000-dollar monthly best-submission award is not a forecast for this project. No award probability or expected monetary value is defensible from this package alone. Community adoption, maintainer-confirmed corrections and an actual before/after downstream use would materially strengthen the case.

## What remains before sending

* Provide an actual accessible project/release URL, email and optional Discord name in the prepared form.
* Have Jonas review the evidence and wording without claiming personal inspection he has not performed.
* For a later upstream villa PR, follow its separate human-use/human-commentary and code-review requirements. The public prize form itself does not expose a separate mandatory “In my own words” field.
* Submit the form and retain its receipt. This review did not publish code, contact maintainers or submit anything.

Primary rule sources checked: [prizes](https://scrollprize.org/prizes), [current form](https://docs.google.com/forms/d/e/1FAIpQLSc4flEfgK2nyjoczz2_U_XrIGMlgrnSknWatLqrFPnbtKfZwg/viewform), [winding constraints](https://scrollprize.org/open_problems/winding_annotations), [data attribution](https://scrollprize.org/data), and [villa contributing policy](https://github.com/ScrollPrize/villa/blob/main/CONTRIBUTING.md).
