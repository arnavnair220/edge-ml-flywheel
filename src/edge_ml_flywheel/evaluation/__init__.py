"""Scoring a detector, and every statistic read off one scoring pass.

Four pure modules and the job that runs them, split by where they run -- the
arrangement `scoring` uses, and the split matters more here. **The four are data
in, values out: no `boto3`, no `torch`, no HTTP.** The serialization helpers take
a local path, so the test suite runs with no credentials and a promotion decision
is reproducible from a saved `.npz` rather than from an account.

- `coco` puts boxes and detections in the shapes `pycocotools` reads
- `match` scores once, keeping the per-image match arrays at every IoU threshold
  and area range in one pass
- `metrics` accumulates those arrays into AP over an arbitrary image list, which
  makes a resample a numpy operation rather than a re-scoring
- `bootstrap` resamples image IDs and reads the paired lower bound off `metrics`

The other three carry those into a cycle, and hold everything that is not pure:

- `job` is what one `CreateProcessingJob` request is, pure, with no client
- `launch` is the caller -- it resolves the request against the account
- `entrypoint` is the container: match, cache, compare, gate, write

The layer's shape follows from one number. The paired bootstrap needs 1,000
resamples over both models, and re-scoring per resample would be 2,000 full mAP
evaluations over 5,000 images for a single-seed cycle -- hours, and multiplied
again by every seed a run adds (design section 4.2). So scoring happens once and
every later statistic reads the cache.

**This is the half of the plane that holds ground truth.** `scoring` runs the
model and may read no label anywhere; the job here reads the eval cohort's boxes
and never sees a checkpoint. Two jobs and two roles, so the identity that could
contaminate the eval and the identity that could read the answer key are not the
same one.

Matching is borrowed from `pycocotools`; accumulation is reimplemented. The rule
is the same in both cases -- borrow what must agree with published numbers, own
what must be callable on a subset:

- Greedy assignment with area-range ignores is a few hundred lines whose bugs are
  silent, and a reimplementation's mAP compares to no published COCO result.
  `types-pycocotools` types the array shapes crossing back.
- `COCOeval.accumulate()` reads `evalImgs` for every image it was given and takes
  no subset, which is what a resample is. Correctness of the replacement is
  established by equality with `accumulate()` on the identity resample rather
  than by hand-computed fixtures.
"""
