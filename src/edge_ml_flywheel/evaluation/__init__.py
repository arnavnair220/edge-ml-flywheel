"""Scoring a detector, and every statistic read off one scoring pass.

Data in, values out. Nothing here opens a socket: no `boto3`, no `torch`, no
HTTP. The serialization helpers take a local path and are the package's only
contact with the outside world, so the test suite runs with no credentials and a
promotion decision is reproducible from a saved `.npz`.

The layer's shape follows from one number. The paired bootstrap needs 1,000
resamples over both models, and re-scoring per resample would be 2,000 full mAP
evaluations over 5,000 images for a single-seed cycle -- hours, and multiplied
again by every seed a run adds (design section 4.2). So scoring happens once and
every later statistic reads the cache:

- `coco` puts boxes and detections in the shapes `pycocotools` reads
- `match` scores once, keeping the per-image match arrays at every IoU threshold
  and area range in one pass
- `metrics` accumulates those arrays into AP over an arbitrary image list, which
  makes a resample a numpy operation rather than a re-scoring
- `bootstrap` resamples image IDs and reads the paired lower bound off `metrics`

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
