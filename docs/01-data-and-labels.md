# Plane 1 — Data and label supply

Plane 1 spans ingest, partitioning, selection and label purchase. It owns two invariants: labels are
obtainable only by paying the oracle, and the eval set is fixed at partition time and never enters
training. See the [architecture overview](00-overview.md) for the plane's position in the loop.

---

## Ingest

Ingest converts the BDD100K distribution archives into the immutable `raw/` prefix and the derived
image manifest. It runs once per label format as a single CodeBuild job.

### Source archives

| Archive | Bytes | sha256 |
|---|---|---|
| `bdd100k_images_100k.zip` | 5,669,071,832 | not published |
| `bdd100k_labels.zip` | 189,638,612 | `7f1f9043c70a6ff0788a323cfb914aeced109a37b7150740d670a347c881394a` |

Both are served over plain HTTP from a UC Berkeley host with no authentication, and only one
publishes a digest. All integrity checks therefore run on received content rather than on transfer
metadata. The data is licensed to the UC Regents for non-commercial research use, accepted on the
distribution page.

`bdd100k_labels.zip` is the 2018 Scalabel per-image format. It supplies both the detection boxes and
the `weather`, `scene` and `timeofday` attributes the eval slices are defined over. The `det_20`
release requires authenticated access: the file of that name on this host contains 2,000 tracking
JPEGs and no labels. The selected format is recorded as `label_source` on every manifest row and as
the `raw/labels/scalabel/` prefix. Both derive from one constant, so changing formats is a re-ingest
rather than a second manifest.

### Test split exclusion

The archive contains 100,000 labelled images: 70,000 train, 10,000 val and 20,000 test. The test
annotations are complete ground truth (367,728 boxes, 18.39 per image against 18.41 for train), which
the benchmark does not publish. Ingest retains train and val only, giving a pool of exactly 80,000
images. Excluding the split forgoes 25 percent additional unlabelled capacity; the label budget
rather than pool size is the binding constraint, and results reported against this dataset must not
depend on withheld annotations.

The exclusion is enforced at three levels:

- Test paths are excluded during extraction, so the files are never written to disk.
- `conventions.Split` defines `TRAIN` and `VAL` only, making the split unrepresentable in code.
- The verification suite asserts the row count and the absence of known test image IDs.

### Storage layout

```
s3://<project>-data-<account>/
  raw/                                        immutable, never rewritten
    images/100k/{train,val}/<id>.jpg          80,000 objects, ~4.6 GB
    labels/scalabel/{train,val}/<id>.json     80,000 objects
    _provenance/                              host, digests, fetch time, license, commit
  derived/
    manifest/part-*.parquet                   80,000 rows of image facts
```

| Property | Rationale |
|---|---|
| `raw/` is immutable and writable only by the ingest role | All reprocessing reads from it, which is what allows inference outputs to be regenerated offline |
| The training role is denied `raw/labels/` in both its own policy and the bucket policy | The `oracle_labels` table cannot prevent direct object access; an explicit deny can |
| `_provenance/` is underscore-prefixed | Glue and Athena skip such paths, so a crawler over `raw/` does not index it |
| All keys are constructed by `edge_ml_flywheel.conventions` | A key formatted at two call sites diverges silently, with the writer still succeeding |

### Manifest schema

One row per image: `image_id`, `split`, `weather`, `scene`, `timeofday`, `n_boxes`, `box_areas`,
`sha256`, `label_source`. Three schema decisions:

- **Columns record archive facts, not derived judgements.** `box_areas` is stored in native
  1280x720 pixels rather than as a small-object count, since a count would fix a threshold and a
  resolution that the source does not specify and would compete with the threshold eval applies at
  416 px. Integrity results are judgements and are written to `_provenance/integrity.json`.
- **No `cohort` column.** Cohort assignment is a function of the partition, not the archive, and is
  stored in a separate `assignments/` table keyed by `partition_version`. Including it here would
  require rewriting 80,000 rows of image facts per re-partition with no authoritative copy.
- **`weather`, `scene` and `timeofday` are typed as strings pending measurement.** Their value
  domains are properties of the archive. `StrEnum` serializes to the same representation, so
  adopting enums later requires no re-ingest.

The manifest precedes partitioning because cohort sizing, eval stratification and per-slice counts
are all queries against it.

### Verification

Two archives on this host do not match their filenames, and the images archive publishes no digest.
Verification runs against the extracted dataset before any upload, as `raw/` is write-once.

| Check | Establishes |
|---|---|
| Labels archive sha256 | The archive with a published digest matches it |
| 80,000 rows, unique IDs, 70,000 / 10,000 | The manifest has the expected cardinality and split ratio |
| No test image IDs | The excluded split did not survive extraction |
| Image ID set equals label ID set | Integrity of the undigested archive, and completeness of the download |
| Decodable, 1280x720, non-blank | The data gate's per-image integrity checks, applied at ingest |
| Staged key set equals the S3 listing | Upload completeness, which the pre-upload suite cannot assess |

Expected values are literals in the suite and are not derived from the code under test.

### Execution environment

Ingest runs in CodeBuild outside any VPC, under an IAM role, with CloudWatch logs. Build containers
are ephemeral, which keeps the extracted annotations inside the label wall: the wall is enforced by
the bucket policy and the `oracle_labels` table, neither of which covers a copy held on a
workstation. No downstream step requires browsable local data, since the dataset counts are queries
against the manifest.

`buildspecs/ingest.yml` defines the sequence. Shell commands handle download, extraction and upload;
all interpretation of the data is implemented in `edge_ml_flywheel.ingest`, which is linted, typed
and unit tested. The buildspec contains no S3 keys: staging materializes a local tree at the keys
`conventions` produces, and the upload is a recursive copy of that tree.

Three properties of the sequence:

- A sixteen-byte range request precedes the 5.7 GB download. A 206 response confirms that `curl -C -`
  can resume; without it, an interrupted transfer restarts from zero.
- The labels archive is fetched first. At 190 MB with a published digest, it fails the build within a
  minute if the host has changed what it serves.
- The upload phase asserts `CODEBUILD_BUILD_SUCCEEDING`, since `post_build` executes regardless of
  build outcome and this phase writes to the immutable bucket.

### Invocation

Ingest has no webhook, and a run re-downloads 5.7 GB and rewrites `raw/`, so builds are started by
hand. The CLI and the console's Start build action are equivalent, since the project's saved
configuration supplies the source, the buildspec path and a source version of `refs/heads/main`, and
the build assumes the ingest role whichever principal starts it.

```
aws codebuild start-build --project-name edge-ml-flywheel-ingest
```

Two values are overridable per build by either route. The source version selects a commit or branch,
which is what allows an hour-long job to be iterated on without pushing to `main`. `BDD100K_HOST`
selects the endpoint: the download page's buttons point at a raw IP, and `dl.yf.io` resolves to the
same host and serves byte-identical files, so a name that stops answering is a per-build override
rather than a re-deploy.

---

## Partition

The partitioner runs once per `partition_version` and assigns each of the 80,000 images to exactly
one cohort. The assignment is disjoint and complete, which is what makes `cohort=` valid as a
storage prefix and what the Phase 1 partition assertion checks.

| Cohort | Images | Split | Labels |
|---|---:|---|---|
| `eval` | 5,000 | `val` | Labeled at partition time, never trained on |
| `bootstrap` | 8,000 | `train` | Labeled at partition time; the champion's starting set |
| `pool` | 62,000 | `train` | Withheld, purchasable one budgeted batch at a time |
| `reserve` | 5,000 | `val` | Untouched |

| Property | Rationale |
|---|---|
| Eval comes from `val`, everything trainable from `train` | BDD already publishes the split, so every leakage question is answerable by naming a cohort's source rather than by re-deriving an image ID set |
| The bootstrap is random, not curated | A seed stratified to be easy or hard moves part of the loop's measured gain into the partition instead of the selector |
| `reserve` exists and is left alone | Growing the eval set mid-run invalidates every earlier cycle's comparison, so a larger eval has to be a next-run decision rather than a dead end |
| No condition is withheld from the pool | The pool is IID across weather, scene and time of day from cycle one, so any concentration in what gets bought comes from the selector rather than a release schedule |

### Eval stratification

`eval` is sampled in proportion to the pool rather than balanced across conditions, so that overall
mAP is the fleet-weighted number it appears to be. Proportional sampling of 5,000 yields:

| Axis | Slice | Images | Axis | Slice | Images |
|---|---|---:|---|---|---:|
| `weather` | `clear` | 2,673 | `scene` | `city street` | 3,106 |
| | `overcast` | 627 | | `highway` | 1,245 |
| | `undefined` | 581 | | `residential` | 585 |
| | `snowy` | 396 | `timeofday` | `daytime` | 2,629 |
| | `rainy` | 364 | | `night` | 1,998 |
| | `partly cloudy` | 352 | | `dawn/dusk` | 363 |

A slice is gated only if it holds at least 300 images. Below that the bootstrap noise band is wide
enough to admit almost any delta and the gate stops discriminating. Twelve slices clear the floor.
Six do not and are reported without being gated: `foggy` (9), `parking lot` (27), `tunnel` (10),
`gas stations` (2), and the `undefined` members of `scene` (26) and `timeofday` (11). Fog is not
measurable at any eval size drawn from this dataset, at 143 images in the full 80,000.

Class and object-scale slices are queries over the eval cohort's labels, not partition decisions,
and the scale threshold belongs to the evaluation module that owns the metric: at 416 px, 87.8
percent of every box in the pool is small by the COCO rule, so the threshold moves such a slice far
more than any partition choice does.

---

## Selection and purchase

One cycle spends its budget in five steps:

1. The champion scores every remaining pool image. This is inference over image features only; no
   label is read, so the step sits entirely inside the label wall.
2. Each image gets a **mean per-object uncertainty** score. Per-object rather than per-image,
   because an image-level maximum is decided by its single worst box and ranks a frame with one
   ambiguous detection above a frame the model is uniformly unsure of.
3. A diversity pass reduces the ranked list.
4. The top 1,000 go to the oracle.
5. The oracle checks the idempotency key, debits the ledger, releases those labels, and appends them
   to the cumulative labeled set.

| Quantity | Value |
|---|---|
| Budget per cycle | 1,000 labels |
| Cycles per run | 8 |
| Total purchased | 8,000 |
| Training set | 8,000 at cycle 0, 16,000 after cycle 8 |
| Pool remaining | 62,000, falling to 54,000 |
| Selectivity | 1,000 of 62,000, about 1.6 percent |

Selectivity is what decides whether selection can matter. Buying a quarter of what was scored is
barely a choice, and the uncertainty and random arms would land on nearly the same training set. At
1.6 percent they diverge from the first cycle. A 500-label budget is affordable but puts each
cycle's gain closer to the quality gate's noise band, so cycles would fail to promote for lack of
signal rather than lack of learning.

### Why the ranked list is not bought directly

Raw top-N on uncertainty degrades in two known ways, and the diversity pass exists for both:

- **High-uncertainty frames are often uninformative.** Motion blur, heavy occlusion and genuinely
  ambiguous objects all score highly and teach nothing that generalizes. Confusing and instructive
  are correlated, not identical.
- **The top of the list is redundant.** Images are uncertain for shared reasons, so an unfiltered
  top 1,000 can be a thousand near-identical night highway frames: one lesson bought a thousand
  times.

The pass caps how much of one cycle's purchase any single condition may take, and deduplicates near
neighbours in embedding space before the list reaches the oracle. Both are recorded per cycle: a cap
that binds every cycle reports that the ranking has collapsed onto one condition.

### The label wall

The oracle is the only route ground truth takes into the system, enforced structurally rather than
by convention: the training role is denied `raw/labels/` in both its own policy and the bucket
policy. The ledger is keyed by `run_id` and every purchase carries an idempotency key, since a
retried purchase that double debits has no undo and overstates the cost of every cycle after it.

`eval` sits behind the same wall for a different reason. Its labels are read only by the evaluation
plane, and are never purchasable, never appended to the training set and never re-drawn within a
run. Zero image-ID overlap between the labeled set and `eval` is a hard gate failure with no
override.
