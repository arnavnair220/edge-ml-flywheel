# Plane 1 — Data and label supply

Plane 1 spans ingest, partitioning, selection and label purchase. It owns two invariants: labels are
obtainable only by paying the oracle, and the eval set is fixed at partition time and never enters
training. See the [architecture overview](00-overview.md) for the plane's position in the loop.

---

## Ingest

Converts the BDD100K distribution archives into the immutable `raw/` prefix and the derived image
manifest. One CodeBuild job, run once per label format.

### Source archives

| Archive | Bytes | sha256 |
|---|---|---|
| `bdd100k_images_100k.zip` | 5,669,071,832 | not published |
| `bdd100k_labels.zip` | 189,638,612 | `7f1f9043c70a6ff0788a323cfb914aeced109a37b7150740d670a347c881394a` |

Both are served over plain HTTP from a UC Berkeley host with no authentication, and only one
publishes a digest, so all integrity checks run on received content. The data is licensed to the UC
Regents for non-commercial research use, accepted on the distribution page.

`bdd100k_labels.zip` is the 2018 Scalabel per-image format, supplying both the detection boxes and
the `weather`, `scene` and `timeofday` attributes the eval slices are defined over. The `det_20`
release requires authenticated access: the file of that name on this host contains 2,000 tracking
JPEGs and no labels. The format is recorded as `label_source` on every manifest row and as the
`raw/labels/scalabel/` prefix, both from one constant, so changing formats is a re-ingest.

### Test split exclusion

The archive contains 100,000 labeled images: 70,000 train, 10,000 val and 20,000 test. The test
annotations are complete ground truth (367,728 boxes, 18.39 per image against 18.41 for train) that
the benchmark does not publish. Ingest retains train and val only, giving a pool of exactly 80,000
images. The exclusion is enforced at three levels:

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
    partition_version=<v>/
      assignments/part-*.parquet              80,000 rows of image_id and cohort
      labels/cohort={bootstrap,eval}/         boxes for the two labeled cohorts
    purchases/run_id=<id>/cycle=<n>/          boxes one cycle bought
```

Only boxes are stored under `derived/`. Training reads its images from `raw/images/`, so no labeled
image is held a second time. The cumulative labeled set is a scattered subset of that prefix rather
than a range within it, so a cycle names its images object by object in a manifest at
`training_manifest_key`, which a SageMaker `ManifestFile` channel reads directly.

Access rules over the layout:

- `raw/` is immutable and writable only by the ingest role.
- The training role is denied `raw/labels/` in both its own policy and the bucket policy.
- The training role reads `raw/images/100k/train/` and no wider; split is a path component.
- `bootstrap` labels are copied out of `raw/labels/` rather than read from it.
- `labels/cohort=` is writable only by the partition role, on every partition version.
- `_provenance/` is underscore-prefixed, so Glue and Athena skip it.
- All keys are constructed by `edge_ml_flywheel.conventions`.

### Manifest schema

One row per image: `image_id`, `split`, `weather`, `scene`, `timeofday`, `n_boxes`, `box_areas`,
`sha256`, `label_source`.

- `box_areas` is in native 1280x720 pixels, not a small-object count. Integrity results are
  judgements and are written to `_provenance/integrity.json`.
- No `cohort` column. Assignment is a function of the partition and lives in `assignments/`, keyed by
  `partition_version`.
- `weather`, `scene` and `timeofday` are `StrEnum`s over vocabularies measured across all 80,000
  images: seven weather values, seven scene values, four for time of day, with `undefined` a
  populated member of each rather than a null. The literals are the archive's own — `dawn/dusk`
  carries a slash, `gas stations` is plural — and are never S3 key components. The parquet columns
  remain `string`.

The manifest precedes partitioning: cohort sizing, eval composition and per-slice counts are all
queries against it.

### Object class vocabulary

The label parser selects boxes by structure — an object carrying `box2d` rather than an `area/*` or
`lane/*` polygon — so the boxed vocabulary is measured at ingest and recorded in
`_provenance/integrity.json`. Ten categories carry boxes, from `car` at 816,423 to `train` at 151.

`conventions.CLASS_SET` names which of them a model predicts, and so which the metric covers:
`car`, `traffic sign`, `traffic light`, `person`, `truck`, `bus`, `bike`, `rider`, `motor`.

Nine rather than the four COCO-native ones, because a COCO-pretrained detector starts strong on
`car`, `person`, `truck` and `bus` and a cycle's labels have little left to move. One set rather than
a versioned table: the project trains one kind of model, so there is nothing to select between and no
version to carry. Declaration order is permanent — a category ID is a position in that tuple and is
stored in every cached match array — so a class is appended, never inserted or reordered.

`train` is the tenth boxed category and is deliberately outside the set: at 151 boxes archive-wide it
is too rare to learn or to score.
Both follow `partition_version`'s rule: add a version, never edit one.

- Category IDs are 1-based positions in the version's tuple, ordered by measured frequency, and are
  stored in every cached match array.
- A class set is validated against the measured vocabulary at construction. The legacy archive spells
  three categories `person`, `motor` and `bike` where `det_20` says `pedestrian`, `motorcycle` and
  `bicycle`; an unmatched spelling would score 0.0 AP every cycle without failing.
- `train` is in neither set, at 151 boxes archive-wide.

### Verification

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

CodeBuild, outside any VPC, under an IAM role, with CloudWatch logs. Build containers are ephemeral,
which keeps the extracted annotations inside the label wall.

`buildspecs/ingest.yml` defines the sequence. Shell commands handle download, extraction and upload;
all interpretation of the data is in `edge_ml_flywheel.ingest`. The buildspec contains no S3 keys:
staging materializes a local tree at the keys `conventions` produces, and the upload is a recursive
copy of that tree.

- A sixteen-byte range request precedes the 5.7 GB download; a 206 confirms `curl -C -` can resume.
- The labels archive is fetched first, so a changed host fails the build within a minute.
- The upload phase asserts `CODEBUILD_BUILD_SUCCEEDING`, since `post_build` executes regardless of
  build outcome and this phase writes to the immutable bucket.

### Invocation

There is no webhook, and a run re-downloads 5.7 GB and rewrites `raw/`, so builds are started by
hand. The CLI and the console's Start build action are equivalent.

```
aws codebuild start-build --project-name edge-ml-flywheel-ingest
```

Two values are overridable per build. The source version selects a commit or branch, defaulting to
`refs/heads/main`. `BDD100K_HOST` selects the endpoint: the download page's buttons point at a raw
IP, and `dl.yf.io` resolves to the same host and serves byte-identical files.

---

## Partition

The partitioner runs once per `partition_version` and assigns each of the 80,000 images to exactly
one cohort. The assignment is disjoint and complete, so `cohort=` is valid as a storage prefix; both
properties are asserted before either output file is written.

| Cohort | Images | Split | Labels |
|---|---:|---|---|
| `eval` | 5,000 | `val` | Labeled at partition time, never trained on |
| `bootstrap` | 8,000 | `train` | Labeled at partition time; the champion's starting set |
| `pool` | 62,000 | `train` | Withheld, purchasable one budgeted batch at a time |
| `reserve` | 5,000 | `val` | Untouched |

Eval comes from `val` and everything trainable from `train`. The bootstrap is random, not curated. No
condition is withheld from the pool, so it is IID across weather, scene and time of day from cycle
one. `reserve` is left alone for the life of the run.

### The draw

`conventions.PARTITIONS` records the seed and the four cohort sizes per `partition_version`, and the
partitioner takes a version and no other argument. Within a split, images are ordered by
`sha256(<seed>:<image_id>)` and each cohort takes the next `n`; `COHORT_SPLIT` fixes which split a
cohort draws from and the order the cohorts draw in — `eval` before `reserve`, `bootstrap` before
`pool`, so a later version that spends the reserve at the same seed holds every earlier `eval` image.
No predicate over `weather`, `scene` or `timeofday` appears in the assignment. Rows are sorted by
`image_id` on output, so two runs of one version produce byte-identical parquet.

The partition assertion covers 80,000 rows, unique `image_id`, every manifest ID assigned, each
cohort at its specified size, and no cohort holding an image from the other split. It runs against
the manifest rather than against the draw's own bookkeeping, and before either file is written.

Output is `assignments/part-00000.parquet` (715 KB) and `_partition.json`, which records the seed,
the sizes and the draw rule so they are answerable from the bucket; the `PARTITIONS` entry stays
authoritative. Realized cohort compositions are logged rather than stored.

```
python -m edge_ml_flywheel.partition assign --stage-dir ./stage --partition-version 0
```

### The labels the draw fixes

`bootstrap` and `eval` own labels from the moment they are drawn, so their boxes are copied out of
`raw/labels/` into `labels/cohort=<name>/part-00000.parquet` under the same partition prefix. Two
columns: `image_id`, and the boxes in the compact encoding `oracle.labels` writes a purchase in, so
training decodes an owned label and a bought one with one function.

The copy runs as three steps, because which documents to fetch is a function of the draw that just
happened. `label-keys` names them off the assignments, the shell copies exactly that list, and
`labels` reads back only what it named. Nothing enumerates `raw/labels/` — `ListBucket` on the
partition role stays scoped to `derived/`, and `cohort_labels.label_key` raises on any cohort outside
`LABELED_COHORTS`. Each file is checked for exactly its cohort's ID set, not a count, before it is
written. Both prefixes are write-denied to everything but the partitioner.

```
python -m edge_ml_flywheel.partition label-keys --stage-dir ./stage --partition-version 0
python -m edge_ml_flywheel.partition labels --stage-dir ./stage --partition-version 0
```

### Eval composition

`eval` is not stratified. No quota is placed on any slice, and the counts below are what the uniform
draw over `val` yielded at `partition_version` 0 — a measurement of the cohort, not a target it was
sampled to:

| Axis | Slice | Images | Axis | Slice | Images |
|---|---|---:|---|---|---:|
| `weather` | `clear` | 2,687 | `scene` | `city street` | 3,032 |
| | `overcast` | 628 | | `highway` | 1,260 |
| | `undefined` | 581 | | `residential` | 642 |
| | `rainy` | 391 | `timeofday` | `daytime` | 2,613 |
| | `partly cloudy` | 361 | | `night` | 1,995 |
| | `snowy` | 349 | | `dawn/dusk` | 378 |

`val` tracks the 80,000-image pool within 1.0 point on every tag value, so overall mAP is the
fleet-weighted number it appears to be with no per-slice quota enforcing it. Nothing pins a slice to
a count, so each lands somewhat off its pool share — `snowy` at 349 against 396, `city street` at
3,032 against 3,106.

No slice votes and none carries a minimum. Promotion is decided on the overall metric over all 5,000
images; per-slice scores are computed, charted, and read against a cycle's purchase mix in the
composition chart. Slices run from `city street` at 3,032 images down to `gas stations` at 4. Fog is
a permanent limit: 143 images in the full 80,000.

Slices are keyed `axis=value`, since `undefined` is a populated member of all three vocabularies and
names three different slices. Class and object-scale slices are queries over the eval cohort's
labels, not partition decisions, and the scale threshold belongs to the evaluation module: at 416 px,
87.8 percent of every box in the pool is small by the COCO rule.

### Execution environment

CodeBuild under its own role, not ingest's. It reads `derived/manifest/` and `raw/labels/`, and
writes one `derived/partition_version=` prefix. It appears in `label_reader_arns` and not in
`raw_writer_arns`, so the bucket policy permits those reads and denies every write to `raw/`. It is
the second of the two principals on the label-reader allowlist. `ListBucket` stays scoped to
`derived/`: it can read a label key it can name and cannot walk the label tree to discover one.

`buildspecs/partition.yml` defines the sequence and contains no S3 keys: `partition prefix` prints
both prefixes out of `conventions`, and each copy is a recursive copy of one of them. The manifest is
staged at the key it has in S3.

- A version already in the bucket is checked against this commit before it is redrawn. A re-run of an
  unchanged version is byte-identical.
- The upload covers the partition prefix only; writing the manifest back is a permission the role
  does not hold.
- The uploaded document is read back and re-checked, applying the pre-draw comparison to what landed.

```
aws codebuild start-build --project-name edge-ml-flywheel-partition
```

There is no webhook. `PARTITION_VERSION` selects the version, defaults to 0, and is overridable per
build; a value outside `conventions.PARTITIONS` is refused before the manifest is read.

---

## Run registration

A run opens before any cycle turns. Registration mints a `run_id`, claims it, and records the two
version stamps and the label budget. Nothing downstream can write until it
exists, since `run_id` is the partition key of every stateful table and the top prefix of every
artifact.

The claim is a conditional put on `attribute_not_exists(run_id)`. The role holds `PutItem` and
`GetItem` on `runs` and no S3 at all, so the item is write-once: the condition refuses a second
write, and the absent `UpdateItem` refuses an edit. The job runs in CodeBuild rather than as a
script, so `git_commit` cannot name a commit that did not produce the run.

### Invocation

```
aws codebuild start-build --project-name edge-ml-flywheel-register \
  --environment-variables-override \
    name=RUN_SLUG,value=v1-uncertainty,type=PLAINTEXT \
    name=RUN_NOTE,value="first real loop",type=PLAINTEXT
```

The build prints the minted `run_id`, which is the input to every later step. `RUN_SLUG` and
`RUN_NOTE` have no defaults. `SELECTOR`, `LABEL_BUDGET`, `PARTITION_VERSION`, `CLASS_SET_VERSION` and
`RECIPE_VERSION` default to the loop as designed and are overridable per build.

There is no webhook: this job mints a new `run_id` on every invocation, so a push to `main` would
start a run.

---

## Selection and purchase

One cycle spends its budget in five steps:

1. The champion scores every remaining pool image as a batch transform job. This is inference over
   image features only; no label is read. The fleet's own scores over the frames it replayed are
   reported beside this ranking, never used to rank the purchase.
2. Each image gets a **mean per-object uncertainty** score, rather than an image-level maximum, which
   would be decided by a frame's single worst box.
3. The top 1,000 of the ranked list are the batch, bought unfiltered.
4. The batch's `weather` and `timeofday` mix is recorded beside the remaining pool's.
5. The oracle checks the idempotency key, debits the ledger, releases those labels, and appends them
   to the cumulative labeled set.

The budget is a property of the run, not a constant of the system: it is set at registration, and the
oracle creates a cycle's ledger item on that cycle's first purchase, seeded from the registered
figure. Creation and the first debit are one conditional write.

### The charge

Step 5 is a single `TransactWriteItems` over two tables, each carrying its own condition.

| Item | Condition | Refuses |
|---|---|---|
| `audit_log` put, keyed `purchase#c<cycle>#<digest>` | `attribute_not_exists(event)` | The second charge for a batch already bought |
| `label_budget` update, `SET remaining = if_not_exists(remaining, :budget) - :n` | `attribute_not_exists(remaining) OR remaining >= :n` | The overspend, and the negative balance |

One write rather than two in sequence: the key and the ledger are in different tables, so two writes
leave a window a crash cannot be resumed from. The transaction applies both or neither.

The digest is over the sorted image IDs, so a retry that re-ranks a tie is the same purchase. A batch
larger than the whole cycle's budget is refused before the call, since on a first purchase there is
no `remaining` to compare against. When both conditions refuse, the audit condition is the one
reported — that case is a retry of a purchase that already succeeded. A replay returns the original
receipt and re-serves the same labels, which is what lets the label write after it repeat over the
same keys.

| Quantity | Value |
|---|---|
| Budget per cycle | 1,000 labels, per the run registration |
| Selected by uncertainty | 1,000, the whole batch |
| Cycles per run | 8, planned rather than enforced |
| Total purchased | 8,000 at eight cycles |
| Training set | 8,000 at cycle 0, 16,000 after cycle 8 |
| Pool remaining | 62,000, falling to 54,000 |
| Selectivity | 1,000 of 62,000, about 1.6 percent |

### The selector

Steps 2 and 3 are one rule: rank the remaining pool by mean per-object uncertainty and take the top
of it. `selection.select` takes a pool, its scores and a budget, and nothing that names a rule —
there is no selector field on a run and no argument to pass, because every run ranks the same way.

The controls that would buy by a different rule are deferred, and adding one means adding a rule
rather than setting a value. See the planned additions in `00-overview.md`.

### The label wall

The oracle is the only route ground truth takes into the system, enforced structurally: the training
role is denied `raw/labels/` in both its own policy and the bucket policy. The ledger is keyed by
`run_id` and every purchase carries an idempotency key.

The oracle reads labels out of `raw/labels/` directly, with no per-run copy and no intermediate
table. Cohort is a column in the assignments parquet rather than a component of any key, so no
storage boundary separates a purchasable `pool` label from the `eval` labels beside it, and no IAM
policy can express one. `edge_ml_flywheel.oracle.cohorts` is the single place the eval guarantee
lives.

- The gate runs before a key is built, and the key builder routes through it, so a refused image is
  unread rather than unsold.
- One non-pool image refuses the whole batch. Filtering would charge a run for a batch it did not ask
  for.
- A repeated image ID refuses the batch: a duplicate is one label billed twice.
- `bootstrap` is refused with the rest. Its labels are already owned, and it reaches training through
  its own label file, so the oracle does one thing: charge, then serve.

`eval` labels are read only by the evaluation plane: never purchasable, never appended to the
training set, never re-drawn within a run. The gate refuses them by cohort, and no key builder in the
oracle can address the split they are drawn from. Zero image-ID overlap between the labeled set and
`eval` is a hard gate failure with no override, and is the backstop rather than the mechanism.

The eval boxes exist a second time under `labels/cohort=eval/`, outside the `raw/labels/` deny.
`EvalLabelsAreScoringOnly` denies reads on that prefix to every principal; the evaluation plane is
named there when it is built. `cohort=bootstrap/` carries no read deny, because training owns those
8,000 labels. Both prefixes are write-denied to every principal but the partitioner by
`LabelsAreFrozenExceptThePartitioner`, so cycle eight's number is comparable to cycle one's.
