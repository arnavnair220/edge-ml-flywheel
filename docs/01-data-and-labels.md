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

The archive contains 100,000 labeled images: 70,000 train, 10,000 val and 20,000 test. The test
annotations are complete ground truth (367,728 boxes, 18.39 per image against 18.41 for train), which
the benchmark does not publish. Ingest retains train and val only, giving a pool of exactly 80,000
images. Excluding the split forgoes 25 percent additional unlabeled capacity; the binding constraint
is the label budget, not pool size, and results reported against this dataset must not depend on
withheld annotations.

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
    partition_version=<v>/
      assignments/part-*.parquet              80,000 rows of image_id and cohort
      labels/cohort={bootstrap,eval}/         boxes for the two labeled cohorts
    purchases/run_id=<id>/cycle=<n>/          boxes one cycle bought
```

Only boxes are stored under `derived/`. Training reads its images from `raw/images/`, so no labeled
image is held a second time. The cumulative labeled set is a scattered subset of that prefix rather
than a range within it, so a cycle names its images object by object in a manifest at
`training_manifest_key`, which a SageMaker `ManifestFile` channel reads directly.

| Property | Rationale |
|---|---|
| `raw/` is immutable and writable only by the ingest role | All reprocessing reads from it, so inference outputs can be regenerated offline |
| The training role is denied `raw/labels/` in both its own policy and the bucket policy | The cohort gate governs what the oracle sells; only an explicit deny governs what another job reads directly |
| `bootstrap` labels are copied out of `raw/labels/` rather than read from it | `bootstrap` and `pool` are siblings under `train/` and cohort is a column, so the grant that would give training its own 8,000 labels would give it the 62,000 it is supposed to buy |
| `labels/cohort=` is writable only by the partition role, on every partition version | The two files are the ruler and the starting line for every cycle of a run, and `derived/` is otherwise writable by ingest too |
| The training role reads `raw/images/100k/train/` and no wider | Split is a path component, so an eval image stays unreachable even if a manifest names one |
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
- **`weather`, `scene` and `timeofday` are enums over the measured vocabularies.** Their value
  domains are properties of the archive, so they were counted over all 80,000 images before being
  written down: seven weather values, seven scene values, four for time of day, with `undefined` a
  populated member of each rather than a null. Every eval slice and the selection mix record is a
  predicate over these three columns, and a misspelled value raises nothing and matches nothing, so
  the slice empties and charts as a flat line. The literals are the archive's own — `dawn/dusk` carries a
  slash, `gas stations` is plural — which also makes a tag the one categorical here that is never an
  S3 key component. `StrEnum` serializes identically, so the parquet columns remain `string` and the
  typing cost no re-ingest.

The manifest precedes partitioning because cohort sizing, eval composition and per-slice counts are
all queries against it.

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
the bucket policy and by the oracle's cohort gate, neither of which covers a copy held on a
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
so an hour-long job can be iterated on without pushing to `main`. `BDD100K_HOST`
selects the endpoint: the download page's buttons point at a raw IP, and `dl.yf.io` resolves to the
same host and serves byte-identical files, so a name that stops answering is a per-build override
rather than a re-deploy.

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

| Property | Rationale |
|---|---|
| Eval comes from `val`, everything trainable from `train` | BDD already publishes the split, so every leakage question is answerable by naming a cohort's source rather than by re-deriving an image ID set |
| The bootstrap is random, not curated | A seed stratified to be easy or hard moves part of the loop's measured gain into the partition instead of the selector |
| `reserve` exists and is left alone | Growing the eval set mid-run invalidates every earlier cycle's comparison, so a larger eval has to be a next-run decision rather than a dead end |
| No condition is withheld from the pool | The pool is IID across weather, scene and time of day from cycle one, so any concentration in what gets bought comes from the selector rather than a release schedule |

### The draw

`conventions.PARTITIONS` records the seed and the four cohort sizes per `partition_version`, and the
partitioner takes a version and no other argument. Within a split, images are ordered by
`sha256(<seed>:<image_id>)` and each cohort takes the next `n`; `COHORT_SPLIT` fixes which split a
cohort draws from and the order the cohorts draw in. No predicate over `weather`, `scene` or
`timeofday` appears in the assignment.

| Property | Rationale |
|---|---|
| The seed is a property of the version, not a run argument | Every cycle's result is conditional on which 8,000 images the champion started from. A per-run seed is a number that can be mistyped into a partition that is valid, different and indistinguishable from the intended one |
| A digest per image rather than a shuffle of the split | A shuffle depends on the order its input arrived in, and the manifest is built by walking a directory, so a re-ingest that lists files differently would repartition the dataset under the same seed. A digest also removes the dependency on `random`, whose sampling algorithm CPython does not fix across versions |
| Cohort sizes are recorded per version alongside the seed | Growing `eval` moves the ruler earlier cycles were measured against, so it is a new version rather than a re-run of an existing one |
| `eval` draws before `reserve`, `bootstrap` before `pool` | A later version that spends the reserve at the same seed holds every earlier `eval` image, so a larger eval adds images rather than exchanging them |
| Rows are sorted by `image_id` on output | Two runs of one version produce byte-identical parquet, so "this is the same partition" is a checksum rather than a claim |

The partition assertion covers 80,000 rows, unique `image_id`, every manifest ID assigned, each
cohort at its specified size, and no cohort holding an image from the other split. It runs against
the manifest rather than against the draw's own bookkeeping, and it runs before either file is
written.

Output is `assignments/part-00000.parquet` (715 KB) and `_partition.json`, which records the seed,
the sizes and the draw rule. The JSON duplicates the `PARTITIONS` entry, which stays authoritative,
and exists so the seed is answerable from the bucket rather than from a source checkout at an unknown
commit. Realized cohort compositions are logged rather than stored, since they are a query over the
manifest joined to the assignments.

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
partition role stays scoped to `derived/`, so a key is reachable only by being named from the
assignments, and `cohort_labels.label_key` raises on any cohort outside `LABELED_COHORTS`.

| Property | Rationale |
|---|---|
| `bootstrap` is copied because it must be | Its labels sit beside `pool`'s under `train/` with cohort as a column, so no prefix names them and `raw/labels/scalabel/train/` is ungrantable to training |
| `eval` is copied although it need not be | `val` holds only `eval` and `reserve`, so a raw grant would leak nothing. Copying it anyway makes both labeled cohorts one mechanism rather than a second path existing to save a file |
| The partition role reads all of `raw/labels/` | The grant is the whole label tree or nothing, for the reason `bootstrap` needs the copy. A separate job for the write would hold the identical grant behind another role and buildspec |
| Each file is checked for exactly its cohort's IDs before it is written | Set equality, not a count: a count passes a file holding one `pool` image and missing one `bootstrap` image, which is the shape a wrong key list produces |
| Both prefixes are write-denied to everything but the partitioner | Immutability under the one exempt writer is the redraw guard, which is what `assignments/` already runs under |

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

Uniform rather than proportional because a draw at `val`'s own rates reproduces those rates in
expectation, and `val` tracks the 80,000-image pool within 1.0 point on every tag value. Overall mAP
is therefore the fleet-weighted number it appears to be, with no per-slice quota enforcing it. What
that costs is exactness: nothing pins a slice to a count, so each lands somewhat off its pool share —
`snowy` at 349 against 396, `city street` at 3,032 against 3,106. The gap is `val`'s share differing
from the pool's plus the draw being one sample of it.

No slice votes. Promotion is decided on the overall metric over all 5,000 images, and the per-slice
scores are computed and charted. Acquisition is condition-blind — the selector ranks on uncertainty
and never targets a slice — so a cycle makes no per-condition bet for a per-condition gate to
settle. The spread above is the second reason: slices run from `city street` at 3,032 images down to
`gas stations` at 4, and a veto from the small end would fire on chance alone, while excluding them
would take a size floor, a per-slice noise band and a persistence rule.

The slices feed the composition chart, which reads a cycle's purchase mix against how each
condition's score then moved. None carries a minimum, since none of them votes.

Slices are keyed `axis=value`, since `undefined` is a populated member of all three vocabularies and
names three different slices. Fog is a permanent limit rather than a sizing choice: 143 images in
the full 80,000, so no draw from this dataset makes the slice large.

Class and object-scale slices are queries over the eval cohort's labels, not partition decisions,
and the scale threshold belongs to the evaluation module that owns the metric: at 416 px, 87.8
percent of every box in the pool is small by the COCO rule, so the threshold moves such a slice far
more than any partition choice does.

### Execution environment

The partitioner runs in CodeBuild under its own role, not ingest's. It reads `derived/manifest/` and
`raw/labels/`, and writes one `derived/partition_version=` prefix. It appears in
`label_reader_arns` and not in `raw_writer_arns`, so the bucket policy permits those reads and
denies it every write to `raw/`. `ListBucket` stays scoped to `derived/`, so it can read a label key
it can name and cannot walk the label tree to discover one.

That makes two principals on the label-reader allowlist rather than one. The widening is the floor
rather than a concession: some job has to read a raw label to write the cohort files, and a separate
job for it would hold the identical grant behind another role and buildspec. What bounds it is in
the package — `cohort_labels.label_key` refuses any cohort outside `LABELED_COHORTS`, the staged key
list is derived from the assignments, and each output file is checked for exactly its cohort's IDs
before it is written.

`buildspecs/partition.yml` defines the sequence and contains no S3 keys: `partition prefix` prints
both prefixes out of `conventions`, and each copy is a recursive copy of one of them. The manifest is
staged at the key it has in S3, so the partitioner reads it where `conventions` says it is.

Three properties of the sequence:

- A version already in the bucket is checked against this commit before it is redrawn. A re-run of
  an unchanged version is byte-identical; an edited seed would replace the assignments every
  existing run was measured against.
- The upload covers the partition prefix only. The manifest was staged to be read, and writing it
  back is a permission the role does not hold.
- The uploaded document is read back and re-checked, applying the pre-draw comparison to what
  landed.

```
aws codebuild start-build --project-name edge-ml-flywheel-partition
```

There is no webhook: a push to `main` is not a reason to redraw the partition. `PARTITION_VERSION`
selects the version, defaults to 0, and is overridable per build; a value outside
`conventions.PARTITIONS` is refused before the manifest is read.

---

## Run registration

A run opens before any cycle turns. Registration mints a `run_id`, claims it, and records the
selector, the three version stamps and the label budget. Nothing downstream can write until it
exists, since `run_id` is the partition key of every stateful table and the top prefix of every
artifact.

| Property | Rationale |
|---|---|
| The claim is a conditional put on `attribute_not_exists(run_id)` | An id is a UTC second plus a slug, so a collision is unlikely rather than impossible, and a second run under one adopts the first's ledger, champion and locks |
| The role holds `PutItem` and `GetItem` on `runs`, and no S3 at all | The condition refuses a second write; the absent `UpdateItem` refuses an edit to a write-once item |
| The job runs in CodeBuild, not as a script | `git_commit` is required and written once, and CodeBuild resolves the source, so the field cannot name a commit that did not produce the run |

### Invocation

```
aws codebuild start-build --project-name edge-ml-flywheel-register \
  --environment-variables-override \
    name=RUN_SLUG,value=v1-uncertainty,type=PLAINTEXT \
    name=RUN_NOTE,value="first real loop",type=PLAINTEXT
```

The build prints the minted `run_id`, which is the input to every later step. `RUN_SLUG` and
`RUN_NOTE` have no defaults, since a default slug names two runs the same thing and a default note
records nothing. `SELECTOR`, `LABEL_BUDGET`, `PARTITION_VERSION`, `CLASS_SET_VERSION` and
`RECIPE_VERSION` default to the loop as designed and are overridable per build.

There is no webhook, and unlike the two data jobs that is not a preference: those are idempotent,
and this one mints a new `run_id` on every invocation, so a push to `main` would start a run.

---

## Selection and purchase

One cycle spends its budget in five steps:

1. The champion scores every remaining pool image. This is inference over image features only; no
   label is read, so the step sits entirely inside the label wall.
2. Each image gets a **mean per-object uncertainty** score. Per-object rather than per-image,
   because an image-level maximum is decided by its single worst box and ranks a frame with one
   ambiguous detection above a frame the model is uniformly unsure of.
3. The top 1,000 of the ranked list are the batch.
4. The batch's `weather` and `timeofday` mix is recorded beside the remaining pool's.
5. The oracle checks the idempotency key, debits the ledger, releases those labels, and appends them
   to the cumulative labeled set.

The budget is a property of the run, not a constant of the system: it is set at registration, and
the oracle creates a cycle's ledger item on that cycle's first purchase, seeded from the registered
figure. Creation and the first debit are one conditional write, so a budget item's only writer is
the step that spends against it and its value never rises.

### The charge

Step 6 is a single `TransactWriteItems` over two tables, each carrying its own condition.

| Item | Condition | Refuses |
|---|---|---|
| `audit_log` put, keyed `purchase#c<cycle>#<digest>` | `attribute_not_exists(event)` | The second charge for a batch already bought |
| `label_budget` update, `SET remaining = if_not_exists(remaining, :budget) - :n` | `attribute_not_exists(remaining) OR remaining >= :n` | The overspend, and the negative balance |

One write rather than two conditional writes in sequence. The idempotency key and the ledger are in
different tables, so two writes leave a window in which one has landed and the other has not, and a
crash there is unrecoverable because the retry cannot tell which half it is resuming. The
transaction applies both or neither.

The digest is over the sorted image IDs, so a retry that re-ranks a tie is the same purchase rather
than a second one. A batch larger than the whole cycle's budget is refused before the call: on a
first purchase there is no `remaining` for the condition to compare against, and the create branch
would otherwise write a negative.

When both conditions refuse, the audit condition is the one reported. That case is a retry of a
purchase that already succeeded, and answering `over budget` would send the caller to buy less when
the correct answer is that the work is done. A replay returns the original receipt and re-serves the
same labels, which is what allows the label write after it to be repeated over the same keys.

| Quantity | Value |
|---|---|
| Budget per cycle | 1,000 labels, per the run registration |
| Selected by uncertainty | 1,000, the whole batch |
| Cycles per run | 8, planned rather than enforced |
| Total purchased | 8,000 at eight cycles |
| Training set | 8,000 at cycle 0, 16,000 after cycle 8 |
| Pool remaining | 62,000, falling to 54,000 |
| Selectivity | 1,000 of 62,000, about 1.6 percent |

Selectivity determines whether selection can matter. At a quarter of the scored pool, any
ranking and a random draw converge on nearly the same training set; at 1.6 percent they diverge from
the first cycle. A 500-label budget is affordable but puts each cycle's gain closer to the quality
gate's noise band, so cycles would fail to promote for lack of signal rather than lack of learning.

### The selector is a config value

Steps 2 and 3 are one swappable function. All three rules are named before any of them is needed, so
the deferred label-efficiency arm — see [planned additions](00-overview.md#planned-additions) —
changes one field rather than adding a second code path. A selector change deliberately does not
force a fresh champion baseline, since the arms are paired against one.

`random` needs only the remaining pool and a seed, with no inference at all, which also makes it the
smoke test for the ranking-to-purchase path before a champion exists to score with.

### Why the ranked list is bought unfiltered

Raw top-N on uncertainty is known to degrade in two ways, with unequal consequences:

- **The top of the list is redundant.** Images are uncertain for shared reasons, so an unfiltered
  top 1,000 can be a thousand near-identical night highway frames. The budget is spent and the
  training set barely moves, which voids a cycle rather than degrading it.
- **High-uncertainty frames are often uninformative.** Motion blur, heavy occlusion and genuinely
  ambiguous objects all score highly and teach nothing that generalizes. This costs a fraction of a
  batch, and the fraction is unmeasured.

Neither is corrected. Any correction is a threshold — how much of a batch one condition may take, how
blurred is too blurred — and no cycle has yet reported what those thresholds should be. Selection
buys the ranking as it stands, and the mix record is what turns the first failure into a number.

Step 4 records the batch's `weather` and `timeofday` mix beside the remaining pool's. The pool is the
only available reference: condition tags come from the vehicle, so they exist on unbought frames, but
a true population proportion does not. A batch collapsed onto one condition is visible in that record
the cycle it happens; a batch whose mix tracks the pool's shows redundancy is not the problem.

The record is also what disambiguates a null result. A cycle that gains nothing over the random
control is either a ranking that does not work or a batch that was a thousand copies of one scene,
and the two call for opposite responses.

### Deferred

Both degradation modes above are left in place, along with redundancy inside a single condition
bucket. All three need a threshold or a finer signal than the manifest carries, and none can be sized
before the first cycles report what they bought.

What ends a run is also open. No cycle count is recorded at registration and the ledger seeds a
cycle whenever one asks, so the per-cycle budget holds and the run-level total does not. A fixed
count, a stalled gate or a total-spend ceiling all remain available without a schema change, since
cumulative spend is a sum over the run's ledger partition.

| Addition | Condition that unlocks it |
|---|---|
| A per-condition cap on the batch, sized off the remaining pool | The mix record shows a batch collapsed onto one condition |
| A random fraction of every batch, bought without reference to the champion | The uncertainty arm fails to beat the random control and the mix record does not explain why |
| Dedup on image embeddings | The cap is in place and batches stay redundant inside one bucket |
| A blur and exposure screen over a derived quality table | Unlabelable frames appear in what was bought |
| Seed disagreement in place of single-model uncertainty | The above are in place and cycles still fail the quality gate |

### The label wall

The oracle is the only route ground truth takes into the system, enforced structurally rather than
by convention: the training role is denied `raw/labels/` in both its own policy and the bucket
policy. The ledger is keyed by `run_id` and every purchase carries an idempotency key, since a
retried purchase that double debits has no undo and overstates the cost of every cycle after it.

The oracle reads labels out of `raw/labels/` directly, with no per-run copy and no intermediate
table. Cohort is a column in the assignments parquet rather than a component of any key, so no
storage boundary separates a purchasable `pool` label from the `eval` labels beside it, and no IAM
policy can express one. `edge_ml_flywheel.oracle.cohorts` is what does, which makes it the single
place the eval guarantee lives.

| Rule | Rationale |
|---|---|
| The gate runs before a key is built, and the key builder routes through it | A refused image is unread rather than unsold, and no function in the package returns a key under the split `eval` is drawn from |
| One non-pool image refuses the whole batch | Filtering would charge a run for a batch it did not ask for, and would turn a selector reaching into `eval` into a cycle that quietly bought fewer labels |
| A repeated image ID refuses the batch | The batch is a set of images to the oracle and a count to the budget, so a duplicate is one label billed twice |
| `bootstrap` is refused with the rest | Its labels are already owned, and serving them would put a zero-value charge inside the component whose premise is that no label is free |

`bootstrap` reaches training through its own label file, so the oracle does one thing: charge, then
serve.

`eval` sits behind the same wall for a different reason. Its labels are read only by the evaluation
plane, and are never purchasable, never appended to the training set and never re-drawn within a
run. It is unpurchasable because the gate refuses it by cohort and because no key builder in the
oracle can address the split it is drawn from. Zero image-ID overlap between the labeled set and
`eval` is a hard gate failure with no override, and is the backstop rather than the mechanism.

The eval boxes exist a second time under `labels/cohort=eval/`, outside the `raw/labels/` deny.
`EvalLabelsAreScoringOnly` denies reads on that prefix to every principal; the evaluation plane is
named there when it is built. The exposure is contamination rather than a bypassed budget, since
eval labels are never charged.

`cohort=bootstrap/` is the other half of that copy and carries no read deny, because training owns
those 8,000 labels and reading them is the point. Both prefixes are write-denied to every principal
but the partitioner by `LabelsAreFrozenExceptThePartitioner`, which is what makes cycle eight's
number comparable to cycle one's: the ruler and the starting line are two objects nothing else in
the account can replace, and `derived/` is otherwise writable by ingest as well.
