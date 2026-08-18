# Plane 1 — Data and label supply

Plane 1 spans ingest, partitioning, wave release, selection and label purchase. It owns two
invariants: labels are obtainable only by paying the oracle, and the selection pool is the
cumulative union of released waves rather than the newest wave alone. See the
[architecture overview](00-overview.md) for the plane's position in the loop.

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
the `weather`, `scene` and `timeofday` attributes that the wave schedule is defined over. The `det_20`
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

The manifest precedes shard generation because wave sizing, eval stratification and per-slice counts
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
