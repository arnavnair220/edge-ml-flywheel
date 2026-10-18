"""BDD100K ingest: archives in, `raw/` and the manifest out.

Runs in CodeBuild and nowhere else (see `buildspecs/ingest.yml` for why). Every
step here is a subcommand of `python -m edge_ml_flywheel.ingest`, with the shell
doing the downloading and unzipping and this package owning every decision about
what the data is and every copy to the bucket.

Every step but the last two operates on local directories only, so it is
exercisable from a temp directory with no credentials. The last two, `upload`
and `verify-upload`, reach the data bucket through `storage.ObjectStore`: the
upload copies a tree whose paths are already the keys, and the check lists the
two prefixes it wrote.

The submodules, in the order a build passes through them:

- `source` -- what we download and what we expect it to be
- `labels` -- parsing one Scalabel document
- `images` -- one image's digest and its integrity facts
- `stage` -- the extracted archives moved to the keys `conventions` builds
- `manifest` -- the 80,000-row parquet, plus the integrity report
- `provenance` -- where the bytes came from, recorded beside them
"""
