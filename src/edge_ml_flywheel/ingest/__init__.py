"""BDD100K ingest: archives in, `raw/` and the manifest out.

Runs in CodeBuild and nowhere else (see `buildspecs/ingest.yml` for why). Every
step here is a subcommand of `python -m edge_ml_flywheel.ingest`, with the shell
doing the fetching and copying and this package owning every decision about what
the data is.

Deliberately free of AWS SDK calls. The steps operate on local directories and
on a listing file the shell produces, so the whole module is exercisable from a
temp directory with no credentials -- which is what makes the unit tests real
tests rather than mocks. The one thing that talks to S3 is `aws s3 cp`, and it
copies a tree whose paths are already the keys.

The submodules, in the order a build passes through them:

- `source` -- what we download and what we expect it to be
- `labels` -- parsing one Scalabel document
- `images` -- one image's digest and its integrity facts
- `stage` -- the extracted archives moved to the keys `conventions` builds
- `manifest` -- the 80,000-row parquet, plus the integrity report
- `provenance` -- where the bytes came from, recorded beside them
"""
