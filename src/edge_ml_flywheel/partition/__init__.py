"""The manifest in, one cohort per image out.

Runs once per `partition_version`. `conventions.PARTITIONS` fixes the seed and
the four cohort sizes per version, so the partitioner takes a version and no
other argument: a re-run of a version reproduces its assignment exactly, and a
different partition is a different version rather than a different afternoon.

Built on `ingest`'s shape and for its reasons -- local directories in, local
files out, no AWS SDK calls anywhere. The steps are exercisable from a temp
directory with no credentials, and the shell copies the staged tree to keys
`conventions` already built.

- `assign` -- the draw, the disjoint-and-complete check, and the two files
"""
