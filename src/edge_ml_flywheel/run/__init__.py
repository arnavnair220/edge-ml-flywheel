"""Minting a run and claiming its name.

The first step of every run, and the one that makes the rest addressable:
`run_id` is the partition key of all five tables and the top prefix of every
artifact, so until the registration exists there is no key to write to.

Unlike `ingest` and `partition`, this package talks to AWS. It has to -- the
claim on a run's name *is* a conditional write against `Table.RUNS`, and there
is no local artifact that could stand in for it. The dependency is confined to
`registration`, which does nothing else, so the encoding of a registration is
still testable without credentials.

- `registration` -- the item encoding, and the conditional put that claims a run
"""
