# The five DynamoDB tables. Names come from `conventions.Table`; keys are decided
# here and cannot be changed afterwards -- a different key design is a different
# table with a different name and a data migration, so this file is the one in the
# stack worth reading twice.
#
# **Every table carries `run_id` in its partition key.** A re-run must be
# physically unable to see the previous run's spent budget, promoted models or
# locks: not by a filter someone remembers to apply, but because the items live in
# a partition its key expression cannot name. `runs` is the only table whose
# partition key is `run_id` alone, because a registration is a function of the run
# and nothing else.
#
# **Sort keys follow the same rule the S3 layout follows** -- key a thing by
# exactly what its content is a function of. A lock is a function of the run and
# of which resource is locked, so it has a sort key even though the design names
# only one lock. Adding one later is not an option, and the cost of an unused
# dimension is one constant at the call site.
#
# **On-demand billing throughout.** Volumes are trivial -- five figures of items,
# single-digit writes a second at peak -- and provisioned capacity would trade a
# few cents for a throttle during the one cycle that matters.
#
# **Numbers in sort keys are padded only where the sort is lexicographic.** A
# DynamoDB `N` attribute sorts numerically, so `cycle` needs no padding here,
# unlike the same value in an S3 key. Where a sort key is a composed string, the
# padding rule from `conventions` applies inside it.

locals {
  table_prefix = var.project
}

# One item per run, written once at mint time and never updated. The conditional
# put on `attribute_not_exists(run_id)` against this table is what turns a
# second-precision timestamp collision into a loud failure at run start instead
# of a second run quietly adopting the first's label ledger -- which is why
# registration is mandatory rather than a nicety.
resource "aws_dynamodb_table" "runs" {
  name         = "${local.table_prefix}-runs"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "run_id"

  attribute {
    name = "run_id"
    type = "S"
  }

  # The registration is the only record of what a run was configured as. Losing
  # it makes every artifact under `run_id=.../` unattributable, and it is a few
  # kilobytes.
  point_in_time_recovery {
    enabled = true
  }

  deletion_protection_enabled = true
}

# There is deliberately no table of withheld labels.
#
# An earlier design copied the 62,000 `pool` labels into one per run, so that an
# `eval` label was unpurchasable by being absent from what the oracle could read.
# That bought the guarantee with a fifteen-minute job at the start of every run
# and a second copy of the archive per run, and it bought it in the wrong place:
# the copy existed only because IAM cannot express the pool/eval line, since
# cohort is a column in the assignments parquet and not a component of any key.
#
# The oracle now reads `raw/labels/` directly and enforces that line itself, in
# `edge_ml_flywheel.oracle.cohorts`. The gate runs before a key is built and the
# key builder routes through it, so a refused image is unread rather than merely
# unsold and no function in the package returns a key under the split `eval` is
# drawn from. `raw/labels/` stays denied at the bucket to everything but ingest
# and the oracle, so the budget is still not bypassable from outside.
#
# Recorded here rather than silently omitted, because the absence is a decision:
# five tables where the design once said six, and the sixth is not pending.

# The ledger. One item per (run, cycle) holding what remains of that cycle's cap,
# decremented by a conditional write that fails rather than going negative.
#
# The item is per cycle because the budget is per cycle, and a cycle that fails to
# promote still keeps what it bought. Cumulative spend is a sum over the
# partition rather than a running total in one item, so a lost update can
# overcount at most one cycle instead of corrupting the whole run's arithmetic.
resource "aws_dynamodb_table" "label_budget" {
  name         = "${local.table_prefix}-label_budget"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "run_id"
  range_key    = "cycle"

  attribute {
    name = "run_id"
    type = "S"
  }

  attribute {
    name = "cycle"
    type = "N"
  }

  # Cost per label spent is a headline deliverable and there is no undo on a
  # charge, so this is the table whose loss cannot be repaired by re-running
  # anything.
  point_in_time_recovery {
    enabled = true
  }

  deletion_protection_enabled = true
}

# Deployment intent, and the only mutable state the fleet reads. Two shapes of
# item share the table, distinguished by the sort key:
#
#   entity = "run"           the run's own config, including the current cycle
#   entity = "device#<n>"    one device's `desired_version`
#
# They belong together because they are read together -- a device resolving what
# to run and the control plane resolving which cycle is current are both a query
# on one `run_id` partition -- and because the cycle is advanced by a conditional
# write on this item. That conditional write is the single-flight lock: it is the
# one place two overlapping cycles become representable, so it is the one place
# they can be refused. A double-fired cron tick loses the condition and does
# nothing, rather than opening a second cycle against the same budget.
resource "aws_dynamodb_table" "fleet_config" {
  name         = "${local.table_prefix}-fleet_config"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "run_id"
  range_key    = "entity"

  attribute {
    name = "run_id"
    type = "S"
  }

  attribute {
    name = "entity"
    type = "S"
  }

  # Small, mutable, and the one table where a bad write is not recoverable by
  # re-running a job: `desired_version` is what the fleet is actually running.
  point_in_time_recovery {
    enabled = true
  }

  deletion_protection_enabled = true
}

# Append-only. Every label purchase, and later every promotion and rejection,
# recorded with its reason.
#
# The sort key is a composed `event` string rather than a timestamp, and the
# choice is load-bearing: the oracle's idempotency key is
# `(run_id, cycle, sha256(sorted_image_ids))`, so a conditional put on a sort key
# derived from exactly those three fields makes one write do three jobs at once --
# record the charge, refuse the double charge, and cache the response a retry
# replays. A timestamp sort key cannot do that, because a retry has a different
# timestamp and would land beside the original as a second charge.
#
# The grammar of that string lives in `conventions` rather than here, for the
# reason the whole module exists: a key spelled at two call sites drifts at one of
# them. What is fixed here, permanently, is that the sort key is a string.
# Ordering by time is an attribute and a scan over a partition that holds a few
# thousand items.
resource "aws_dynamodb_table" "audit_log" {
  name         = "${local.table_prefix}-audit_log"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "run_id"
  range_key    = "event"

  attribute {
    name = "run_id"
    type = "S"
  }

  attribute {
    name = "event"
    type = "S"
  }

  # The audit trail is evidence rather than state: it is the answer to "what did
  # this run spend and why", and unlike the labels it cannot be recomputed from
  # anything.
  point_in_time_recovery {
    enabled = true
  }

  deletion_protection_enabled = true
}

# Single-flight, so two cycles in one run cannot overlap. Taken at cycle start,
# released at the end.
#
# The lease expiry is the part worth getting right. A cycle that dies between
# taking the lock and releasing it would otherwise wedge the run permanently, and
# the failure looks like nothing happening -- the worst shape a failure can have
# in a scheduled system. So the lock item carries `expires_at` and the acquire
# condition is "no holder, or the holder's lease has passed", which means
# correctness never depends on the TTL firing. DynamoDB's TTL deletion is
# best-effort and can lag by up to 48 hours; it is enabled here to sweep expired
# items, not to release locks.
resource "aws_dynamodb_table" "run_locks" {
  name         = "${local.table_prefix}-run_locks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "run_id"
  range_key    = "lock"

  attribute {
    name = "run_id"
    type = "S"
  }

  attribute {
    name = "lock"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }

  # No backup and no deletion protection: a lock is the one item here whose
  # correct value after a disaster is "absent".
}
