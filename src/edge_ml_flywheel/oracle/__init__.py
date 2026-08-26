"""The oracle: the only route ground truth takes into the training set.

It reads labels straight out of `raw/labels/`. There is no per-run copy and no
intermediate table -- the storage layout says nothing about cohorts, so what
separates a purchasable `pool` label from the `eval` labels beside it is the
assignments parquet the partitioner froze, and `cohorts` is where that
separation is enforced.

**The guarantee is a gate, and the gate produces the path.** `check_purchasable`
runs while a request is still a list of image IDs, before any key exists; the
key is then built by `labels.label_key`, which routes through the same gate. So
a refused image is not merely unsold, it is unread -- and there is no function
in this package that returns a key under the split `eval` is drawn from.

That places the whole eval guarantee in one module with one set of tests, rather
than in an IAM policy that would have to name a prefix the layout does not have.
The bucket policy still denies `raw/labels/` to everything but ingest and the
oracle, so the budget is not bypassable from outside; what it cannot express is
the pool/eval line, and this is what does.

Three cohorts are refused for three different reasons, and refusals say which:
`bootstrap` is already owned and paying for it again is wasted budget, `eval` is
the ruler and buying it is contamination, and `reserve` is deliberately inert.
An image the partition never assigned is refused separately again -- it means the
selector is not working from this partition at all.
"""
