"""Reporting: the run summary, and what later reads it.

A cross-cutting concern rather than a stage, for the reason the architecture
overview gives -- a cycle does not pass through it. Nothing here owns
infrastructure, produces an artifact a stage needs, or is allowed to decide
anything: it reduces what the stages already recorded into one document per run.

**Reporting cannot change a verdict.** Every number it publishes was written by
the job that measured it, and the reduction is lookup and a carry. So the one
document this concern owns is written once, when the run ends, like every
artifact it reads -- summarizing the same run twice cannot reach two answers
unless one of those artifacts changed.
"""
