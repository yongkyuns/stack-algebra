# Permutation optimization protocol

Issue #53 targets the measured cost of #51, not a rollback of its contracts.
The `Sparse permutation optimization` workflow first profiles unchanged repaired
source at `d1cd847ec4ef307dab7a91518d2bf1a2a11f90dc` in an isolated checkout.
Temporary wrappers expose the bounds scan and complete pattern installation to
a separate profiling binary. They are never inserted into repository library
source, package inputs, or the whole-call comparison builds.

Profiles compare the existing source-index scan, a proposed constant-time length
comparison, direct full-pattern copying, and exact-equality-guarded copying for
matching destinations. The scalar comparison alone proves no safety property;
any cached bound must be established and tested through construction/rebuild.
Component measurements are not additive decompositions of full-call timings.

The second phase runs the unchanged #52 public fixtures against the repaired
baseline, not the earlier implementation with broken arbitrary destinations.
`compare.py --before COMMIT` selects the baseline and records both exact commits
in the report and provenance; the historical default is retained for #52's
workflow. Twelve rounds alternate revision order, with common compiler flags,
one dependency lock, separate builds and CPU affinity. All 384 cases and their
correctness checks remain unchanged, including nonidentity ordered pipelines.

No universal non-regression or MCU-latency claim follows from these warm-cache
hosted-runner fixtures. Review individual case medians and raw rounds, not only
overall averages. Archive hashes, source/compiler/lock/fixture provenance and
regenerated summaries must be verified before accepting an optimization.
