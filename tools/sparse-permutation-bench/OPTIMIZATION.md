# Permutation optimization protocol

Issue #53 targets the measured cost of #51, not a rollback of its contracts.
The `Sparse permutation optimization` workflow first profiles unchanged repaired
source at `d1cd847ec4ef307dab7a91518d2bf1a2a11f90dc` in an isolated checkout.
Temporary wrappers expose the bounds scan and complete pattern installation to
a separate profiling binary. They are never inserted into repository library
source, package inputs, or the whole-call comparison builds.

Profiles compare the existing source-index scan, a proposed constant-time length
comparison, direct full-pattern copying, and exact-equality-guarded copying for
matching destinations. Component measurements are not additive decompositions
of full-call timings. The first verified profiles found guarded copying slower
on ARM, so the initial implementation leaves complete pattern copying unchanged.

## Cached source-bound invariant

`StaticCscPermutation` previously duplicated its pattern's active entry count in
a separate `usize`. That slot now holds the exact required source length: zero
for an empty map, otherwise one past the greatest referenced source index.
The count still comes from `pattern.nnz()`; no field or capacity array is added.
A private field's name and derived Debug presentation change, not public method
signatures. Tests compare the previous and current workspace size/alignment.

Construction visits canonical source indices in increasing order and updates
the bound only for stored lower entries. Consequently, upper entries before a
lower entry count toward its offset, while ignored trailing upper entries do
not inflate the bound. All fallible integer conversions run before the existing
workspace is changed. The second pass casts only previously validated indices;
sorting preserves the maximum. Rebuilds reset the bound, and new/new_into/default
initialize it to zero. Derived Clone/Copy carry the same established bound.

Checking `required_source_len <= matrix.values().len()` is exactly equivalent
to the previous all-indices bounds scan. It still runs before modifying either
destination field. Numeric reads remain safely indexed. Destination patterns,
active writes and untouched inactive backing values retain #51's behavior.
This does not validate source-coordinate changes: the original source-pattern
requirement remains documented and unchanged.

Independent coordinate-lookup tests cover all 32 canonical 2x2 source/ordering
pairs and all 3,072 canonical 3x3 pairs, including empty, full and upper-only
layouts. Each cached bound is compared with the old per-index predicate for
every possible source length through capacity. Both enumerations run in native
debug/release; Miri runs the 2x2 enumeration and other internal contract tests.
Public tests additionally cover caller-owned initialization/reinitialization,
rebuilds, Clone/Copy and largest-offset rejection after reverse sorting. The
existing package consumer automatically includes the expanded public suite.

## Whole-call comparisons

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
