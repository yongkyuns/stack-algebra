# Solver invariant qualification

The `0.3` release gate requires every public solver family to have
invariant-based numerical evidence. Differential agreement with Eigen remains
useful secondary evidence, but it is not the specification: a solver must also
satisfy its own reconstruction, residual, orthogonality, rank, pivot, or
failure contract.

This page is the traceability record for that gate. Pattern and ordering types
are listed with the solver that consumes them rather than treated as numerical
solvers themselves.

## Dense solvers

| Public solver | Primary invariant evidence | Additional coverage |
| --- | --- | --- |
| `LowerTriangular` | `src/algebra/triangular.rs::lower_view_solves_and_multiplies` solves two RHS columns and verifies `L * X = B` through the independent triangular multiply path. | The public doctest also checks the dense residual. |
| `UpperTriangular` | `src/algebra/triangular.rs::upper_view_solves_in_place_and_reconstructs_rhs` solves two RHS columns in place and verifies `U * X = B` through the independent triangular multiply path. | `solve`, `solve_into`, and `solve_in_place` share the same substitution core. |
| `Cholesky` | `src/algebra/cholesky.rs::generated_f64_reconstruction_and_solve_contract` and `generated_f32_reconstruction_and_solve_contract` verify `L * Lᵀ = A` and solve residuals for dimensions 1, 2, 3, 6, and 15 over multiple scales. | View decomposition/recompute, inverse, typed non-finite/not-positive-definite failures, and the blocked 64x64 path have dedicated tests. |
| `PartialPivLu` | `tests/numerical_contracts.rs::f64_dense_solver_contracts_across_scales` verifies `P * A = L * U` and multi-RHS solve residuals; `f32_dense_solver_contracts` verifies solve residuals. | `dense_solver_view_contracts_match_owned_input` exercises a row-major `StridedMap`. |
| `Ldlt` | `tests/numerical_contracts.rs::f64_dense_solver_contracts_across_scales` verifies `P * A * Pᵀ = L * D * Lᵀ` and multi-RHS solve residuals; the f32 contract verifies solve residuals. | View contracts plus decomposition-specific pivot/failure tests cover the Bunch–Kaufman path. |
| `HouseholderQr` | `tests/numerical_contracts.rs::f64_dense_solver_contracts_across_scales` verifies `Q * R = A`, orthonormal `Q`, and least-squares residuals; the f32 contract verifies reconstruction/solve behavior. | The view contract exercises row-major strided input. |
| `ColPivHouseholderQr` | The same cross-solver suite verifies `Q * R = A * P`, rank, and least-squares residuals for f64 and solve/rank behavior for f32. | Rank-deficient and pivot semantics have decomposition-specific tests and Eigen differential coverage. |
| `Svd` | The cross-solver suite verifies `U * Σ * Vᵀ = A`, orthonormal factors, descending singular values, rank, and solve residuals across scales; f32 has reconstruction and solve checks. | The row-major view path is covered independently. |
| `SelfAdjointEigen` | The cross-solver suite verifies reconstruction, orthonormal eigenvectors, and ascending eigenvalues across scales for f64, plus f32 reconstruction. | The row-major view path and workspace/recompute paths have dedicated tests. |

`SelfAdjointLower`, `SelfAdjointUpper`, and `SelfAdjointView` are structured
matrix views rather than independent solvers; their validity and arithmetic
contracts are covered with the self-adjoint/view tests.

## Scalar sparse solvers

| Public solver | Primary invariant evidence | Additional coverage |
| --- | --- | --- |
| `StaticCscCholesky` | `tests/sparse_cholesky.rs::spd_factorization_solves_and_reconstructs_action` verifies the solve residual and lower-triangular factor structure. | f32 residuals, symbolic/numeric reuse, minimum-degree ordering, asymmetric/non-finite rejection, non-SPD rejection, and exact fill-capacity errors are covered in the same integration suite. |
| `StaticCscLdlt` | `tests/sparse_ldlt.rs::sparse_ldlt_solves_indefinite_system` verifies the residual for an indefinite system. | Numeric factor reuse, diagonal pivoting, scale-relative thresholds, ordering, zero-pivot errors, and 2x2-pivot fallback behavior are covered in the same suite. |
| `StaticCscLdltFactor` | The unified-factor tests verify residuals on the native sparse path, sparse diagonal-pivot path, dense fallback path, and after numeric recomputation changes representation. | A failed fallback recomputation is checked to preserve the previous valid factor. |

`StaticCscCholeskyPattern`, `StaticCscLdltPattern`, `StaticCscOrdering`, and
`StaticCscPermutation` describe reusable symbolic structure/order. Their
correctness is exercised by factor/recompute/ordering tests above rather than by
inventing a separate numerical-solver contract.

## Block-sparse solvers

| Public solver | Primary invariant evidence | Additional coverage |
| --- | --- | --- |
| `StaticBlockCscCholesky` | `tests/block_sparse.rs::block_csc_expands_and_factors_without_heap_storage` and `native_block_cholesky_handles_block_fill_in` verify native solutions against expanded scalar/dense systems and direct residuals. | Symbolic reuse, block fill-in, minimum-degree ordering, multi-RHS reuse, and capacity behavior are covered in the block-sparse suite. |
| `StaticBlockCscLdlt` | `tests/block_sparse.rs::symbolic_block_pattern_reuses_numeric_storage_for_cholesky_and_ldlt` verifies a dense multi-RHS residual after recomputation; additional native LDLT tests compare against scalar expansion. | Ordering, diagonal pivoting, local 2x2 Bunch–Kaufman pivots, cross-block dense fallback, extreme-scale local solves, and failure behavior have dedicated tests. |

The block CSC/CSR matrix types themselves also have matvec-vs-expanded-matrix
contracts, which protects the storage/action layer used by the factors.

## Release interpretation

The solver-invariant gate is satisfied when these tests pass on the release
commit. This does **not** mean every algorithm is suitable for every problem:

- Cholesky still requires positive-definite input.
- Dense LDLT implements its documented pivot model; fixed sparse LDLT has
  explicit bounded pivot/fallback semantics rather than claiming general sparse
  indefinite coverage.
- Fixed-capacity sparse and block-sparse solvers require the caller to choose
  adequate capacities.
- Numerical tolerances remain scalar-, scale-, algorithm-, and target-specific.

The API/semver gate, package/public-API snapshot, pinned-host benchmark evidence,
and physical embedded qualification are separate release gates and should not
be inferred from this solver table.
