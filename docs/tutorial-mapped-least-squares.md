# Walkthrough: fit a line from a borrowed buffer

Fit `y = a*x + b` to five observations. You will describe a matrix using
caller-owned storage, solve for slope and intercept with column-pivoted QR,
and evaluate predictions without constructing a separate owned design matrix.
The goal is to learn the existing view and solver APIs, not to build a
regression framework.

Start with [the shared setup](tutorials.md#before-you-start). Basic Rust
references and matrix multiplication are sufficient; you do not need to
implement QR yourself. [Open the example source](https://github.com/yongkyuns/stack-algebra/blob/main/examples/mapped_least_squares.rs)
or use the [complete listing](#complete-runnable-example) below. The excerpts
and full listing are included from the actual example at documentation-build
time. Excerpts are parts of that program, not independent programs to join.

From the repository root:

```sh
cargo run --example mapped_least_squares --no-default-features
```

This is a host executable with `std` printing and a `no_std` library
configuration. Run it through Cargo locally, not the browser playground.

## 1. Turn the model into a small linear system

For each observation, `y_i ~= a*x_i + b`. The unknown coefficient vector is
`c = [a, b]^T`. Put one observation in each row of a design matrix `A`:

```text
A: 5x2                 c: 2x1         y: 5x1
   [0  1]                 [a]           [1.1]
   [1  1]                 [b]           [2.9]
   [2  1]                               [5.2]
   [3  1]                               [6.8]
   [4  1]                               [9.0]

Find c that minimizes ||y - A*c||^2.
```

There are five observations and only two coefficients. The fixed observations
are slightly perturbed rather than exactly on a line, so an exact solution to
all five equations is not expected. The fit treats `x` as known and gives
every observation equal weight. It does not model uncertainty in `x`, apply
robust outlier rejection, or generate random samples.

## 2. Lay out and borrow the design buffer

Each conceptual row is `[x_i, 1]`, but `Map` expects contiguous
**column-major** data: all five `x` values first, then all five ones.

```rust,noplayground
{{#include ../examples/mapped_least_squares.rs:12:16}}
```

**Layout checkpoint:** `A[(row, column)]` reads
`DESIGN_STORAGE[column * 5 + row]`. For instance, `A[(3, 0)]` is `3.0`,
and `A[(3, 1)]` is `1.0`. Interleaving `[x_0, 1, x_1, 1, ...]` would instead
be a row-major layout and would represent the wrong matrix through this map.

Create the view and observation vector:

```rust,noplayground
{{#include ../examples/mapped_least_squares.rs:31:32}}
```

[`Map::from_slice`](api/stack_algebra/struct.Map.html#method.from_slice)
checks the buffer length and borrows its storage. `Map<5, 2, f64>` describes
five rows and two columns without owning ten new matrix entries. In the
helper signature, the lifetime in `Map<'_, 5, 2, f64>` ties the view to the
borrowed input; that input must remain alive while the view is used.

[`Matrix::from_columns`](api/stack_algebra/struct.Matrix.html#method.from_columns)
constructs one column containing the five observations. The small owned
observation vector is intentional. The example avoids an owned **design**
copy; it is not claiming that every value in the computation is a view.

For genuinely strided input, choose an appropriate
[`StridedMap`](api/stack_algebra/struct.StridedMap.html) rather than pretending
the same buffer is contiguous column-major.

## 3. Factor the borrowed design and solve

The fitting helper is short enough to read in full:

```rust,noplayground
{{#include ../examples/mapped_least_squares.rs:18:27}}
```

[`ColPivHouseholderQr::try_decompose_view`](api/stack_algebra/struct.ColPivHouseholderQr.html#method.try_decompose_view)
accepts the map directly. QR reads the design into the factor object's own
inline storage and computes its factors there. The caller's buffer is not
modified. This avoids a separate owning input matrix, but **QR still has
factor storage**: it is not a workspace-free or in-place factorization of the
borrowed slice.

[`try_solve_least_squares`](api/stack_algebra/struct.ColPivHouseholderQr.html#method.try_solve_least_squares)
uses those factors to solve the least-squares problem. The example does not
form an inverse or explicitly build `A^T*A`. Column pivoting can rearrange
columns internally; the returned coefficients are restored to the original
column order, so entry zero is the slope and entry one is the intercept.

The return type is `Result<Matrix<2, 1, f64>, DecompositionError>`. The `?`
propagates a decomposition error immediately; the final expression returns the
solve result. This helper is part of the example, not a new library API.

The executable calls it with its fixed, valid inputs:

```rust,noplayground
{{#include ../examples/mapped_least_squares.rs:33}}
```

`expect` is appropriate here only because the teaching data are known. A
caller accepting arbitrary inputs should handle the returned error instead
of assuming a solution always exists.

## 4. Check what the coefficients mean

For these samples, the coefficients are approximately:

```text
a = 1.970
b = 1.060
fitted_y = 1.970*x + 1.060
```

An independent scalar calculation provides a useful checkpoint without
repeating QR:

```text
mean_x = 2                 mean_y = 5
sum((x_i - mean_x)^2)                    = 10
sum((x_i - mean_x)*(y_i - mean_y))        = 19.7
slope     = 19.7 / 10                    = 1.97
intercept = mean_y - slope * mean_x      = 1.06
```

These are reference calculations for this simple model, not another solver
inside the executable. The separate integration tests use centered scalar
regression to check the library result.

## 5. Evaluate predictions into an output vector

Use the same borrowed design to calculate `A*c`:

```rust,noplayground
{{#include ../examples/mapped_least_squares.rs:35:38}}
```

[`Map::matvec_into`](api/stack_algebra/struct.Map.html#method.matvec_into)
reads the design and coefficient vector and writes all five predictions into
`fitted`. The caller explicitly provides that output. There is no conversion
of `design` into an owned matrix for this multiplication.

The subtraction produces `residuals = observed - fitted`. A positive residual
means the observation is above the fitted line; a negative residual means it
is below. For example, at `x = 2`, the fitted value is `5.0` and the observation
is `5.2`, giving residual `+0.2`.

**Checkpoint for the full residual vector:**

```text
fitted_y = [1.06, 3.03, 5.00, 6.97, 8.94]
residual = [0.04, -0.13, 0.20, -0.17, 0.06]
sum(residual_i^2) = 0.091
||residual||     = sqrt(0.091) ~= 0.301662
```

[`Matrix::norm`](api/stack_algebra/struct.Matrix.html#method.norm) gives the
Euclidean norm for this column vector. It is **not** a mean absolute error or
root-mean-square error; the latter would also divide the squared sum by the
number of samples before taking the square root.

At the least-squares solution, `A^T*residual` is approximately zero. For the
written decimal data, both `sum(residual_i)` and `sum(x_i*residual_i)` are zero
in exact arithmetic. The tests check these relationships with tolerances.
A nonzero residual norm is normal: the solution minimizes it, rather than
forcing all observations onto a perfect line.

## 6. Compare the complete output

The program prints the coefficients, then each observation and its prediction:

```text
slope = 1.970, intercept = 1.060
x observed_y fitted_y residual
0.0 1.100 1.060 0.040
1.0 2.900 3.030 -0.130
2.0 5.200 5.000 0.200
3.0 6.800 6.970 -0.170
4.0 9.000 8.940 0.060
residual norm = 0.302
```

The values are rounded for display. Use numerical tolerances rather than
requiring identical floating-point representations when checking a solution.
The sample fit is a teaching example, not a performance or accuracy claim
about a larger application.

## 7. Explore a success case and a failure case

For an exact-line experiment, keep the design and change observations to
`[1.0, 3.0, 5.0, 7.0, 9.0]`. You should recover slope `2` and intercept `1`,
with residuals close to zero in floating-point arithmetic.

For a rank-deficiency experiment, restore the observations and change the
**first five design entries** to `2.0`, leaving the ones column unchanged.
Every row then says `y_i ~= 2*a + b`. The data constrain only that combination,
not slope and intercept separately. The helper returns
[`DecompositionError::Singular`](api/stack_algebra/enum.DecompositionError.html#variant.Singular);
the executable's final `expect` therefore panics instead of printing a fit.

A design or observation containing NaN or either infinity returns
[`DecompositionError::NonFinite`](api/stack_algebra/enum.DecompositionError.html#variant.NonFinite).
Finite input alone is not a success guarantee: the design must also have
full column rank at the solver's numerical threshold. This walkthrough does
not promise reliable coefficients for arbitrarily ill-conditioned data.

Restore the original input constants before checking the documented output or
running the unchanged reference tests. The helper assumes the second column
contains ones for a line fit; it is not an input-schema validator for a general
regression model.

## 8. Run the focused correctness checks

```sh
cargo test --no-default-features --test mapped_least_squares
cargo test --features std --test mapped_least_squares
cargo test --release --no-default-features --test mapped_least_squares
```

The [six integration tests](https://github.com/yongkyuns/stack-algebra/blob/main/tests/mapped_least_squares.rs)
import the actual helper. They check independent centered regression,
hand-calculated predictions and residuals including orthogonality, an exact
line with reordered samples, pointer identity and input preservation for the
borrowed map, rank deficiency, and non-finite inputs in both the design and
observations. The checks are kept out of the small runnable example.

## Complete runnable example

This is the actual `examples/mapped_least_squares.rs` source included during
the guide build. Run it using the Cargo command above. `pub(crate)` allows the
integration tests to call the helper within their crate, while
`#[cfg(not(test))]` excludes the printing entry point when they import it.
Neither adds a public fitting API to `stack-algebra`.

```rust,noplayground
{{#include ../examples/mapped_least_squares.rs}}
```

Continue with the [two-state Kalman walkthrough](tutorial-kalman-1d.md),
consult [solver choices and reuse](api-usage.md), or return to the
[tutorial overview](tutorials.md).
