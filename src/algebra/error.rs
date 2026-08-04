/// Failure reported by a fixed-size matrix decomposition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecompositionError {
    /// The input or an intermediate value is not finite.
    NonFinite,
    /// The matrix is not positive-definite.
    NotPositiveDefinite,
    /// The matrix is not symmetric within numerical tolerance.
    NotSymmetric,
    /// The matrix is singular.
    Singular,
    /// A factorization pivot is zero.
    ZeroPivot,
    /// The iterative factorization did not converge within its fixed budget.
    NoConvergence,
    /// A matrix view did not provide one of its compile-time coordinates.
    InvalidView,
}
