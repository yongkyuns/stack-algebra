use crate::DecompositionError;

/// Errors returned while constructing or updating a fixed-capacity CSC matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CscError {
    /// The supplied value, index, or pointer arrays have incompatible lengths.
    LengthMismatch,
    /// A column pointer is out of range or is not monotonic.
    InvalidColumnPointers,
    /// A row index is outside the matrix dimensions or is not strictly sorted
    /// within its column.
    InvalidRowIndices,
    /// An operation requires more stored entries than the fixed capacity.
    CapacityExceeded {
        /// Number of entries required to complete the operation.
        required: usize,
        /// Number of entries available in the fixed-capacity storage.
        capacity: usize,
    },
    /// The requested row/column is outside the matrix dimensions.
    IndexOutOfBounds,
    /// The requested entry is not present in the sparse pattern.
    EntryNotFound,
    /// A permutation has an out-of-range or repeated index.
    InvalidPermutation,
}

/// Errors returned while analyzing or numerically factoring a sparse matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseCholeskyError {
    /// The sparse factor requires more entries than its fixed capacity.
    CapacityExceeded {
        /// Number of factor entries required by symbolic analysis.
        required: usize,
        /// Number of factor entries available in the fixed-capacity storage.
        capacity: usize,
    },
    /// A diagonal pivot was not strictly positive.
    NotPositiveDefinite,
    /// A diagonal LDLᵀ pivot was zero.
    ZeroPivot,
    /// A non-finite value was encountered during factorization.
    NonFinite,
    /// The input contains an unmatched or numerically inconsistent symmetric
    /// counterpart.
    NonSymmetric,
    /// The numeric matrix contains a structural entry absent from the
    /// analyzed factor pattern.
    PatternMismatch,
    /// The underlying CSC pattern operation failed validation.
    Csc(CscError),
}

impl From<CscError> for SparseCholeskyError {
    #[inline]
    fn from(error: CscError) -> Self {
        match error {
            CscError::CapacityExceeded { required, capacity } => {
                Self::CapacityExceeded { required, capacity }
            }
            other => Self::Csc(other),
        }
    }
}

pub(crate) fn map_ldlt_error(error: DecompositionError) -> SparseCholeskyError {
    match error {
        DecompositionError::NonFinite => SparseCholeskyError::NonFinite,
        DecompositionError::ZeroPivot
        | DecompositionError::Singular
        | DecompositionError::NotPositiveDefinite
        | DecompositionError::NotSymmetric
        | DecompositionError::NoConvergence
        | DecompositionError::InvalidView => SparseCholeskyError::ZeroPivot,
    }
}
