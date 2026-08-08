//! Fixed-capacity block sparse storage and factorization.
//!
//! Block sparse matrices store dense compile-time-sized blocks in CSC or CSR
//! structure. The block pattern stays compact while each block retains dense,
//! cache-friendly arithmetic. All storage is inline and bounded by the type's
//! capacity parameter.
//!
//! ```
//! use stack_algebra::{Matrix, StaticBlockCscMatrix};
//!
//! type Blocks = StaticBlockCscMatrix<1, 1, 2, 2, 3, f64>;
//! let a = Blocks::from_pattern(
//!     &[Matrix::from_rows([[4.0]]), Matrix::from_rows([[1.0]]), Matrix::from_rows([[3.0]])],
//!     &[0, 1, 1],
//!     &[0, 2, 3],
//! ).unwrap();
//! let mut y = [0.0; 2];
//! a.matvec_into(&[1.0, 2.0], &mut y).unwrap();
//! assert_eq!(y, [4.0, 7.0]);
//! ```
//!
//! Native block Cholesky and LDLᵀ require square blocks and square block
//! grids. For rectangular or unsupported block layouts, use
//! [`StaticBlockCscMatrix::to_scalar_csc`] as an explicit bounded adapter.

mod cholesky;
mod csr;
mod detail;
mod ldlt;
mod storage;
mod symbolic;

pub use cholesky::StaticBlockCscCholesky;
pub use csr::StaticBlockCsrMatrix;
pub use ldlt::StaticBlockCscLdlt;
pub use storage::StaticBlockCscMatrix;
pub use symbolic::{StaticBlockCscCholeskyPattern, StaticBlockCscLdltPattern};
