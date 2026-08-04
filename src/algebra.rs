//! Fixed-size matrix decompositions.

mod cholesky;
mod eigen;
mod error;
mod ldlt;
mod lu;
mod qr;
mod self_adjoint;
mod svd;
mod triangular;

pub use cholesky::Cholesky;
pub use eigen::{SelfAdjointEigen, SelfAdjointEigenWorkspace};
pub use error::DecompositionError;
pub use ldlt::Ldlt;
pub use lu::PartialPivLu;
pub use qr::{ColPivHouseholderQr, HouseholderQr};
pub use self_adjoint::{SelfAdjointLower, SelfAdjointUpper, SelfAdjointView};
pub use svd::Svd;
pub use triangular::{LowerTriangular, UpperTriangular};
