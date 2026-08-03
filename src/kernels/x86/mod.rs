use super::{MatrixScalar, ReductionScalar};

#[cfg(target_feature = "avx2")]
mod avx2;
#[cfg(all(target_feature = "avx2", target_feature = "fma"))]
mod avx2_fma;
#[cfg(all(target_feature = "avx2", not(target_feature = "fma")))]
use avx2::{X86Avx2Matmul, X86Avx2Reduction};
#[cfg(all(target_feature = "avx2", target_feature = "fma"))]
use avx2_fma::{X86Avx2FmaMatmul as X86Avx2Matmul, X86Avx2FmaReduction as X86Avx2Reduction};

#[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
mod sse2;
#[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
use sse2::{X86Sse2Matmul, X86Sse2Reduction};

#[cfg(target_feature = "avx2")]
impl MatrixScalar for f32 {
    type Matmul = X86Avx2Matmul;
}

#[cfg(target_feature = "avx2")]
impl MatrixScalar for f64 {
    type Matmul = X86Avx2Matmul;
}

#[cfg(target_feature = "avx2")]
impl ReductionScalar for f32 {
    type Reduction = X86Avx2Reduction;
}

#[cfg(target_feature = "avx2")]
impl ReductionScalar for f64 {
    type Reduction = X86Avx2Reduction;
}

#[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
impl MatrixScalar for f32 {
    type Matmul = X86Sse2Matmul;
}

#[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
impl MatrixScalar for f64 {
    type Matmul = X86Sse2Matmul;
}

#[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
impl ReductionScalar for f32 {
    type Reduction = X86Sse2Reduction;
}

#[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
impl ReductionScalar for f64 {
    type Reduction = X86Sse2Reduction;
}
