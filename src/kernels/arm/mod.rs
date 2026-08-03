use super::{MatrixScalar, ReductionScalar};

mod neon;

use neon::{NeonMatmul, NeonReduction};

impl MatrixScalar for f32 {
    type Matmul = NeonMatmul;
}

impl MatrixScalar for f64 {
    type Matmul = NeonMatmul;
}

impl ReductionScalar for f32 {
    type Reduction = NeonReduction;
}

impl ReductionScalar for f64 {
    type Reduction = NeonReduction;
}
