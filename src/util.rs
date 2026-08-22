//! Small cross-cutting helpers that do not belong to a larger subsystem.

use crate::kernels::MatrixScalar;
use crate::ops::matmul_view_into;
use crate::view::StridedMap;
use crate::Matrix;

impl<const M: usize, const N: usize, T> StridedMap<'_, M, N, T>
where
    T: MatrixScalar,
{
    /// Multiplies two column-major strided views without repacking padded columns.
    ///
    /// When both views have unit inner stride, this streams their contiguous
    /// columns directly even when `outer_stride` is larger than the logical row
    /// count. This is the common leading-dimension layout used by borrowed
    /// submatrices and padded external buffers.
    ///
    /// Exact contiguous column-major inputs are delegated to [`Self::mul_into`]
    /// so they retain the target-specific owned-matrix kernel. If either input
    /// has a non-unit inner stride, this method falls back to the generic view
    /// multiplication path; arbitrary-stride semantics therefore remain
    /// unchanged.
    #[inline]
    pub fn mul_leading_dimension_into<const P: usize>(
        &self,
        rhs: &StridedMap<'_, N, P, T>,
        output: &mut Matrix<M, P, T>,
    ) {
        let lhs_unit_inner = M <= 1 || self.inner_stride() == 1;
        let rhs_unit_inner = N <= 1 || rhs.inner_stride() == 1;

        if !lhs_unit_inner || !rhs_unit_inner {
            matmul_view_into(self, rhs, output);
            return;
        }

        let lhs_exact = N <= 1 || self.outer_stride() == M;
        let rhs_exact = P <= 1 || rhs.outer_stride() == N;
        if lhs_exact && rhs_exact {
            self.mul_into(rhs, output);
            return;
        }

        let lhs_data = self.as_slice();
        let rhs_data = rhs.as_slice();
        let lhs_outer = self.outer_stride();
        let rhs_outer = rhs.outer_stride();
        let output_data = output.as_mut_slice();

        for value in output_data.iter_mut() {
            *value = T::zero();
        }

        for column in 0..P {
            let output_start = column * M;
            for shared in 0..N {
                let rhs_value = rhs_data[column * rhs_outer + shared];
                let lhs_start = shared * lhs_outer;
                for row in 0..M {
                    let output_index = output_start + row;
                    output_data[output_index] = T::mul_add(
                        lhs_data[lhs_start + row],
                        rhs_value,
                        output_data[output_index],
                    );
                }
            }
        }
    }
}
