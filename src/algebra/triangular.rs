use crate::{Matrix, MatrixScalar, Real};

/// Lower-triangular view of a fixed-size square matrix.
///
/// The view borrows the matrix and ignores entries above the diagonal. Use
/// [`Matrix::lower_triangular`] to create one without copying data.
///
/// # Examples
///
/// ```
/// use stack_algebra::matrix;
///
/// let lower = matrix![2.0_f64, 0.0; 1.0, 3.0];
/// let rhs = matrix![2.0_f64; 7.0];
/// let x = lower.lower_triangular().solve(&rhs);
/// assert!((lower.lower_triangular().matrix() * x - rhs).norm() < 1.0e-12);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct LowerTriangular<'a, const D: usize, T> {
    matrix: &'a Matrix<D, D, T>,
}

impl<'a, const D: usize, T: Real + MatrixScalar> LowerTriangular<'a, D, T> {
    /// Solves `L * X = B` and returns an owned solution.
    #[inline]
    pub fn solve<const P: usize>(&self, rhs: &Matrix<D, P, T>) -> Matrix<D, P, T> {
        let mut solution = *rhs;
        self.solve_in_place(&mut solution);
        solution
    }

    /// Solves `L * X = B` into a caller-provided output matrix.
    #[inline]
    pub fn solve_into<const P: usize>(&self, rhs: &Matrix<D, P, T>, output: &mut Matrix<D, P, T>) {
        *output = *rhs;
        self.solve_in_place(output);
    }

    /// Solves `L * X = B` in place.
    #[inline]
    pub fn solve_in_place<const P: usize>(&self, rhs: &mut Matrix<D, P, T>) {
        if P == 1 {
            for row in 0..D {
                let mut value = rhs[(row, 0)];
                for previous in 0..row {
                    value = value - self.matrix[(row, previous)] * rhs[(previous, 0)];
                }
                rhs[(row, 0)] = value / self.matrix[(row, row)];
            }
            return;
        }

        for row in 0..D {
            let diagonal = self.matrix[(row, row)];
            for column in 0..P {
                rhs[(row, column)] = rhs[(row, column)] / diagonal;
            }

            for next in (row + 1)..D {
                let factor = self.matrix[(next, row)];
                for column in 0..P {
                    let solved = rhs[(row, column)];
                    let value = rhs[(next, column)] - factor * solved;
                    rhs[(next, column)] = value;
                }
            }
        }
    }

    /// Computes `L * X` into a caller-provided output matrix.
    #[inline]
    pub fn mul_into<const P: usize>(&self, rhs: &Matrix<D, P, T>, output: &mut Matrix<D, P, T>) {
        for column in 0..P {
            for row in 0..D {
                let mut value = T::zero();
                for shared in 0..=row {
                    value = value + self.matrix[(row, shared)] * rhs[(shared, column)];
                }
                output[(row, column)] = value;
            }
        }
    }

    /// Returns the underlying matrix.
    #[inline]
    pub fn matrix(&self) -> &'a Matrix<D, D, T> {
        self.matrix
    }
}

/// Upper-triangular view of a fixed-size square matrix.
///
/// The view borrows the matrix and ignores entries below the diagonal. Use
/// [`Matrix::upper_triangular`] to create one without copying data.
#[derive(Clone, Copy, Debug)]
pub struct UpperTriangular<'a, const D: usize, T> {
    matrix: &'a Matrix<D, D, T>,
}

impl<'a, const D: usize, T: Real + MatrixScalar> UpperTriangular<'a, D, T> {
    /// Solves `U * X = B` and returns an owned solution.
    #[inline]
    pub fn solve<const P: usize>(&self, rhs: &Matrix<D, P, T>) -> Matrix<D, P, T> {
        let mut solution = *rhs;
        self.solve_in_place(&mut solution);
        solution
    }

    /// Solves `U * X = B` into a caller-provided output matrix.
    #[inline]
    pub fn solve_into<const P: usize>(&self, rhs: &Matrix<D, P, T>, output: &mut Matrix<D, P, T>) {
        *output = *rhs;
        self.solve_in_place(output);
    }

    /// Solves `U * X = B` in place.
    #[inline]
    pub fn solve_in_place<const P: usize>(&self, rhs: &mut Matrix<D, P, T>) {
        for row in (0..D).rev() {
            let diagonal = self.matrix[(row, row)];
            for column in 0..P {
                rhs[(row, column)] = rhs[(row, column)] / diagonal;
            }

            for previous in 0..row {
                let factor = self.matrix[(previous, row)];
                for column in 0..P {
                    let solved = rhs[(row, column)];
                    let value = rhs[(previous, column)] - factor * solved;
                    rhs[(previous, column)] = value;
                }
            }
        }
    }

    /// Computes `U * X` into a caller-provided output matrix.
    #[inline]
    pub fn mul_into<const P: usize>(&self, rhs: &Matrix<D, P, T>, output: &mut Matrix<D, P, T>) {
        for column in 0..P {
            for row in 0..D {
                let mut value = T::zero();
                for shared in row..D {
                    value = value + self.matrix[(row, shared)] * rhs[(shared, column)];
                }
                output[(row, column)] = value;
            }
        }
    }

    /// Returns the underlying matrix.
    #[inline]
    pub fn matrix(&self) -> &'a Matrix<D, D, T> {
        self.matrix
    }
}

impl<const D: usize, T: Real + MatrixScalar> Matrix<D, D, T> {
    /// Creates a lower-triangular view of this matrix.
    #[inline]
    pub fn lower_triangular(&self) -> LowerTriangular<'_, D, T> {
        LowerTriangular { matrix: self }
    }

    /// Creates an upper-triangular view of this matrix.
    #[inline]
    pub fn upper_triangular(&self) -> UpperTriangular<'_, D, T> {
        UpperTriangular { matrix: self }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{matrix, Matrix};

    fn assert_lower_reconstructs<const P: usize>(rhs: Matrix<3, P, f64>) {
        let lower = matrix![2.0_f64, 0.0, 0.0; 3.0, 4.0, 0.0; -1.0, 2.0, 5.0];
        let solution = lower.lower_triangular().solve(&rhs);
        let mut reconstructed = Matrix::<3, P, f64>::zeros();
        lower
            .lower_triangular()
            .mul_into(&solution, &mut reconstructed);
        assert_relative_eq!(reconstructed, rhs, epsilon = 1e-12, max_relative = 1e-12);
    }

    fn assert_upper_reconstructs<const P: usize>(rhs: Matrix<3, P, f64>) {
        let upper = matrix![2.0_f64, -1.0, 3.0; 0.0, 4.0, 2.0; 0.0, 0.0, 5.0];
        let mut solution = rhs;
        upper.upper_triangular().solve_in_place(&mut solution);

        let mut reconstructed = Matrix::<3, P, f64>::zeros();
        upper
            .upper_triangular()
            .mul_into(&solution, &mut reconstructed);
        assert_relative_eq!(reconstructed, rhs, epsilon = 1e-12, max_relative = 1e-12);
    }

    #[test]
    fn lower_view_solves_and_multiplies_single_rhs() {
        assert_lower_reconstructs(Matrix::<3, 1, f64>::from_rows([[2.0], [11.0], [7.0]]));
    }

    #[test]
    fn lower_view_solves_and_multiplies_multiple_rhs() {
        assert_lower_reconstructs(Matrix::<3, 4, f64>::from_rows([
            [2.0, 4.0, -1.0, 3.0],
            [11.0, 6.0, 5.0, -2.0],
            [7.0, 13.0, 9.0, 8.0],
        ]));
    }

    #[test]
    fn upper_view_solves_in_place_single_rhs() {
        assert_upper_reconstructs(Matrix::<3, 1, f64>::from_rows([[7.0], [12.0], [10.0]]));
    }

    #[test]
    fn upper_view_solves_in_place_multiple_rhs() {
        assert_upper_reconstructs(Matrix::<3, 4, f64>::from_rows([
            [7.0, 4.0, -2.0, 9.0],
            [12.0, 8.0, 3.0, -1.0],
            [10.0, 15.0, 5.0, 20.0],
        ]));
    }
}
