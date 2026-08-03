use crate::{Matrix, MatrixScalar, Real};

/// Lower-triangular view of a fixed-size square matrix.
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
        for column in 0..P {
            for row in 0..D {
                let mut value = rhs[(row, column)];
                for previous in 0..row {
                    value = value - self.matrix[(row, previous)] * rhs[(previous, column)];
                }
                rhs[(row, column)] = value / self.matrix[(row, row)];
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
        for column in 0..P {
            for row in (0..D).rev() {
                let mut value = rhs[(row, column)];
                for next in (row + 1)..D {
                    value = value - self.matrix[(row, next)] * rhs[(next, column)];
                }
                rhs[(row, column)] = value / self.matrix[(row, row)];
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

    #[test]
    fn lower_view_solves_and_multiplies() {
        let lower = matrix![2.0_f64, 0.0, 0.0; 3.0, 4.0, 0.0; -1.0, 2.0, 5.0];
        let rhs = Matrix::<3, 2, f64>::from_rows([[2.0, 4.0], [11.0, 6.0], [7.0, 13.0]]);
        let solution = lower.lower_triangular().solve(&rhs);
        let mut reconstructed = Matrix::<3, 2, f64>::zeros();
        lower
            .lower_triangular()
            .mul_into(&solution, &mut reconstructed);
        assert_relative_eq!(reconstructed, rhs, epsilon = 1e-12, max_relative = 1e-12);
    }

    #[test]
    fn upper_view_solves_in_place() {
        let upper = matrix![2.0_f64, -1.0, 3.0; 0.0, 4.0, 2.0; 0.0, 0.0, 5.0];
        let rhs = Matrix::<3, 2, f64>::from_rows([[7.0, 4.0], [12.0, 8.0], [10.0, 15.0]]);
        let expected = upper.upper_triangular().solve(&rhs);
        let mut actual = rhs;
        upper.upper_triangular().solve_in_place(&mut actual);
        assert_relative_eq!(actual, expected, epsilon = 1e-12, max_relative = 1e-12);
    }
}
