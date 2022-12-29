#![allow(non_snake_case)]

use core::{
    // fmt,
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::{
    eye,
    num::{Abs, One, Zero},
    Matrix,
};

impl<const D: usize, T> Matrix<D, D, T>
where
    T: Abs
        + PartialOrd
        + Copy
        + Zero
        + One
        // + fmt::Display
        + Sum
        + Add<Output = T>
        + Neg<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Div<Output = T>,
{
    pub fn inv(&self) -> Option<Self> {
        let (mut L, mut U, P) = self.lu();
        if let (Some(L_inv), Some(U_inv)) = (
            Self::invert_lower_triangular(&mut L),
            Self::invert_upper_triangular(&mut U),
        ) {
            Some(U_inv * L_inv * P)
        } else {
            None
        }
    }

    pub fn det(&self) -> T {
        let (L, U, _) = self.lu();
        let mut det = T::one();
        for i in 0..D {
            det = det * L[(i, i)] * U[(i, i)];
        }
        if D % 2 != 0 {
            det = -det;
        }
        det
    }

    pub fn lu(&self) -> (Matrix<D, D, T>, Matrix<D, D, T>, Matrix<D, D, T>) {
        let mut P = eye!(D, T);
        let mut L = eye!(D, T);
        let mut U = *self;

        for d in 0..D {
            // Find row index of maximum absolute value equal to or below given diagonal
            let max_row = Self::find_max_row(&U, d);
            // Swap rows if a non-diagonal row is larger (i.e. partial pivot)
            Self::partial_pivot(&mut P, &mut L, &mut U, d, max_row);
            // Perform single step of gaussian-elimination
            Self::gauss_eliminate(&mut L, &mut U, d);
        }
        (L, U, P)
    }

    fn invert_upper_triangular(U: &mut Matrix<D, D, T>) -> Option<Matrix<D, D, T>> {
        let mut I = eye!(D, T);
        for i in (0..D).rev() {
            let diag = U[(i, i)];
            if diag == T::zero() {
                return None;
            }
            let coeff = T::one() / diag;

            // Make current diagonal identity and scale by same in the row of `I`
            U[(i, i)] = U[(i, i)] * coeff;
            for c in i..D {
                I[(i, c)] = I[(i, c)] * coeff;
            }

            // Perform gaussian elimination on upper rows of current diagonal
            for r in 0..i {
                let coeff = -U[(r, i)];
                U[(r, i)] = T::zero();
                for c in i..D {
                    I[(r, c)] = I[(r, c)] + coeff * I[(i, c)];
                }
            }
        }
        Some(I)
    }

    fn invert_lower_triangular(L: &mut Matrix<D, D, T>) -> Option<Matrix<D, D, T>> {
        let mut I = eye!(D, T);
        for i in 0..D {
            let diag = L[(i, i)];
            if diag != T::one() {
                return None;
            }
            for r in (i + 1)..D {
                let coeff = -L[(r, i)];
                L[(r, i)] = T::zero();
                for c in 0..(i + 1) {
                    I[(r, c)] = I[(r, c)] + coeff * I[(i, c)];
                }
            }
        }
        // println!("{}", L);
        // println!("{}", I);
        Some(I)
    }

    fn gauss_eliminate(L: &mut Matrix<D, D, T>, U: &mut Matrix<D, D, T>, diag: usize) {
        let d = diag;
        for r in (d + 1)..D {
            L[(r, d)] = U[(r, d)] / U[(d, d)];
            for c in 0..D {
                U[(r, c)] = U[(r, c)] - L[(r, d)] * U[(d, c)];
            }
        }
    }

    fn partial_pivot(
        P: &mut Matrix<D, D, T>,
        L: &mut Matrix<D, D, T>,
        U: &mut Matrix<D, D, T>,
        diag: usize,
        max_row: usize,
    ) {
        P.swap_rows(diag, max_row);
        U.swap_rows(diag, max_row);
        // Swap partial rows of L
        for c in 0..diag {
            let temp = L[(max_row, c)];
            L[(max_row, c)] = L[(diag, c)];
            L[(diag, c)] = temp;
        }
    }

    fn find_max_row(U: &Matrix<D, D, T>, diag: usize) -> usize {
        let mut max_row = diag;
        for r in diag..D {
            if U[(max_row, diag)].abs() < U[(r, diag)].abs() {
                max_row = r;
            }
        }
        max_row
    }
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    use super::*;
    use crate::matrix;

    #[test]
    fn LU_decomp() {
        let A = matrix![
            1.0, 3.0, 5.0;
            2.0, 4.0, 7.0;
            1.0, 1.0, 0.0;
        ];
        let (L, U, P) = A.lu();
        let L_exp = matrix![
            1.0,  0.0, 0.0;
            0.5,  1.0, 0.0;
            0.5, -1.0, 1.0;
        ];
        let U_exp = matrix![
            2.0, 4.0,  7.0;
            0.0, 1.0,  1.5;
            0.0, 0.0, -2.0;
        ];
        let P_exp = matrix![
            0.0, 1.0, 0.0;
            1.0, 0.0, 0.0;
            0.0, 0.0, 1.0;
        ];
        assert_eq!(L, L_exp);
        assert_eq!(U, U_exp);
        assert_eq!(P, P_exp);
    }

    #[test]
    fn determinant() {
        let A = matrix![
            6.0, 2.0, 3.0;
            1.0, 1.0, 1.0;
            0.0, 4.0, 9.0;
        ];
        assert_abs_diff_eq!(A.det(), 24.0, epsilon = 1e-10);

        let A = matrix![
            3.0, 7.0;
            1.0, -4.0;
        ];
        assert_abs_diff_eq!(A.det(), -19.0, epsilon = 1e-10);

        let A = matrix![
            1.0, 2.0;
            4.0, 8.0;
        ];
        assert_abs_diff_eq!(A.det(), 0.0, epsilon = 1e-10);

        let A = matrix![
            1.0, 1.0;
            2.0, 2.0;
        ];
        assert_abs_diff_eq!(A.det(), 0.0, epsilon = 1e-10);

        let A = matrix![
            1.0, 2.0, 3.0, 4.0;
            2.0, 5.0, 7.0, 3.0;
            4.0, 10.0, 14.0, 6.0;
            3.0, 4.0, 2.0, 7.0;
        ];
        assert_abs_diff_eq!(A.det(), 0.0, epsilon = 1e-10);

        let A = matrix![
            1.0, 2.0, 3.0, 4.0;
            2.0, 5.0, 7.0, 3.0;
            4.0, 10.0, 14.0, 6.0;
            3.0, 4.0, 2.0, 7.0;
        ];
        assert_abs_diff_eq!(A.det(), 0.0, epsilon = 1e-10);

        let A = matrix![
            11.0, 9.0, 24.0, 2.0;
            1.0, 5.0, 2.0, 6.0;
            3.0, 17.0, 18.0, 1.0;
            2.0, 5.0, 7.0, 1.0;
        ];
        assert_abs_diff_eq!(A.det(), -284.0, epsilon = 1e-10);

        let A = matrix![
              2.0, 3.0, 0.0, 9.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0;
              1.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 9.0, 2.0, 3.0, 1.0;
              1.0, 4.0, 0.0, 2.0, 8.0, 5.0, 0.0, 3.0, 6.0, 1.0, 9.0;
              0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0;
              2.0, 2.0, 4.0, 1.0, 1.0, 2.0, 1.0, 6.0, 9.0, 0.0, 7.0;
              0.0, 0.0, 0.0, 6.0, 0.0, 7.0, 0.0, 1.0, 0.0, 0.0, 0.0;
              2.0, 5.0, 0.0, 7.0, 0.0, 4.0, 6.0, 8.0, 5.0, 1.0, 3.0;
              0.0, 0.0, 0.0, 1.0, 0.0, 4.0, 0.0, 1.0, 0.0, 0.0, 0.0;
              0.0, 0.0, 0.0, 8.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0;
              2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0;
              2.0, 6.0, 0.0, 1.0, 0.0,30.0, 0.0, 2.0, 3.0, 2.0, 1.0;
        ];
        assert_abs_diff_eq!(A.det(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn upper_inverse() {
        let mut A = matrix![
            2.0, 4.0, 6.0;
            0.0,-1.0,-8.0;
            0.0, 0.0,96.0;
        ];
        let I = Matrix::invert_upper_triangular(&mut A).unwrap();
        let E = matrix![
            0.5, 2.0, 0.13541667;
            0.0,-1.0,-0.08333333;
            0.0, 0.0, 0.01041667;
        ];
        assert_relative_eq!(I, E, max_relative = 1e-6);
    }

    #[test]
    fn lower_inverse() {
        let mut A = matrix![
            1.0, 0.0, 0.0;
            8.0, 1.0, 0.0;
            4.0, 9.0, 1.0;
        ];
        let I = Matrix::invert_lower_triangular(&mut A).unwrap();
        let E = matrix![
            1.0, 0.0, 0.0;
            -8.0, 1.0, 0.0;
            68.0, -9.0, 1.0;
        ];
        assert_relative_eq!(I, E, max_relative = 1e-6);
    }

    #[test]
    fn inverse() {
        let A = matrix![
            6.0, 2.0, 3.0;
            1.0, 1.0, 1.0;
            0.0, 4.0, 9.0;
        ];
        let exp = matrix![
            0.20833333, -0.25, -0.04166667;
                -0.375,  2.25, -0.125;
            0.16666667,  -1.0,  0.16666667;
        ];
        assert_relative_eq!(A.inv().unwrap(), exp, max_relative = 1e-6);

        let A = matrix![
            11.0, 9.0, 24.0, 2.0;
            1.0, 5.0, 2.0, 6.0;
            3.0, 17.0, 18.0, 1.0;
            2.0, 5.0, 7.0, 1.0;
        ];
        let exp = matrix![
         0.72183099,  0.46126761,  1.02112676, -5.23239437;
         0.28521127,  0.23591549,  0.59859155, -2.58450704;
         -0.37676056, -0.29929577, -0.65492958,  3.20422535;
         -0.23239437, -0.00704225, -0.45070423,  1.95774648;
        ];
        assert_relative_eq!(A.inv().unwrap(), exp, max_relative = 1e-6);
    }
}
