#![allow(non_snake_case)]

use core::{
    fmt,
    iter::Sum,
    ops::{Div, Mul, Neg, Sub},
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
        + fmt::Display
        + Sum
        + Neg<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Div<Output = T>,
{
    pub fn determinant(&self) -> T {
        let (L, U, _) = self.lu_decomp();
        let mut det = T::one();
        for i in 0..D {
            det = det * L[(i, i)] * U[(i, i)];
        }
        if D % 2 != 0 {
            det = -det;
        }
        det
    }

    pub fn lu_decomp(&self) -> (Matrix<D, D, T>, Matrix<D, D, T>, Matrix<D, D, T>) {
        let mut P = eye!(D, T);
        let mut L = eye!(D, T);
        let mut U = self.clone();

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
    use crate::matrix;

    #[test]
    fn LU_decomp() {
        let A = matrix![
            1.0, 3.0, 5.0;
            2.0, 4.0, 7.0;
            1.0, 1.0, 0.0;
        ];
        let (L, U, P) = A.lu_decomp();
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
        assert_eq!(A.determinant(), 24.000000000000007);

        let A = matrix![
            3.0, 7.0;
            1.0, -4.0;
        ];
        assert_eq!(A.determinant(), -19.0);

        let A = matrix![
            1.0, 2.0;
            4.0, 8.0;
        ];
        assert_eq!(A.determinant(), 0.0);

        let A = matrix![
            1.0, 1.0;
            2.0, 2.0;
        ];
        assert_eq!(A.determinant(), 0.0);

        let A = matrix![
            1.0, 2.0, 3.0, 4.0;
            2.0, 5.0, 7.0, 3.0;
            4.0, 10.0, 14.0, 6.0;
            3.0, 4.0, 2.0, 7.0;
        ];
        assert_eq!(A.determinant(), 0.0);

        let A = matrix![
            1.0, 2.0, 3.0, 4.0;
            2.0, 5.0, 7.0, 3.0;
            4.0, 10.0, 14.0, 6.0;
            3.0, 4.0, 2.0, 7.0;
        ];
        assert_eq!(A.determinant(), 0.0);

        let A = matrix![
            11.0, 9.0, 24.0, 2.0;
            1.0, 5.0, 2.0, 6.0;
            3.0, 17.0, 18.0, 1.0;
            2.0, 5.0, 7.0, 1.0;
        ];
        assert_eq!(A.determinant(), -284.0000000000006);

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
        assert_eq!(A.determinant(), -5.195843755245731e-13);
    }
}
