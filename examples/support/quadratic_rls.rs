//! Fixed-memory square-root RLS example support; no std, alloc, or sample history.
//! This is an application example, not a public stack-algebra estimator API.

use stack_algebra::{Matrix, MatrixScalar, Real};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RlsError {
    InvalidConfiguration,
    NonFiniteInput,
    NumericalBreakdown,
    SampleCountOverflow,
}

// ANCHOR: state
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadraticRls<T> {
    // R^T R is accumulated information; R * theta = transformed_rhs.
    r: Matrix<3, 3, T>,
    transformed_rhs: Matrix<3, 1, T>,
    input_scale: T,
    sqrt_forgetting: T,
    samples: usize,
}
// ANCHOR_END: state

#[derive(Clone, Copy, Debug)]
pub struct Update<T> {
    pub prediction_before: T,
    pub innovation: T,
    // Coefficients in original x coordinates, not the normalized basis.
    pub coefficients: Matrix<3, 1, T>,
}

impl<T: Real + MatrixScalar> QuadraticRls<T> {
    /// The prior is precision * ||theta - initial||^2 in the u=x/scale basis.
    pub fn new(
        input_scale: T,
        prior_precision: T,
        forgetting: T,
        initial: Matrix<3, 1, T>,
    ) -> Result<Self, RlsError> {
        let valid = input_scale.is_finite()
            && input_scale > T::zero()
            && prior_precision.is_finite()
            && prior_precision > T::zero()
            && forgetting.is_finite()
            && forgetting > T::zero()
            && forgetting <= T::one()
            && initial.as_slice().iter().all(|v| v.is_finite());
        if !valid {
            return Err(RlsError::InvalidConfiguration);
        }
        let root = prior_precision.sqrt();
        let estimator = Self {
            r: Matrix::from_fn(|row, col| if row == col { root } else { T::zero() }),
            transformed_rhs: Matrix::from_fn(|row, _| root * initial[(row, 0)]),
            input_scale,
            sqrt_forgetting: forgetting.sqrt(),
            samples: 0,
        };
        if !estimator.finite_factor()
            || !estimator
                .coefficients()
                .as_slice()
                .iter()
                .all(|v| v.is_finite())
        {
            return Err(RlsError::InvalidConfiguration);
        }
        Ok(estimator)
    }

    pub fn samples(&self) -> usize {
        self.samples
    }

    pub fn normalized_coefficients(&self) -> Matrix<3, 1, T> {
        self.r.upper_triangular().solve(&self.transformed_rhs)
    }

    pub fn coefficients(&self) -> Matrix<3, 1, T> {
        let mut theta = self.normalized_coefficients();
        theta[(0, 0)] = theta[(0, 0)] / self.input_scale / self.input_scale;
        theta[(1, 0)] = theta[(1, 0)] / self.input_scale;
        theta
    }

    pub fn predict(&self, x: T) -> Result<T, RlsError> {
        if !x.is_finite() {
            return Err(RlsError::NonFiniteInput);
        }
        let theta = self.normalized_coefficients();
        let u = x / self.input_scale;
        let prediction = (theta[(0, 0)] * u + theta[(1, 0)]) * u + theta[(2, 0)];
        if !u.is_finite() || !prediction.is_finite() {
            return Err(RlsError::NumericalBreakdown);
        }
        Ok(prediction)
    }

    fn finite_factor(&self) -> bool {
        self.r.as_slice().iter().all(|v| v.is_finite())
            && self
                .transformed_rhs
                .as_slice()
                .iter()
                .all(|v| v.is_finite())
            && (0..3).all(|j| self.r[(j, j)] > T::zero())
    }

    // ANCHOR: update
    pub fn update(&mut self, x: T, measured_y: T) -> Result<Update<T>, RlsError> {
        if !x.is_finite() || !measured_y.is_finite() {
            return Err(RlsError::NonFiniteInput);
        }
        let prediction_before = self.predict(x)?;
        let innovation = measured_y - prediction_before;
        let u = x / self.input_scale;
        let mut row = [u * u, u, T::one()];
        if !innovation.is_finite() || !row.iter().all(|v| v.is_finite()) {
            return Err(RlsError::NumericalBreakdown);
        }
        // Stage the complete update. Rejection never partially changes self.
        let mut next = *self;
        next.samples = self
            .samples
            .checked_add(1)
            .ok_or(RlsError::SampleCountOverflow)?;
        for value in next.r.as_mut_slice() {
            *value = *value * self.sqrt_forgetting;
        }
        for value in next.transformed_rhs.as_mut_slice() {
            *value = *value * self.sqrt_forgetting;
        }
        let mut rhs = measured_y;
        // Append one [u^2, u, 1 | y] row and eliminate it with three Givens rotations.
        for j in 0..3 {
            let radius = next.r[(j, j)].hypot(row[j]);
            if !radius.is_finite() || radius <= T::zero() {
                return Err(RlsError::NumericalBreakdown);
            }
            let cosine = next.r[(j, j)] / radius;
            let sine = row[j] / radius;
            for (col, incoming) in row.iter_mut().enumerate().skip(j) {
                let old = next.r[(j, col)];
                let value = *incoming;
                next.r[(j, col)] = cosine * old + sine * value;
                *incoming = cosine * value - sine * old;
            }
            next.r[(j, j)] = radius;
            row[j] = T::zero();
            let old = next.transformed_rhs[(j, 0)];
            next.transformed_rhs[(j, 0)] = cosine * old + sine * rhs;
            rhs = cosine * rhs - sine * old;
        }
        if !next.finite_factor() || !rhs.is_finite() {
            return Err(RlsError::NumericalBreakdown);
        }
        let coefficients = next.coefficients();
        if !coefficients.as_slice().iter().all(|v| v.is_finite()) {
            return Err(RlsError::NumericalBreakdown);
        }
        *self = next;
        Ok(Update {
            prediction_before,
            innovation,
            coefficients,
        })
    }
    // ANCHOR_END: update
}
