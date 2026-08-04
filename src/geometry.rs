//! Fixed-size 3D rotation and rigid/affine transform primitives.
//!
//! The geometry types use the same compile-time dimensions and scalar generic
//! as [`Matrix`](crate::Matrix). Constructors validate inputs that represent a
//! rotation and return `Option` when the value is degenerate or non-finite.
//! All values are plain fixed-size structs: creating or composing them does
//! not require a heap allocation.
//!
//! # Example
//!
//! ```
//! use stack_algebra::{vector, Isometry, Quaternion};
//!
//! let rotation = Quaternion::from_axis_angle(
//!     &vector![0.0_f64; 0.0; 1.0],
//!     core::f64::consts::FRAC_PI_2,
//! )
//! .unwrap();
//! let pose = Isometry::from_parts(
//!     rotation.to_rotation_matrix().unwrap(),
//!     vector![1.0_f64; 2.0; 3.0],
//! );
//! let point = pose.apply_point(&vector![1.0_f64; 0.0; 0.0]);
//! assert!((point[0] - 1.0).abs() < 1e-12);
//! assert!((point[1] - 3.0).abs() < 1e-12);
//! ```

use core::ops::Mul;

use crate::{Matrix, MatrixScalar, Real, ReductionScalar, Vector};

/// A scalar-first quaternion `(w, x, y, z)`.
///
/// Quaternions are preferred for composing rotations and interpolation. Use
/// [`Quaternion::from_axis_angle`] or [`Quaternion::from_rotation_matrix`] to
/// construct validated rotations, then [`Quaternion::rotate_vector`] to apply
/// one without manually expanding the formula.
///
/// # Example
///
/// ```
/// use stack_algebra::{vector, Quaternion};
/// let q = Quaternion::from_axis_angle(
///     &vector![0.0_f32; 0.0; 1.0],
///     core::f32::consts::FRAC_PI_2,
/// )
/// .unwrap();
/// let result = q.rotate_vector(&vector![1.0_f32; 0.0; 0.0]).unwrap();
/// assert!(result[1] > 0.99);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion<T> {
    scalar: T,
    vector: Vector<3, T>,
}

impl<T: Real + MatrixScalar + ReductionScalar> Quaternion<T> {
    /// Creates a quaternion from its scalar and vector components.
    #[inline]
    pub fn new(scalar: T, x: T, y: T, z: T) -> Self {
        Self {
            scalar,
            vector: Vector::from_columns([[x, y, z]]),
        }
    }

    /// Returns the identity rotation quaternion.
    #[inline]
    pub fn identity() -> Self {
        Self::new(T::one(), T::zero(), T::zero(), T::zero())
    }

    /// Returns the scalar component.
    #[inline]
    pub fn scalar(&self) -> T {
        self.scalar
    }

    /// Returns the vector component.
    #[inline]
    pub fn vector(&self) -> &Vector<3, T> {
        &self.vector
    }

    /// Returns the squared norm.
    #[inline]
    pub fn norm_squared(&self) -> T {
        self.scalar * self.scalar
            + self.vector[0] * self.vector[0]
            + self.vector[1] * self.vector[1]
            + self.vector[2] * self.vector[2]
    }

    /// Returns the norm.
    #[inline]
    pub fn norm(&self) -> T {
        self.norm_squared().sqrt()
    }

    /// Returns a normalized quaternion, or `None` for a zero/non-finite input.
    #[inline]
    pub fn normalized(&self) -> Option<Self> {
        let norm = self.norm();
        if !norm.is_finite() || norm <= T::epsilon() {
            return None;
        }
        Some(Self {
            scalar: self.scalar / norm,
            vector: self.vector / norm,
        })
    }

    /// Returns the conjugate quaternion.
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self {
            scalar: self.scalar,
            vector: -self.vector,
        }
    }

    /// Returns the multiplicative inverse, or `None` for a zero/non-finite input.
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let norm_squared = self.norm_squared();
        if !norm_squared.is_finite() || norm_squared <= T::epsilon() {
            return None;
        }
        Some(Self {
            scalar: self.scalar / norm_squared,
            vector: -self.vector / norm_squared,
        })
    }

    /// Creates a unit quaternion from an axis-angle pair.
    #[inline]
    pub fn from_axis_angle(axis: &Vector<3, T>, angle: T) -> Option<Self> {
        let axis_norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if !angle.is_finite() || !axis_norm.is_finite() || axis_norm <= T::epsilon() {
            return None;
        }
        let half_angle = angle / (T::one() + T::one());
        let (sine, cosine) = half_angle.sin_cos();
        Some(Self::new(
            cosine,
            axis[0] / axis_norm * sine,
            axis[1] / axis_norm * sine,
            axis[2] / axis_norm * sine,
        ))
    }

    /// Converts this quaternion to a rotation matrix after normalization.
    #[inline]
    pub fn to_rotation_matrix(&self) -> Option<RotationMatrix<T>> {
        let normalized = self.normalized()?;
        let two = T::one() + T::one();
        let x = normalized.vector[0];
        let y = normalized.vector[1];
        let z = normalized.vector[2];
        let w = normalized.scalar;
        Some(RotationMatrix {
            matrix: Matrix::from_rows([
                [
                    T::one() - two * (y * y + z * z),
                    two * (x * y - z * w),
                    two * (x * z + y * w),
                ],
                [
                    two * (x * y + z * w),
                    T::one() - two * (x * x + z * z),
                    two * (y * z - x * w),
                ],
                [
                    two * (x * z - y * w),
                    two * (y * z + x * w),
                    T::one() - two * (x * x + y * y),
                ],
            ]),
        })
    }

    /// Creates a quaternion from a rotation matrix.
    #[inline]
    pub fn from_rotation_matrix(matrix: &Matrix<3, 3, T>) -> Option<Self> {
        for value in matrix.as_slice() {
            if !value.is_finite() {
                return None;
            }
        }
        let trace = matrix[(0, 0)] + matrix[(1, 1)] + matrix[(2, 2)];
        let two = T::one() + T::one();
        let four = two + two;
        let quaternion = if trace > T::zero() {
            let scale = (trace + T::one()).sqrt() * two;
            Self::new(
                scale / four,
                (matrix[(2, 1)] - matrix[(1, 2)]) / scale,
                (matrix[(0, 2)] - matrix[(2, 0)]) / scale,
                (matrix[(1, 0)] - matrix[(0, 1)]) / scale,
            )
        } else if matrix[(0, 0)] > matrix[(1, 1)] && matrix[(0, 0)] > matrix[(2, 2)] {
            let scale = (T::one() + matrix[(0, 0)] - matrix[(1, 1)] - matrix[(2, 2)]).sqrt() * two;
            Self::new(
                (matrix[(2, 1)] - matrix[(1, 2)]) / scale,
                scale / four,
                (matrix[(0, 1)] + matrix[(1, 0)]) / scale,
                (matrix[(0, 2)] + matrix[(2, 0)]) / scale,
            )
        } else if matrix[(1, 1)] > matrix[(2, 2)] {
            let scale = (T::one() + matrix[(1, 1)] - matrix[(0, 0)] - matrix[(2, 2)]).sqrt() * two;
            Self::new(
                (matrix[(0, 2)] - matrix[(2, 0)]) / scale,
                (matrix[(0, 1)] + matrix[(1, 0)]) / scale,
                scale / four,
                (matrix[(1, 2)] + matrix[(2, 1)]) / scale,
            )
        } else {
            let scale = (T::one() + matrix[(2, 2)] - matrix[(0, 0)] - matrix[(1, 1)]).sqrt() * two;
            Self::new(
                (matrix[(1, 0)] - matrix[(0, 1)]) / scale,
                (matrix[(0, 2)] + matrix[(2, 0)]) / scale,
                (matrix[(1, 2)] + matrix[(2, 1)]) / scale,
                scale / four,
            )
        };
        quaternion.normalized()
    }

    /// Rotates a 3D vector, returning `None` for a zero/non-finite quaternion.
    #[inline]
    pub fn rotate_vector(&self, vector: &Vector<3, T>) -> Option<Vector<3, T>> {
        Some(self.to_rotation_matrix()?.apply(vector))
    }

    /// Interpolates between two rotations using spherical linear interpolation.
    #[inline]
    pub fn slerp(&self, other: &Self, amount: T) -> Option<Self> {
        let first = self.normalized()?;
        let mut second = other.normalized()?;
        let mut dot = first.scalar * second.scalar
            + first.vector[0] * second.vector[0]
            + first.vector[1] * second.vector[1]
            + first.vector[2] * second.vector[2];
        if dot < T::zero() {
            second = Self {
                scalar: -second.scalar,
                vector: -second.vector,
            };
            dot = -dot;
        }
        let near = T::epsilon().sqrt() * (T::one() + T::one());
        if dot > T::one() - near {
            return Self {
                scalar: first.scalar + amount * (second.scalar - first.scalar),
                vector: first.vector + (second.vector - first.vector) * amount,
            }
            .normalized();
        }
        let angle = dot.max(-T::one()).min(T::one()).acos();
        let sine = angle.sin();
        if !sine.is_finite() || sine.abs() <= T::epsilon() {
            return None;
        }
        let first_scale = ((T::one() - amount) * angle).sin() / sine;
        let second_scale = (amount * angle).sin() / sine;
        Self {
            scalar: first.scalar * first_scale + second.scalar * second_scale,
            vector: first.vector * first_scale + second.vector * second_scale,
        }
        .normalized()
    }
}

/// An angle-axis representation of a 3D rotation.
///
/// The axis is normalized during construction. This representation is useful
/// at API boundaries where a rotation is naturally expressed as an axis and a
/// signed angle; convert it to a [`Quaternion`] or [`RotationMatrix`] for
/// repeated application.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngleAxis<T> {
    angle: T,
    axis: Vector<3, T>,
}

impl<T: Real + MatrixScalar + ReductionScalar> AngleAxis<T> {
    /// Creates a normalized angle-axis rotation.
    #[inline]
    pub fn new(axis: &Vector<3, T>, angle: T) -> Option<Self> {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if !angle.is_finite() || !norm.is_finite() || norm <= T::epsilon() {
            return None;
        }
        Some(Self {
            angle,
            axis: *axis / norm,
        })
    }

    /// Returns the rotation angle.
    #[inline]
    pub fn angle(&self) -> T {
        self.angle
    }

    /// Returns the normalized rotation axis.
    #[inline]
    pub fn axis(&self) -> &Vector<3, T> {
        &self.axis
    }

    /// Converts the angle-axis rotation to a quaternion.
    #[inline]
    pub fn to_quaternion(&self) -> Quaternion<T> {
        let half_angle = self.angle / (T::one() + T::one());
        let (sine, cosine) = half_angle.sin_cos();
        Quaternion::new(
            cosine,
            self.axis[0] * sine,
            self.axis[1] * sine,
            self.axis[2] * sine,
        )
    }

    /// Converts the angle-axis rotation to a rotation matrix.
    #[inline]
    pub fn to_rotation_matrix(&self) -> Option<RotationMatrix<T>> {
        self.to_quaternion().to_rotation_matrix()
    }
}

impl<T: Real + MatrixScalar + ReductionScalar> Mul for Quaternion<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let w = self.scalar * rhs.scalar
            - self.vector[0] * rhs.vector[0]
            - self.vector[1] * rhs.vector[1]
            - self.vector[2] * rhs.vector[2];
        let cross = Vector::from_columns([[
            self.vector[1] * rhs.vector[2] - self.vector[2] * rhs.vector[1],
            self.vector[2] * rhs.vector[0] - self.vector[0] * rhs.vector[2],
            self.vector[0] * rhs.vector[1] - self.vector[1] * rhs.vector[0],
        ]]);
        let vector = rhs.vector * self.scalar + self.vector * rhs.scalar + cross;
        Self { scalar: w, vector }
    }
}

/// A validated 3D rotation matrix.
///
/// A `RotationMatrix` can only be constructed from an orthonormal, finite
/// 3-by-3 matrix or a valid quaternion. Its [`RotationMatrix::apply`] method
/// applies the rotation to a direction without translation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotationMatrix<T> {
    matrix: Matrix<3, 3, T>,
}

/// A fixed-size rigid transform consisting of a rotation and translation.
///
/// `Isometry` models a pose in 3D. `apply_point` includes translation while
/// `apply_direction` intentionally omits it. Composition follows the usual
/// frame convention: `a.compose(&b)` applies `b` first, then `a`.
///
/// # Example
///
/// ```
/// use stack_algebra::{vector, Isometry, RotationMatrix};
/// let pose = Isometry::from_parts(RotationMatrix::identity(), vector![1.0_f32; 2.0; 3.0]);
/// assert_eq!(pose.apply_point(&vector![4.0_f32; 5.0; 6.0]), vector![5.0; 7.0; 9.0]);
/// assert_eq!(pose.apply_direction(&vector![4.0_f32; 5.0; 6.0]), vector![4.0; 5.0; 6.0]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Isometry<T> {
    rotation: RotationMatrix<T>,
    translation: Vector<3, T>,
}

/// A fixed-size affine transform represented by a homogeneous 4-by-4 matrix.
///
/// Affine transforms support a general 3-by-3 linear part (for example,
/// scaling or shear) plus translation. `from_matrix` validates the homogeneous
/// bottom row; use [`AffineTransform::from_parts`] when the two components are
/// already available separately.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineTransform<T> {
    matrix: Matrix<4, 4, T>,
}

impl<T: Real + MatrixScalar + ReductionScalar> RotationMatrix<T> {
    /// Returns the identity rotation.
    #[inline]
    pub fn identity() -> Self {
        Self {
            matrix: Matrix::eye(),
        }
    }

    /// Creates a rotation matrix from a quaternion.
    #[inline]
    pub fn from_quaternion(quaternion: &Quaternion<T>) -> Option<Self> {
        quaternion.to_rotation_matrix()
    }

    /// Creates a rotation matrix from an existing orthonormal matrix.
    #[inline]
    pub fn from_matrix(matrix: Matrix<3, 3, T>) -> Option<Self> {
        let rotation = Quaternion::from_rotation_matrix(&matrix)?.to_rotation_matrix()?;
        let tolerance = T::epsilon() * T::from(100).unwrap_or(T::one());
        for row in 0..3 {
            for column in 0..3 {
                let expected = rotation.matrix[(row, column)];
                let scale = T::one()
                    .max(matrix[(row, column)].abs())
                    .max(expected.abs());
                if (matrix[(row, column)] - expected).abs() > tolerance * scale {
                    return None;
                }
            }
        }
        Some(Self { matrix })
    }

    /// Returns the underlying matrix.
    #[inline]
    pub fn matrix(&self) -> &Matrix<3, 3, T> {
        &self.matrix
    }

    /// Applies this rotation to a vector.
    #[inline]
    pub fn apply(&self, vector: &Vector<3, T>) -> Vector<3, T> {
        let mut output = Vector::<3, T>::zeros();
        self.matrix.matvec_into(vector, &mut output);
        output
    }

    /// Returns the inverse rotation.
    #[inline]
    pub fn inverse(&self) -> Self {
        Self {
            matrix: self.matrix.transpose(),
        }
    }

    /// Composes this rotation with `rhs`.
    #[inline]
    pub fn compose(&self, rhs: &Self) -> Self {
        Self {
            matrix: self.matrix * rhs.matrix,
        }
    }

    /// Converts this rotation to a unit quaternion.
    #[inline]
    pub fn to_quaternion(&self) -> Option<Quaternion<T>> {
        Quaternion::from_rotation_matrix(&self.matrix)
    }
}

impl<T: Real + MatrixScalar + ReductionScalar> Isometry<T> {
    /// Returns the identity transform.
    #[inline]
    pub fn identity() -> Self {
        Self {
            rotation: RotationMatrix::identity(),
            translation: Vector::zeros(),
        }
    }

    /// Creates a transform from a rotation and translation.
    #[inline]
    pub fn from_parts(rotation: RotationMatrix<T>, translation: Vector<3, T>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Returns the rotation component.
    #[inline]
    pub fn rotation(&self) -> &RotationMatrix<T> {
        &self.rotation
    }

    /// Returns the translation component.
    #[inline]
    pub fn translation(&self) -> &Vector<3, T> {
        &self.translation
    }

    /// Applies the transform to a point.
    #[inline]
    pub fn apply_point(&self, point: &Vector<3, T>) -> Vector<3, T> {
        self.rotation.apply(point) + self.translation
    }

    /// Applies only the rotational part to a direction.
    #[inline]
    pub fn apply_direction(&self, direction: &Vector<3, T>) -> Vector<3, T> {
        self.rotation.apply(direction)
    }

    /// Composes this transform with `rhs`.
    #[inline]
    pub fn compose(&self, rhs: &Self) -> Self {
        Self {
            rotation: self.rotation.compose(&rhs.rotation),
            translation: self.apply_point(&rhs.translation),
        }
    }

    /// Returns the inverse transform.
    #[inline]
    pub fn inverse(&self) -> Self {
        let rotation = self.rotation.inverse();
        Self {
            translation: rotation.apply(&(-self.translation)),
            rotation,
        }
    }

    /// Converts this transform to a homogeneous 4-by-4 matrix.
    #[inline]
    pub fn to_homogeneous(&self) -> Matrix<4, 4, T> {
        let mut matrix = Matrix::<4, 4, T>::eye();
        for row in 0..3 {
            for column in 0..3 {
                matrix[(row, column)] = self.rotation.matrix()[(row, column)];
            }
            matrix[(row, 3)] = self.translation[row];
        }
        matrix
    }

    /// Creates a transform from a homogeneous 4-by-4 matrix.
    #[inline]
    pub fn from_homogeneous(matrix: Matrix<4, 4, T>) -> Option<Self> {
        let tolerance = T::epsilon() * T::from(100).unwrap_or(T::one());
        let scale = T::one().max(matrix[(3, 3)].abs());
        if (matrix[(3, 3)] - T::one()).abs() > tolerance * scale {
            return None;
        }
        for column in 0..3 {
            if matrix[(3, column)].abs() > tolerance {
                return None;
            }
        }
        let rotation_matrix = Matrix::<3, 3, T>::from_fn(|row, column| matrix[(row, column)]);
        let rotation = RotationMatrix::from_matrix(rotation_matrix)?;
        let translation = Vector::from_columns([[matrix[(0, 3)], matrix[(1, 3)], matrix[(2, 3)]]]);
        Some(Self::from_parts(rotation, translation))
    }
}

impl<T: Real + MatrixScalar + ReductionScalar> AffineTransform<T> {
    /// Returns the identity affine transform.
    #[inline]
    pub fn identity() -> Self {
        Self {
            matrix: Matrix::eye(),
        }
    }

    /// Creates an affine transform from a linear part and translation.
    #[inline]
    pub fn from_parts(linear: Matrix<3, 3, T>, translation: Vector<3, T>) -> Self {
        let mut matrix = Matrix::<4, 4, T>::eye();
        for row in 0..3 {
            for column in 0..3 {
                matrix[(row, column)] = linear[(row, column)];
            }
            matrix[(row, 3)] = translation[row];
        }
        Self { matrix }
    }

    /// Creates an affine transform from a homogeneous matrix.
    #[inline]
    pub fn from_matrix(matrix: Matrix<4, 4, T>) -> Option<Self> {
        if matrix.as_slice().iter().any(|value| !value.is_finite()) {
            return None;
        }
        let tolerance = T::epsilon() * T::from(100).unwrap_or(T::one());
        let scale = T::one().max(matrix[(3, 3)].abs());
        if (matrix[(3, 3)] - T::one()).abs() > tolerance * scale {
            return None;
        }
        for column in 0..3 {
            if matrix[(3, column)].abs() > tolerance {
                return None;
            }
        }
        Some(Self { matrix })
    }

    /// Returns the homogeneous matrix.
    #[inline]
    pub fn matrix(&self) -> &Matrix<4, 4, T> {
        &self.matrix
    }

    /// Returns the linear part of the transform.
    #[inline]
    pub fn linear(&self) -> Matrix<3, 3, T> {
        Matrix::from_fn(|row, column| self.matrix[(row, column)])
    }

    /// Returns the translation part of the transform.
    #[inline]
    pub fn translation(&self) -> Vector<3, T> {
        Vector::from_columns([[
            self.matrix[(0, 3)],
            self.matrix[(1, 3)],
            self.matrix[(2, 3)],
        ]])
    }

    /// Applies the affine transform to a point.
    #[inline]
    pub fn apply_point(&self, point: &Vector<3, T>) -> Vector<3, T> {
        let mut output = Vector::<3, T>::zeros();
        for row in 0..3 {
            let mut value = self.matrix[(row, 3)];
            for column in 0..3 {
                value = value + self.matrix[(row, column)] * point[column];
            }
            output[row] = value;
        }
        output
    }

    /// Applies only the linear part to a direction.
    #[inline]
    pub fn apply_direction(&self, direction: &Vector<3, T>) -> Vector<3, T> {
        let mut output = Vector::<3, T>::zeros();
        for row in 0..3 {
            for column in 0..3 {
                output[row] = output[row] + self.matrix[(row, column)] * direction[column];
            }
        }
        output
    }

    /// Composes this affine transform with `rhs`.
    #[inline]
    pub fn compose(&self, rhs: &Self) -> Self {
        Self {
            matrix: self.matrix * rhs.matrix,
        }
    }

    /// Returns the inverse, or `None` when the linear part is singular.
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let linear = self.linear();
        let factor = linear.partial_piv_lu();
        let determinant = factor.determinant();
        if !determinant.is_finite() || determinant.abs() <= T::epsilon() {
            return None;
        }
        let inverse_linear = factor.inverse();
        let inverse_translation = -(inverse_linear * self.translation());
        Some(Self::from_parts(inverse_linear, inverse_translation))
    }
}

impl<T: Real + MatrixScalar + ReductionScalar> Mul for AffineTransform<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.compose(&rhs)
    }
}

impl<T: Real + MatrixScalar + ReductionScalar> Mul for Isometry<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.compose(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{matrix, vector, AffineTransform, AngleAxis, Isometry, Quaternion, RotationMatrix};

    #[test]
    fn rotates_vector_and_round_trips_matrix() {
        let axis = vector![0.0_f64; 0.0; 1.0];
        let quaternion = Quaternion::from_axis_angle(&axis, core::f64::consts::FRAC_PI_2)
            .expect("axis is nonzero");
        let vector = vector![1.0_f64; 0.0; 0.0];
        let rotated = quaternion
            .rotate_vector(&vector)
            .expect("quaternion is valid");
        assert_relative_eq!(rotated[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(rotated[1], 1.0, epsilon = 1e-12);
        let matrix = quaternion
            .to_rotation_matrix()
            .expect("quaternion is valid");
        let round_trip = matrix.to_quaternion().expect("matrix is a rotation");
        assert_relative_eq!(round_trip.norm(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(matrix.apply(&vector), rotated, epsilon = 1e-12);
    }

    #[test]
    fn composition_and_inverse_cancel() {
        let first =
            Quaternion::from_axis_angle(&vector![1.0_f64; 0.0; 0.0], 0.3).expect("axis is nonzero");
        let second = Quaternion::from_axis_angle(&vector![0.0_f64; 1.0; 0.0], -0.7)
            .expect("axis is nonzero");
        let composed = first * second;
        let identity = composed * composed.inverse().expect("quaternion is nonzero");
        assert_relative_eq!(identity.scalar(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(identity.vector()[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(identity.vector()[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(identity.vector()[2], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn rejects_zero_quaternion_and_invalid_rotation() {
        assert!(Quaternion::<f64>::new(0.0, 0.0, 0.0, 0.0)
            .to_rotation_matrix()
            .is_none());
        assert!(Quaternion::from_axis_angle(&vector![1.0_f64; 0.0; 0.0], f64::NAN).is_none());
        assert!(RotationMatrix::from_matrix(matrix![
            2.0_f64, 0.0, 0.0;
            0.0, 2.0, 0.0;
            0.0, 0.0, 2.0
        ])
        .is_none());
        assert!(AffineTransform::from_matrix(matrix![
            f64::NAN, 0.0, 0.0, 0.0;
            0.0, 1.0, 0.0, 0.0;
            0.0, 0.0, 1.0, 0.0;
            0.0, 0.0, 0.0, 1.0
        ])
        .is_none());
    }

    #[test]
    fn isometry_composes_inverts_and_round_trips() {
        let first_rotation =
            Quaternion::from_axis_angle(&vector![0.0_f64; 0.0; 1.0], core::f64::consts::FRAC_PI_2)
                .expect("axis is nonzero")
                .to_rotation_matrix()
                .expect("quaternion is valid");
        let second_rotation = Quaternion::from_axis_angle(&vector![1.0_f64; 0.0; 0.0], 0.25)
            .expect("axis is nonzero")
            .to_rotation_matrix()
            .expect("quaternion is valid");
        let first = Isometry::from_parts(first_rotation, vector![1.0_f64; 2.0; 3.0]);
        let second = Isometry::from_parts(second_rotation, vector![-1.0_f64; 0.5; 2.0]);
        let composed = first.compose(&second);
        let point = vector![0.5_f64; -1.0; 2.0];
        let expected = first.apply_point(&second.apply_point(&point));
        assert_relative_eq!(composed.apply_point(&point), expected, epsilon = 1e-12);
        assert_relative_eq!(
            composed
                .inverse()
                .apply_point(&composed.apply_point(&point)),
            point,
            epsilon = 1e-12
        );
        let homogeneous = composed.to_homogeneous();
        let round_trip = Isometry::from_homogeneous(homogeneous).expect("matrix is valid");
        assert_relative_eq!(round_trip.apply_point(&point), expected, epsilon = 1e-12);
    }

    #[test]
    fn angle_axis_slerp_and_affine_transform_work() {
        let axis = vector![0.0_f64; 0.0; 1.0];
        let angle_axis =
            AngleAxis::new(&axis, core::f64::consts::FRAC_PI_2).expect("axis is nonzero");
        let halfway = Quaternion::identity()
            .slerp(&angle_axis.to_quaternion(), 0.5)
            .expect("quaternions are valid");
        let rotated = halfway
            .rotate_vector(&vector![1.0_f64; 0.0; 0.0])
            .expect("quaternion is valid");
        assert_relative_eq!(
            rotated[0],
            core::f64::consts::FRAC_1_SQRT_2,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            rotated[1],
            core::f64::consts::FRAC_1_SQRT_2,
            epsilon = 1e-12
        );

        let affine = AffineTransform::from_matrix(matrix![
            2.0_f64, 0.0, 0.0, 1.0;
            0.0, 3.0, 0.0, 2.0;
            0.0, 0.0, 4.0, 3.0;
            0.0, 0.0, 0.0, 1.0
        ])
        .expect("matrix is affine");
        let affine_from_parts = AffineTransform::from_parts(
            matrix![
                2.0_f64, 0.0, 0.0;
                0.0, 3.0, 0.0;
                0.0, 0.0, 4.0
            ],
            vector![1.0_f64; 2.0; 3.0],
        );
        assert_eq!(affine_from_parts.matrix(), affine.matrix());
        assert_eq!(affine.linear(), affine_from_parts.linear());
        assert_eq!(affine.translation(), vector![1.0_f64; 2.0; 3.0]);
        let point = vector![1.0_f64; 2.0; 3.0];
        assert_relative_eq!(
            affine.apply_point(&point),
            vector![3.0_f64; 8.0; 15.0],
            epsilon = 1e-12
        );
        let inverse = affine.inverse().expect("linear part is invertible");
        assert_relative_eq!(
            inverse.apply_point(&affine.apply_point(&point)),
            point,
            epsilon = 1e-12
        );
    }
}
