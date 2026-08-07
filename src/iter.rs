//! Iteration over fixed-size matrix storage.
//!
//! Matrix element iteration is exposed through the stable slice accessors
//! [`Matrix::iter`](crate::Matrix::iter) and
//! [`Matrix::iter_mut`](crate::Matrix::iter_mut). They traverse the underlying
//! column-major storage and are allocation-free. Row and column views also
//! dereference to `stride::Stride`, which provides slice-style iteration.
//!
//! # Example
//!
//! ```
//! use stack_algebra::matrix;
//! let mut m = matrix![1_i32, 2; 3, 4];
//! let sum: i32 = m.iter().copied().sum();
//! assert_eq!(sum, 10);
//! for value in m.iter_mut() {
//!     *value *= 2;
//! }
//! assert_eq!(m.as_slice(), &[2, 6, 4, 8]);
//! ```
