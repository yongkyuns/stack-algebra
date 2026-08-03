//! Numeric traits used by dense algorithms.

pub use num_traits::{AsPrimitive, Float, One, Zero};

/// Floating-point scalars supported by dense numerical algorithms.
pub trait Real: Float {}

impl<T: Float> Real for T {}
