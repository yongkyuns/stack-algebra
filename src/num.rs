//! Abstractions over number types.

/// Defines the absolute value for a type.
pub trait Abs {
    /// Returns the absolute value of this type.
    fn abs(self) -> Self;
}

/// Defines the square-root value for a type.
pub trait Sqrt {
    /// Returns the square-root value of this type.
    fn sqrt(self) -> Self;
}

/// Defines a multiplicative identity element for a type.
pub trait One {
    /// Returns the multiplicative identity element of this type.
    fn one() -> Self;
}

/// Defines a additive identity element for a type.
pub trait Zero {
    /// Returns the additive identity element of this type.
    fn zero() -> Self;
}

macro_rules! impl_one {
    ($one:literal $($ty:ty)+) => ($(
        impl One for $ty {
            #[inline]
            fn one() -> $ty {
                $one
            }
        }
    )+)
}

macro_rules! impl_zero {
    ($zero:literal $($ty:ty)+) => ($(
        impl Zero for $ty {
            #[inline]
            fn zero() -> $ty {
                $zero
            }
        }
    )+)
}

macro_rules! impl_abs {
    ($($ty:ident)+) => ($(
        impl Abs for $ty {
            #[inline]
            fn abs(self) -> $ty {
                $ty::abs(self)
            }
        }
    )+)
}

impl Abs for f32 {
    fn abs(self) -> Self {
        libm::fabsf(self)
    }
}

impl Abs for f64 {
    fn abs(self) -> Self {
        libm::fabs(self)
    }
}

impl Sqrt for f32 {
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
}

impl Sqrt for f64 {
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
}

macro_rules! impl_abs_self {
    ($($ty:ident)+) => ($(
        impl Abs for $ty {
            #[inline]
            fn abs(self) -> $ty {
                self
            }
        }
    )+)
}

impl_one! { true bool }
impl_one! { 1 usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }
impl_one! { 1.0 f32 f64 }

impl_zero! { false bool }
impl_zero! { 0 usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }
impl_zero! { 0.0 f32 f64 }

impl_abs_self! { usize u8 u16 u32 u64 u128 }
impl_abs! { isize i8 i16 i32 i64 i128 }
