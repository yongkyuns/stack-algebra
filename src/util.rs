#[macro_export]
macro_rules! matrix {
    ($($data:tt)*) => {
        $crate::Matrix::from_column_major_order($crate::proc_macro::matrix!($($data)*))
    };
}

/// A macro for composing vectors.
#[macro_export]
macro_rules! vector {
    ($($data:tt)*) => {
        $crate::Matrix::from_column_major_order($crate::proc_macro::matrix!($($data)*))
    };
}

#[macro_export]
macro_rules! zeros {
    ($cols:expr) => {
        $crate::Matrix::<$cols, $cols>::zeros()
    };
    ($rows:expr, $cols:expr) => {{
        $crate::Matrix::<$rows, $cols>::zeros()
    }};
    ($rows:expr, $cols:expr, $ty:ty) => {{
        $crate::Matrix::<$rows, $cols, $ty>::zeros()
    }};
}

#[macro_export]
macro_rules! ones {
    ($cols:expr) => {
        $crate::Matrix::<$cols, $cols>::ones()
    };
    ($rows:expr, $cols:expr) => {{
        $crate::Matrix::<$rows, $cols>::ones()
    }};
    ($rows:expr, $cols:expr, $ty:ty) => {{
        $crate::Matrix::<$rows, $cols, $ty>::ones()
    }};
}
