#[macro_export]
macro_rules! join {
    (
        $first:expr, $second:expr
        $(,$rest:expr)*
    ) => {{
        let mut out = $first.to_string();
        let second = $second.to_string();
        out.push_str(&second);
        $(out.push_str(&join!($rest));)*
        out
    }};
    ($last:expr $(,)? ) => {{
        let out = $last.to_string();
        out
    }};
}

#[macro_export]
macro_rules! disp {
    (
        $first:expr, $second:expr
        $(,$rest:expr)*
    ) => {
        println!("{}, {}", $first, $second);
        disp!($($rest),*);
    };
    ($single:expr $(,)? ) => {
        println!("{}", $single);
    };
    () => {};
}
