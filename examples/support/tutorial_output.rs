//! Host-only formatting shared by the runnable tutorials; no numerical work.

use std::io;

pub fn csv_requested() -> io::Result<bool> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        [] => Ok(false),
        [flag] if flag == "--csv" => Ok(true),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: example [--csv]",
        )),
    }
}

pub fn csv_row(values: &[f64]) {
    for (column, value) in values.iter().enumerate() {
        if column != 0 {
            print!(",");
        }
        // Sufficient significant digits to round-trip f64 and promoted f32.
        print!("{value:.17e}");
    }
    println!();
}
