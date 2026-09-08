use stack_algebra::{Matrix, Real, StaticCscCholeskyPattern, StaticCscMatrix};
use std::fmt::Debug;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn coefficient(n: usize, star: bool, row: usize, column: usize) -> f64 {
    if row == column {
        if star && row == 0 {
            n as f64 * 0.125 + 2.0
        } else {
            2.0
        }
    } else if (star && (row == 0 || column == 0)) || (!star && row.abs_diff(column) <= 2) {
        -0.125
    } else {
        0.0
    }
}

fn fixture<const N: usize, const A: usize, T: Real>(
    star: bool,
    full: bool,
) -> StaticCscMatrix<N, N, A, T> {
    let mut values = Vec::new();
    let mut rows = Vec::new();
    let mut pointers = vec![0];
    for column in 0..N {
        for row in 0..N {
            let value = coefficient(N, star, row, column);
            if value != 0.0 && (full || row >= column) {
                values.push(T::from(value).unwrap());
                rows.push(row);
            }
        }
        pointers.push(values.len());
    }
    StaticCscMatrix::from_pattern(&values, &rows, &pointers).unwrap()
}

fn check_solution<const N: usize, T: Real>(solution: &Matrix<N, 1, T>, star: bool) {
    let tolerance = if core::mem::size_of::<T>() == 4 {
        2.0e-5
    } else {
        1.0e-11
    };
    for row in 0..N {
        let actual: f64 = (0..N)
            .map(|column| {
                coefficient(N, star, row, column) * solution[(column, 0)].to_f64().unwrap()
            })
            .sum();
        let expected = (row % 5 + 1) as f64;
        assert!(actual.is_finite() && (actual - expected).abs() <= tolerance * expected);
    }
}

fn measure(mut update: impl FnMut(), timed: bool) -> (u64, u128) {
    update();
    if !timed {
        return (1, 0);
    }
    let warmup = Instant::now();
    let mut count = 0_u64;
    while warmup.elapsed() < Duration::from_millis(10) {
        for _ in 0..64 {
            update();
        }
        count += 64;
    }
    let iterations = ((count as f64 * 0.030 / warmup.elapsed().as_secs_f64()) as u64)
        .clamp(64, 5_000_000);
    let start = Instant::now();
    for _ in 0..iterations {
        update();
    }
    (iterations, start.elapsed().as_nanos())
}

#[inline(never)]
fn run<const N: usize, const A: usize, const L: usize, T: Real + Debug>(
    star: bool,
    layout: &str,
    timed: bool,
    sample: usize,
) {
    let lower = fixture::<N, A, T>(star, false);
    let full = fixture::<N, A, T>(star, true);
    let source = if layout == "rebuilt" { &full } else { &lower };
    let input = if layout == "matching" { &lower } else { &full };
    let pattern = StaticCscCholeskyPattern::<N, L>::analyze(source).unwrap();
    let rhs = Matrix::<N, 1, T>::from_fn(|row, _| T::from(row % 5 + 1).unwrap());
    let shape = if star { "star" } else { "banded" };
    let scalar = core::any::type_name::<T>();
    for ordered in [false, true] {
        let entry = if ordered { "ordered" } else { "natural" };
        let mut cholesky = pattern.factor(input).unwrap();
        check_solution(&cholesky.solve(&rhs), star);
        let (iterations, elapsed) = measure(
            || {
                let factor = black_box(&mut cholesky);
                if ordered {
                    factor.recompute_ordered_with_pattern(black_box(&pattern), black_box(input))
                } else {
                    factor.recompute_with_pattern(black_box(&pattern), black_box(input))
                }
                .unwrap();
                black_box(factor);
            },
            timed,
        );
        check_solution(&cholesky.solve(&rhs), star);
        assert_eq!(cholesky.lower().pattern(), pattern.lower());
        println!(
            "{sample},{shape},{N},{scalar},cholesky,{entry},{layout},{iterations},{elapsed},{},{}",
            input.nnz(),
            pattern.lower().nnz()
        );
        let mut ldlt = pattern.factor_ldlt(input).unwrap();
        check_solution(&ldlt.solve(&rhs), star);
        let (iterations, elapsed) = measure(
            || {
                let factor = black_box(&mut ldlt);
                if ordered {
                    factor.recompute_ordered_with_pattern(black_box(&pattern), black_box(input))
                } else {
                    factor.recompute_with_pattern(black_box(&pattern), black_box(input))
                }
                .unwrap();
                black_box(factor);
            },
            timed,
        );
        check_solution(&ldlt.solve(&rhs), star);
        assert_eq!(ldlt.lower().pattern(), pattern.lower());
        println!(
            "{sample},{shape},{N},{scalar},ldlt,{entry},{layout},{iterations},{elapsed},{},{}",
            input.nnz(),
            pattern.lower().nnz()
        );
    }
}

fn suite<T: Real + Debug>(layout: &str, timed: bool, sample: usize) {
    run::<6, 30, 15, T>(false, layout, timed, sample);
    run::<15, 75, 42, T>(false, layout, timed, sample);
    run::<64, 320, 189, T>(false, layout, timed, sample);
    run::<128, 640, 381, T>(false, layout, timed, sample);
    run::<64, 256, 2080, T>(true, layout, timed, sample);
    run::<128, 512, 8256, T>(true, layout, timed, sample);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "check".to_string());
    let sample: usize = args
        .next()
        .unwrap_or_else(|| "0".to_string())
        .parse()
        .unwrap();
    assert!(
        args.next().is_none(),
        "usage: sparse-reuse-bench [check|matching|layouts] [sample]"
    );
    let layouts: &[&str] = match mode.as_str() {
        "check" => &["matching", "repacked", "rebuilt"],
        "matching" => &["matching"],
        "layouts" if sample % 2 == 0 => &["repacked", "rebuilt"],
        "layouts" => &["rebuilt", "repacked"],
        _ => panic!("unknown benchmark mode"),
    };
    println!(
        "sample,shape,n,scalar,solver,entry,layout,iterations,elapsed_ns,input_nnz,factor_nnz"
    );
    for layout in layouts {
        suite::<f32>(layout, mode != "check", sample);
        suite::<f64>(layout, mode != "check", sample);
    }
}
