use stack_algebra::{Matrix, Real, StaticCscCholeskyPattern, StaticCscMatrix, StaticCscOrdering};
use std::fmt::Debug;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn coefficient(row: usize, column: usize) -> f64 {
    if row == column {
        2.0
    } else if row.abs_diff(column) <= 2 {
        -0.125
    } else {
        0.0
    }
}

fn fixture<const N: usize, const A: usize, T: Real>(full: bool) -> StaticCscMatrix<N, N, A, T> {
    let mut values = Vec::new();
    let mut rows = Vec::new();
    let mut pointers = vec![0];
    for column in 0..N {
        for row in 0..N {
            let value = coefficient(row, column);
            if value != 0.0 && (full || row >= column) {
                values.push(T::from(value).unwrap());
                rows.push(row);
            }
        }
        pointers.push(values.len());
    }
    StaticCscMatrix::from_pattern(&values, &rows, &pointers).unwrap()
}

fn ordering<const N: usize>(kind: &str) -> StaticCscOrdering<N> {
    let permutation: Vec<_> = (0..N)
        .map(|i| match kind {
            "identity" => i,
            "reverse" => N - 1 - i,
            "rotate" => (i + N / 2) % N,
            _ => panic!("unknown ordering"),
        })
        .collect();
    StaticCscOrdering::from_permutation(&permutation).unwrap()
}

fn reference<const N: usize, const A: usize, T: Real>(
    order: StaticCscOrdering<N>,
) -> StaticCscMatrix<N, N, A, T> {
    let mut result = StaticCscMatrix::new();
    for column in 0..N {
        for row in column..N {
            let value = coefficient(row, column);
            if value != 0.0 {
                let first = order.inverse()[row];
                let second = order.inverse()[column];
                result
                    .insert(
                        first.max(second),
                        first.min(second),
                        T::from(value).unwrap(),
                    )
                    .unwrap();
            }
        }
    }
    result
}

fn check_matrix<const N: usize, const A: usize, T: Real + Debug>(
    actual: &StaticCscMatrix<N, N, A, T>,
    expected: &StaticCscMatrix<N, N, A, T>,
) {
    assert_eq!(actual.pattern(), expected.pattern());
    assert_eq!(actual.values(), expected.values());
}

fn check_solution<const N: usize, T: Real>(solution: &Matrix<N, 1, T>) {
    let tolerance = if core::mem::size_of::<T>() == 4 {
        2.0e-5
    } else {
        1.0e-11
    };
    for row in 0..N {
        let actual: f64 = (0..N)
            .map(|column| coefficient(row, column) * solution[(column, 0)].to_f64().unwrap())
            .sum();
        let expected = (row % 5 + 1) as f64;
        assert!(actual.is_finite() && (actual - expected).abs() <= tolerance * expected);
    }
}

fn measure(mut operation: impl FnMut(), timed: bool) -> (u64, u128) {
    operation();
    if !timed {
        return (1, 0);
    }
    let warmup = Instant::now();
    let mut count = 0_u64;
    while warmup.elapsed() < Duration::from_millis(3) {
        for _ in 0..64 {
            operation();
        }
        count += 64;
    }
    let iterations =
        ((count as f64 * 0.012 / warmup.elapsed().as_secs_f64()) as u64).clamp(64, 5_000_000);
    let start = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    (iterations, start.elapsed().as_nanos())
}

#[inline(never)]
fn run<const N: usize, const A: usize, const L: usize, T: Real + Debug>(
    capacity: &str,
    layout: &str,
    kind: &str,
    sample: usize,
    timed: bool,
) {
    let input = fixture::<N, A, T>(layout == "full");
    let order = ordering::<N>(kind);
    let map = order.permutation_for_pattern(input.pattern()).unwrap();
    let expected = reference::<N, A, T>(order);
    // Both revisions get a valid matching destination. Broken historical
    // arbitrary-destination calls and changed-source calls are never timed.
    let mut output = StaticCscMatrix::zero_with_pattern(map.pattern());
    map.apply_into(&input, &mut output);
    check_matrix(&output, &expected);
    check_matrix(&map.apply(&input), &expected);
    let pattern = StaticCscCholeskyPattern::<N, L>::analyze_with_ordering(&input, order).unwrap();
    let rhs = Matrix::<N, 1, T>::from_fn(|row, _| T::from(row % 5 + 1).unwrap());
    let mut cholesky = pattern.factor_ordered(&output).unwrap();
    let mut ldlt = pattern.factor_ldlt_ordered(&output).unwrap();
    check_solution(&cholesky.solve(&rhs));
    check_solution(&ldlt.solve(&rhs));
    let mut operations = ["apply_into", "apply", "permute_cholesky", "permute_ldlt"];
    if sample & 1 != 0 {
        operations.reverse();
    }
    for operation in operations {
        let (iterations, elapsed) = match operation {
            "apply" => measure(
                || {
                    black_box(black_box(&map).apply(black_box(&input)));
                },
                timed,
            ),
            "apply_into" => measure(
                || {
                    black_box(&map).apply_into(black_box(&input), black_box(&mut output));
                    black_box(&output);
                },
                timed,
            ),
            "permute_cholesky" => measure(
                || {
                    black_box(&map).apply_into(black_box(&input), black_box(&mut output));
                    black_box(&mut cholesky)
                        .recompute_ordered_with_pattern(black_box(&pattern), black_box(&output))
                        .unwrap();
                    black_box(&cholesky);
                },
                timed,
            ),
            "permute_ldlt" => measure(
                || {
                    black_box(&map).apply_into(black_box(&input), black_box(&mut output));
                    black_box(&mut ldlt)
                        .recompute_ordered_with_pattern(black_box(&pattern), black_box(&output))
                        .unwrap();
                    black_box(&ldlt);
                },
                timed,
            ),
            _ => unreachable!(),
        };
        check_matrix(&output, &expected);
        check_matrix(&map.apply(&input), &expected);
        check_solution(&cholesky.solve(&rhs));
        check_solution(&ldlt.solve(&rhs));
        assert_eq!(cholesky.lower().pattern(), pattern.lower());
        assert_eq!(ldlt.lower().pattern(), pattern.lower());
        println!(
            "{sample},{N},{},{capacity},{layout},{kind},{operation},{A},{L},{},{},{},{iterations},{elapsed}",
            core::any::type_name::<T>(),
            input.nnz(),
            output.nnz(),
            pattern.lower().nnz()
        );
    }
}

fn suite<T: Real + Debug>(sample: usize, timed: bool) {
    for layout in ["lower", "full"] {
        for kind in ["identity", "reverse", "rotate"] {
            run::<6, 30, 48, T>("compact", layout, kind, sample, timed);
            run::<6, 120, 48, T>("roomy", layout, kind, sample, timed);
            run::<15, 75, 120, T>("compact", layout, kind, sample, timed);
            run::<15, 300, 120, T>("roomy", layout, kind, sample, timed);
            run::<64, 320, 512, T>("compact", layout, kind, sample, timed);
            run::<64, 1280, 512, T>("roomy", layout, kind, sample, timed);
            run::<128, 640, 1024, T>("compact", layout, kind, sample, timed);
            run::<128, 2560, 1024, T>("roomy", layout, kind, sample, timed);
        }
    }
}

fn main() {
    let argument = std::env::args()
        .nth(1)
        .expect("pass --check or a sample number");
    let timed = argument != "--check";
    let sample = if timed { argument.parse().unwrap() } else { 0 };
    println!("sample,n,scalar,capacity,layout,ordering,operation,input_capacity,factor_capacity,input_nnz,ordered_nnz,factor_nnz,iterations,elapsed_ns");
    suite::<f32>(sample, timed);
    suite::<f64>(sample, timed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_fixtures_match_coordinates_and_solve_in_original_order() {
        suite::<f32>(0, false);
        suite::<f64>(1, false);
    }
}
