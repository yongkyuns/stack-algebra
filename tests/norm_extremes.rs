use stack_algebra::Vector;

#[test]
fn f32_norm_avoids_subnormal_square_rounding() {
    let value = 3.0e-23_f32;
    let vector = Vector::<1, f32>::from_rows([[value]]);
    let actual = vector.norm();
    let relative_error = (actual / value.abs() - 1.0).abs();

    assert!(
        relative_error <= 8.0 * f32::EPSILON,
        "expected {}, got {actual} (relative error {relative_error})",
        value.abs()
    );
}

#[test]
fn f64_norm_avoids_subnormal_square_rounding() {
    let value = 2.0e-162_f64;
    let vector = Vector::<1, f64>::from_rows([[value]]);
    let actual = vector.norm();
    let relative_error = (actual / value.abs() - 1.0).abs();

    assert!(
        relative_error <= 8.0 * f64::EPSILON,
        "expected {}, got {actual} (relative error {relative_error})",
        value.abs()
    );
}

#[test]
fn norm_remains_stable_when_naive_square_overflows() {
    let value = f64::MAX / 4.0;
    let vector = Vector::<2, f64>::from_rows([[value], [value]]);
    let actual = vector.norm();
    let expected = value * 2.0_f64.sqrt();
    let relative_error = (actual / expected - 1.0).abs();

    assert!(relative_error <= 8.0 * f64::EPSILON);
    assert!(actual.is_finite());
}
