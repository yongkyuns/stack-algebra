//! Fit y = a*x + b to five samples using a caller-owned column-major buffer.
//!
//! Each design row is [x, 1]; the solution is [slope, intercept]. The fixed
//! observations are not exactly on a line, so the fitted residual is nonzero.
//! This unweighted fit treats x as known and minimizes squared errors in y.
//!
//! Run: `cargo run --example mapped_least_squares --no-default-features`.
//! The host executable uses std for printing; the algebra uses the no_std core.
//! Append `-- --csv` to export full-precision values from the same calculation.

use stack_algebra::{ColPivHouseholderQr, DecompositionError, Map, Matrix};

// ANCHOR: inputs
pub(crate) const DESIGN_STORAGE: [f64; 10] = [
    0.0, 1.0, 2.0, 3.0, 4.0, // column 0: x
    1.0, 1.0, 1.0, 1.0, 1.0, // column 1: intercept coefficient
];
pub(crate) const OBSERVATIONS: [f64; 5] = [1.1, 2.9, 5.2, 6.8, 9.0];
// ANCHOR_END: inputs

// ANCHOR: fit
// This helper belongs to the example, not the library API. Tests import it.
pub(crate) fn fit_line(
    design: &Map<'_, 5, 2, f64>,
    observations: &Matrix<5, 1, f64>,
) -> Result<Matrix<2, 1, f64>, DecompositionError> {
    // Read the borrowed design into QR's own inline factor storage. There is
    // no separate owned input matrix, and the caller's buffer is not modified.
    let factor = ColPivHouseholderQr::try_decompose_view(design)?;
    factor.try_solve_least_squares(observations)
}
// ANCHOR_END: fit

#[cfg(not(test))]
fn main() -> std::io::Result<()> {
    let csv = tutorial_output::csv_requested()?;
    // ANCHOR: borrow
    let design = Map::<5, 2, f64>::from_slice(&DESIGN_STORAGE).expect("five [x, 1] rows");
    let observations = Matrix::<5, 1, f64>::from_columns([OBSERVATIONS]);
    // ANCHOR_END: borrow
    // ANCHOR: solve
    let coefficients = fit_line(&design, &observations).expect("finite, full-rank line fit");
    // ANCHOR_END: solve

    // ANCHOR: evaluate
    // Evaluate A * [a, b]^T directly from the same map into caller-owned output.
    let mut fitted = Matrix::<5, 1, f64>::zeros();
    design.matvec_into(&coefficients, &mut fitted);
    let residuals = observations - fitted; // observed minus fitted
    // ANCHOR_END: evaluate
    let residual_norm = residuals.norm();

    if csv {
        println!("x,observed_y,fitted_y,residual,slope,intercept,residual_norm,sample_count");
    } else {
        println!(
            "slope = {:.3}, intercept = {:.3}",
            coefficients[(0, 0)],
            coefficients[(1, 0)]
        );
        println!("x observed_y fitted_y residual");
    }
    for row in 0..5 {
        if csv {
            tutorial_output::csv_row(&[
                design[(row, 0)],
                observations[(row, 0)],
                fitted[(row, 0)],
                residuals[(row, 0)],
                coefficients[(0, 0)],
                coefficients[(1, 0)],
                residual_norm,
                OBSERVATIONS.len() as f64,
            ]);
        } else {
            println!(
                "{:.1} {:.3} {:.3} {:.3}",
                design[(row, 0)],
                observations[(row, 0)],
                fitted[(row, 0)],
                residuals[(row, 0)]
            );
        }
    }
    if !csv {
        println!("residual norm = {:.3}", residual_norm);
    }
    Ok(())
}

#[cfg(not(test))]
#[path = "support/tutorial_output.rs"]
mod tutorial_output;
