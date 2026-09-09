//! Fit y = a*x^2 + b*x + c to 40 noisy synthetic observations with borrowed storage.
//!
//! Each design row is [x^2, x, 1]; the solution is [a, b, c]. The reference curve
//! generates the observations but is never passed to the fitting helper. This
//! unweighted fit treats x as known and minimizes squared errors in y.
//!
//! Run: `cargo run --example mapped_least_squares --no-default-features`.
//! The host executable uses std for printing; the algebra uses the no_std core.
//! `-- --csv` exports observations and diagnostics; `-- --curve-csv` exports
//! the fitted and synthetic reference curves evaluated by Rust on a dense grid.

use stack_algebra::{ColPivHouseholderQr, DecompositionError, Map, Matrix};

#[path = "support/quadratic_data.rs"]
mod data;
pub(crate) use data::{
    curve_x, reference_y, sample_x, CURVE_COUNT, NOISE, NOISE_SCALE, REFERENCE_COEFFICIENTS,
    SAMPLE_COUNT,
};

// ANCHOR: observations
pub(crate) fn make_observations() -> Matrix<SAMPLE_COUNT, 1, f64> {
    Matrix::from_fn(|row, _| reference_y(sample_x(row)) + NOISE_SCALE * NOISE[row])
}
// ANCHOR_END: observations

// ANCHOR: design
pub(crate) fn design_storage() -> [f64; SAMPLE_COUNT * 3] {
    let mut storage = [0.0; SAMPLE_COUNT * 3];
    for row in 0..SAMPLE_COUNT {
        let x = sample_x(row);
        storage[row] = x * x;
        storage[SAMPLE_COUNT + row] = x;
        storage[2 * SAMPLE_COUNT + row] = 1.0;
    }
    storage
}
// ANCHOR_END: design

// ANCHOR: fit
// Example helper, not public library API. It receives no reference coefficients.
pub(crate) fn fit_quadratic(
    design: &Map<'_, SAMPLE_COUNT, 3, f64>,
    observations: &Matrix<SAMPLE_COUNT, 1, f64>,
) -> Result<Matrix<3, 1, f64>, DecompositionError> {
    // QR owns its inline factor storage; borrowing avoids an extra input matrix.
    let factor = ColPivHouseholderQr::try_decompose_view(design)?;
    factor.try_solve_least_squares(observations)
}
// ANCHOR_END: fit

pub(crate) fn curve_design() -> Matrix<CURVE_COUNT, 3, f64> {
    Matrix::from_fn(|row, column| match column {
        0 => curve_x(row).powi(2),
        1 => curve_x(row),
        _ => 1.0,
    })
}

#[cfg(not(test))]
fn main() -> std::io::Result<()> {
    let curve_csv = std::env::args().skip(1).eq(["--curve-csv"]);
    let csv = curve_csv || tutorial_output::csv_requested()?;
    // ANCHOR: borrow
    let storage = design_storage();
    let design = Map::<SAMPLE_COUNT, 3, f64>::from_slice(&storage).expect("40 [x^2, x, 1] rows");
    let observations = make_observations();
    // ANCHOR_END: borrow
    // ANCHOR: solve
    let coefficients =
        fit_quadratic(&design, &observations).expect("finite, full-rank quadratic fit");
    // ANCHOR_END: solve

    // ANCHOR: evaluate
    let mut fitted = Matrix::<SAMPLE_COUNT, 1, f64>::zeros();
    design.matvec_into(&coefficients, &mut fitted);
    let residuals = observations - fitted;
    let residual_norm = residuals.norm();
    // ANCHOR_END: evaluate

    // ANCHOR: curve
    // Evaluate the fitted polynomial on 201 points with stack-algebra, not Python.
    // These extra points draw the curve; they are not extra observations in the fit.
    let grid_design = curve_design();
    let curve = grid_design * coefficients;
    // ANCHOR_END: curve

    if curve_csv {
        println!("x,fitted_y,reference_y,sample_count");
        for row in 0..CURVE_COUNT {
            let x = curve_x(row);
            tutorial_output::csv_row(&[x, curve[(row, 0)], reference_y(x), CURVE_COUNT as f64]);
        }
        return Ok(());
    }
    if csv {
        println!("x,observed_y,reference_y,noise,fitted_y,residual,a,b,c,reference_a,reference_b,reference_c,residual_norm,sample_count");
    } else {
        println!(
            "a = {:.3}, b = {:.3}, c = {:.3}",
            coefficients[(0, 0)],
            coefficients[(1, 0)],
            coefficients[(2, 0)]
        );
        println!("x observed_y fitted_y residual");
    }
    for (row, &noise) in NOISE.iter().enumerate() {
        let x = design[(row, 1)];
        if csv {
            tutorial_output::csv_row(&[
                x,
                observations[(row, 0)],
                reference_y(x),
                NOISE_SCALE * noise,
                fitted[(row, 0)],
                residuals[(row, 0)],
                coefficients[(0, 0)],
                coefficients[(1, 0)],
                coefficients[(2, 0)],
                REFERENCE_COEFFICIENTS[0],
                REFERENCE_COEFFICIENTS[1],
                REFERENCE_COEFFICIENTS[2],
                residual_norm,
                SAMPLE_COUNT as f64,
            ]);
        } else {
            println!(
                "{:.3} {:.3} {:.3} {:.3}",
                x,
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
