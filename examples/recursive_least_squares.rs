//! Learn a quadratic one observation at a time with fixed-memory square-root RLS.
//! cargo run --example recursive_least_squares --no-default-features
//! --csv exports the learning history; --curve-csv exports Rust-evaluated snapshots.

use stack_algebra::Matrix;

#[path = "support/quadratic_rls.rs"]
mod rls;
use rls::{QuadraticRls, Update};

// The host driver shares only the deterministic data, not the batch solver.
// The standalone estimator module has no dependency on these inputs or std.
#[path = "support/quadratic_data.rs"]
mod data;
#[path = "support/tutorial_output.rs"]
mod tutorial_output;

// ANCHOR: configuration
const INPUT_SCALE: f64 = 3.0;
const PRIOR_PRECISION: f64 = 0.01;
const FORGETTING: f64 = 1.0;
const SNAPSHOTS: [usize; 4] = [5, 10, 20, 40];
// ANCHOR_END: configuration

type Record = (usize, f64, f64, Update<f64>);

fn main() -> std::io::Result<()> {
    let curve_csv = std::env::args().skip(1).eq(["--curve-csv"]);
    let csv = curve_csv || tutorial_output::csv_requested()?;
    // ANCHOR: initialize
    let mut estimator =
        QuadraticRls::<f64>::new(INPUT_SCALE, PRIOR_PRECISION, FORGETTING, Matrix::zeros())
            .expect("valid fixed configuration");
    // ANCHOR_END: initialize
    // Host-only reporting storage. The embedded estimator stores no sample history.
    let mut history: Vec<Record> = Vec::new();
    let mut snapshots = Vec::new();
    // ANCHOR: stream
    for step in 0..data::SAMPLE_COUNT {
        // A fixed permutation covers the input range early; it changes no (x,y) pair.
        let source_row = (17 * step) % data::SAMPLE_COUNT;
        let x = data::sample_x(source_row);
        let y = data::reference_y(x) + data::NOISE_SCALE * data::NOISE[source_row];
        let update = estimator.update(x, y).expect("finite fixture update");
        history.push((source_row, x, y, update));
        if SNAPSHOTS.contains(&estimator.samples()) {
            snapshots.push((estimator.samples(), estimator));
        }
    }
    // ANCHOR_END: stream
    if curve_csv {
        println!("step,x,fitted_y,reference_y,a,b,c,sample_count");
        // ANCHOR: snapshots
        for (step, snapshot) in &snapshots {
            let theta = snapshot.coefficients();
            for row in 0..data::CURVE_COUNT {
                let x = data::curve_x(row);
                let fitted_y = snapshot.predict(x).expect("finite curve evaluation");
                // No plotting-grid matrix or observation replay is needed by update().
                tutorial_output::csv_row(&[
                    *step as f64,
                    x,
                    fitted_y,
                    data::reference_y(x),
                    theta[(0, 0)],
                    theta[(1, 0)],
                    theta[(2, 0)],
                    (snapshots.len() * data::CURVE_COUNT) as f64,
                ]);
            }
        }
        // ANCHOR_END: snapshots
        return Ok(());
    }
    let bytes32 = core::mem::size_of::<QuadraticRls<f32>>();
    let bytes64 = core::mem::size_of::<QuadraticRls<f64>>();
    if csv {
        println!("step,source_row,x,observed_y,reference_y,prediction_before,innovation,a,b,c,final_fitted_y,final_residual,reference_a,reference_b,reference_c,input_scale,prior_precision,forgetting,sample_count,storage_bytes_f32,storage_bytes_f64");
    } else {
        println!(
            "samples = {}, forgetting = {:.3}, prior precision = {:.6}",
            history.len(),
            FORGETTING,
            PRIOR_PRECISION
        );
        println!("estimator bytes: f32 = {bytes32}, f64 = {bytes64}");
        println!("step x observed_y prediction_before innovation a b c");
    }
    for (index, (source_row, x, y, update)) in history.iter().enumerate() {
        let theta = update.coefficients;
        let fitted = estimator.predict(*x).expect("finite final evaluation");
        if csv {
            tutorial_output::csv_row(&[
                (index + 1) as f64,
                *source_row as f64,
                *x,
                *y,
                data::reference_y(*x),
                update.prediction_before,
                update.innovation,
                theta[(0, 0)],
                theta[(1, 0)],
                theta[(2, 0)],
                fitted,
                *y - fitted,
                data::REFERENCE_COEFFICIENTS[0],
                data::REFERENCE_COEFFICIENTS[1],
                data::REFERENCE_COEFFICIENTS[2],
                INPUT_SCALE,
                PRIOR_PRECISION,
                FORGETTING,
                history.len() as f64,
                bytes32 as f64,
                bytes64 as f64,
            ]);
        } else {
            println!(
                "{} {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} {:.3}",
                index + 1,
                x,
                y,
                update.prediction_before,
                update.innovation,
                theta[(0, 0)],
                theta[(1, 0)],
                theta[(2, 0)]
            );
        }
    }
    Ok(())
}
