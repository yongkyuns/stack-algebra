//! Estimate one-dimensional position and velocity from scalar position samples.
//!
//! This is a small linear Kalman filter for learning the matrix API, not an
//! EKF/ESKF or a navigation implementation. State is [position (m), velocity (m/s)].
//! Samples arrive every second. Acceleration is zero-mean, constant within each
//! interval, independent between intervals, and has variance 0.04 (m/s^2)^2.
//! Position noise has variance 0.25 m^2 and is independent of process noise.
//! The fixed sample sequence illustrates motion near 1 m/s; no RNG is required.
//!
//! Run: `cargo run --example kalman_1d --no-default-features`.
//! The host executable uses std for printing; the algebra uses the no_std core.
//! Append `-- --csv` to export the inputs, state, covariance, and diagnostics.

use stack_algebra::Matrix;

// ANCHOR: inputs
pub(crate) const DT: f32 = 1.0;
pub(crate) const ACCELERATION_VARIANCE: f32 = 0.04;
pub(crate) const MEASUREMENT_VARIANCE: f32 = 0.25;
pub(crate) const MEASUREMENTS: [f32; 10] = [1.2, 1.8, 3.1, 3.9, 5.2, 5.9, 7.1, 8.0, 8.8, 10.1];
// ANCHOR_END: inputs

// These helpers belong to the example, not the library API. The integration
// tests import this file so they exercise the same implementation as main.
// ANCHOR: predict
pub(crate) fn predict(
    state: &mut Matrix<2, 1, f32>,
    covariance: &mut Matrix<2, 2, f32>,
    dt: f32,
    acceleration_variance: f32,
) {
    let transition = Matrix::<2, 2, f32>::from_rows([[1.0, dt], [0.0, 1.0]]);
    let acceleration = Matrix::<2, 1, f32>::from_rows([[0.5 * dt * dt], [dt]]);
    // Q = G G^T sigma_a^2 for the discrete acceleration model above.
    let process_noise = (acceleration * acceleration.transpose()) * acceleration_variance;

    *state = transition * *state;
    *covariance = transition * *covariance * transition.transpose() + process_noise;
}
// ANCHOR_END: predict

// Returns (innovation, innovation variance, gain) for optional reporting.
// These are the exact intermediate values used below, not a second calculation.
pub(crate) fn update_position(
    state: &mut Matrix<2, 1, f32>,
    covariance: &mut Matrix<2, 2, f32>,
    position: f32,
    measurement_variance: f32,
) -> (f32, f32, Matrix<2, 1, f32>) {
    // ANCHOR: gain
    // H = [1, 0]: innovation variance is scalar and P H^T is column zero.
    // Inputs in this teaching example have valid covariance and positive R.
    let innovation = position - state[(0, 0)];
    let innovation_variance = covariance[(0, 0)] + measurement_variance;
    let gain = Matrix::<2, 1, f32>::from_rows([
        [covariance[(0, 0)] / innovation_variance],
        [covariance[(1, 0)] / innovation_variance],
    ]);
    state.axpy_in_place(innovation, &gain); // state += gain * innovation
    // ANCHOR_END: gain

    // ANCHOR: covariance
    // Joseph form: P = (I - K H) P (I - K H)^T + K R K^T.
    // Readable 2x2 expressions are intentional; this is not a tuned hot loop.
    let residual_map =
        Matrix::<2, 2, f32>::from_rows([[1.0 - gain[(0, 0)], 0.0], [-gain[(1, 0)], 1.0]]);
    *covariance = residual_map * *covariance * residual_map.transpose()
        + (gain * gain.transpose()) * measurement_variance;
    // ANCHOR_END: covariance
    (innovation, innovation_variance, gain)
}

#[cfg(not(test))]
fn main() -> std::io::Result<()> {
    let csv = tutorial_output::csv_requested()?;
    // ANCHOR: initial
    // At t = 0, both estimates are zero, with unit variance and no correlation.
    let mut state = Matrix::<2, 1, f32>::zeros();
    let mut covariance = Matrix::<2, 2, f32>::eye();
    // ANCHOR_END: initial
    let initial_state = state;
    let initial_covariance = covariance;

    if csv {
        println!(
            "time_s,measured_m,dt_s,acceleration_variance,measurement_variance,initial_position_m,initial_velocity_m_s,initial_p00,initial_p01,initial_p10,initial_p11,predicted_position_m,predicted_velocity_m_s,predicted_p00,predicted_p01,predicted_p10,predicted_p11,position_m,velocity_m_s,p00,p01,p10,p11,innovation_m,innovation_variance,position_gain,velocity_gain,sample_count"
        );
    } else {
        println!("time_s measured_m position_m velocity_m_s");
    }
    for (step, position) in MEASUREMENTS.iter().copied().enumerate() {
        // ANCHOR: cycle
        predict(&mut state, &mut covariance, DT, ACCELERATION_VARIANCE);
        let predicted_state = state;
        let predicted_covariance = covariance;
        let (innovation, innovation_variance, gain) =
            update_position(&mut state, &mut covariance, position, MEASUREMENT_VARIANCE);
        // ANCHOR_END: cycle
        if csv {
            let values = [
                (step + 1) as f32 * DT,
                position,
                DT,
                ACCELERATION_VARIANCE,
                MEASUREMENT_VARIANCE,
                initial_state[(0, 0)],
                initial_state[(1, 0)],
                initial_covariance[(0, 0)],
                initial_covariance[(0, 1)],
                initial_covariance[(1, 0)],
                initial_covariance[(1, 1)],
                predicted_state[(0, 0)],
                predicted_state[(1, 0)],
                predicted_covariance[(0, 0)],
                predicted_covariance[(0, 1)],
                predicted_covariance[(1, 0)],
                predicted_covariance[(1, 1)],
                state[(0, 0)],
                state[(1, 0)],
                covariance[(0, 0)],
                covariance[(0, 1)],
                covariance[(1, 0)],
                covariance[(1, 1)],
                innovation,
                innovation_variance,
                gain[(0, 0)],
                gain[(1, 0)],
                MEASUREMENTS.len() as f32,
            ];
            tutorial_output::csv_row(&values.map(f64::from));
        } else {
            println!(
                "{:.0} {:.3} {:.3} {:.3}",
                (step + 1) as f32 * DT,
                position,
                state[(0, 0)],
                state[(1, 0)]
            );
        }
    }
    Ok(())
}

#[cfg(not(test))]
#[path = "support/tutorial_output.rs"]
mod tutorial_output;
