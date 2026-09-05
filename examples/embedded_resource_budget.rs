use stack_algebra::{Matrix, MatrixBuf};

fn main() {
    // Long-lived storage can be budgeted at compile time before selecting placement.
    const STATE_BYTES: usize = Matrix::<15, 1, f32>::storage_bytes();
    const COVARIANCE_BYTES: usize = Matrix::<15, 15, f32>::storage_bytes();
    const BOUNDED_WORK_BYTES: usize = MatrixBuf::<32, 32, f32>::storage_bytes();

    // Keep these as application-level policy values rather than library guarantees.
    const PERSISTENT_BUDGET: usize = 4096;
    const WORKSPACE_BUDGET: usize = 8192;

    assert!(STATE_BYTES + COVARIANCE_BYTES <= PERSISTENT_BUDGET);
    assert!(BOUNDED_WORK_BYTES <= WORKSPACE_BUDGET);
}
