#![no_main]
#![no_std]

use core::arch::global_asm;
use core::panic::PanicInfo;

use stack_algebra::{Matrix, Vector};

global_asm!(
    ".section .text._start, \"ax\"",
    ".global _start",
    ".type _start, %function",
    "_start:",
    "ldr x0, =__stack_top",
    "mov sp, x0",
    "mrs x1, cpacr_el1",
    "orr x1, x1, #(3 << 20)",
    "msr cpacr_el1, x1",
    "isb",
    "bl main",
    "1:",
    "wfe",
    "b 1b",
);

#[no_mangle]
pub extern "C" fn main() -> ! {
    let lhs = Matrix::<4, 4, f32>::from_fn(|row, column| (row + 2 * column + 1) as f32 / 7.0);
    let rhs = Matrix::<4, 4, f32>::from_fn(|row, column| (3 * row + column + 2) as f32 / 11.0);
    let product = lhs * rhs;
    let mut expected = Matrix::<4, 4, f32>::zeros();
    for column in 0..4 {
        for row in 0..4 {
            let mut value = 0.0;
            for shared in 0..4 {
                value += lhs[(row, shared)] * rhs[(shared, column)];
            }
            expected[(row, column)] = value;
        }
    }
    assert_close(&product, &expected, 1e-6);

    let vector = Vector::<4, f32>::from_columns([[1.0, 2.0, 3.0, 4.0]]);
    let matvec = lhs * vector;
    let mut expected_matvec = Vector::<4, f32>::zeros();
    for row in 0..4 {
        let mut value = 0.0;
        for column in 0..4 {
            value += lhs[(row, column)] * vector[column];
        }
        expected_matvec[row] = value;
    }
    assert_close(&matvec, &expected_matvec, 1e-6);

    let no_pivot_matrix =
        Matrix::<4, 4, f32>::from_fn(|row, column| if row == column { 5.0 } else { 0.25 });
    let no_pivot_rhs = Vector::<4, f32>::from_columns([[1.0, 2.0, 3.0, 4.0]]);
    let no_pivot_solution = no_pivot_matrix
        .ldlt_no_pivot()
        .expect("NEON no-pivot LDLT should factor the stable system")
        .solve(&no_pivot_rhs);
    assert_close(&(no_pivot_matrix * no_pivot_solution), &no_pivot_rhs, 1e-5);

    let pivoted_matrix = Matrix::<2, 2, f32>::from_rows([[0.0, 1.0], [1.0, 2.0]]);
    let pivoted_rhs = Vector::<2, f32>::from_columns([[1.0, 3.0]]);
    let pivoted_solution = pivoted_matrix
        .ldlt()
        .expect("NEON pivoted LDLT should handle zero leading diagonal")
        .solve(&pivoted_rhs);
    assert_close(&(pivoted_matrix * pivoted_solution), &pivoted_rhs, 1e-5);

    write_str("stack-algebra qemu aarch64: PASS\n");
    loop {
        core::hint::spin_loop();
    }
}

fn write_str(message: &str) {
    const UART: usize = 0x0900_0000;
    const UART_FLAG: usize = UART + 0x18;
    for byte in message.bytes() {
        unsafe {
            while core::ptr::read_volatile(UART_FLAG as *const u32) & (1 << 5) != 0 {}
            core::ptr::write_volatile(UART as *mut u32, byte as u32);
        }
    }
}

fn assert_close<const M: usize, const N: usize>(
    actual: &Matrix<M, N, f32>,
    expected: &Matrix<M, N, f32>,
    tolerance: f32,
) {
    for (actual, expected) in actual.as_slice().iter().zip(expected.as_slice()) {
        assert!((*actual - *expected).abs() <= tolerance);
    }
}

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {
    write_str("stack-algebra qemu aarch64: FAIL\n");
    loop {
        core::hint::spin_loop();
    }
}
