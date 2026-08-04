#![no_main]
#![no_std]

use core::panic::PanicInfo;

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};
use stack_algebra::{Matrix, Vector};

const STACK_PAINT: u32 = 0xCCCC_CCCC;

unsafe extern "C" {
    static _stack_end: u8;
    static _stack_start: u8;
}

#[entry]
fn main() -> ! {
    let lhs = Matrix::<2, 3, f32>::from_rows([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]);
    let rhs = Matrix::<3, 2, f32>::from_rows([[0.5, 2.0], [-3.0, 4.0], [7.0, -1.0]]);
    let expected = Matrix::<2, 2, f32>::from_rows([[27.5, -9.0], [-55.0, 34.0]]);
    assert_eq!(lhs * rhs, expected);

    let matrix =
        Matrix::<3, 3, f32>::from_rows([[6.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.0, 4.0, 9.0]]);
    let rhs = Vector::<3, f32>::from_columns([[1.0, 2.0, 3.0]]);
    let solution = matrix.partial_piv_lu().solve(&rhs);
    let reconstructed = matrix * solution;
    for index in 0..3 {
        assert!((reconstructed[index] - rhs[index]).abs() < 1e-5);
    }

    let pivoted_matrix = Matrix::<2, 2, f32>::from_rows([[0.0, 1.0], [1.0, 2.0]]);
    let pivoted_rhs = Vector::<2, f32>::from_columns([[1.0, 3.0]]);
    let pivoted_solution = pivoted_matrix
        .ldlt()
        .expect("pivoted LDLT should handle zero leading diagonal")
        .solve(&pivoted_rhs);
    let pivoted_reconstructed = pivoted_matrix * pivoted_solution;
    for index in 0..2 {
        assert!((pivoted_reconstructed[index] - pivoted_rhs[index]).abs() < 1e-5);
    }

    let no_pivot_matrix =
        Matrix::<3, 3, f32>::from_rows([[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]]);
    let no_pivot_rhs = Vector::<3, f32>::from_columns([[1.0, 2.0, 3.0]]);
    let no_pivot_solution = no_pivot_matrix
        .ldlt_no_pivot()
        .expect("stable system should factor without pivoting")
        .solve(&no_pivot_rhs);
    let no_pivot_reconstructed = no_pivot_matrix * no_pivot_solution;
    for index in 0..3 {
        assert!((no_pivot_reconstructed[index] - no_pivot_rhs[index]).abs() < 1e-5);
    }

    let stack_used = stack_watermark_used();
    let stack_budget = stack_capacity();
    assert!(stack_used <= stack_budget);
    hprintln!(
        "stack-algebra qemu cortex-m: PASS stack_used={} stack_budget={}",
        stack_used,
        stack_budget
    );
    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}

fn stack_capacity() -> usize {
    unsafe { (&_stack_start as *const u8 as usize) - (&_stack_end as *const u8 as usize) }
}

fn stack_watermark_used() -> usize {
    let stack_end = unsafe { &_stack_end as *const u8 as usize };
    let stack_start = unsafe { &_stack_start as *const u8 as usize };
    let mut first_used = stack_start;
    let mut address = stack_end;
    while address + core::mem::size_of::<u32>() <= stack_start {
        let value = unsafe { core::ptr::read_volatile(address as *const u32) };
        if value != STACK_PAINT {
            first_used = address;
            break;
        }
        address += core::mem::size_of::<u32>();
    }
    stack_start - first_used
}

#[panic_handler]
fn panic(info: &PanicInfo<'_>) -> ! {
    let _ = hprintln!("stack-algebra qemu cortex-m: FAIL: {}", info);
    debug::exit(debug::EXIT_FAILURE);
    loop {}
}
