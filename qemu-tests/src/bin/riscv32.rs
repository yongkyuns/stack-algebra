#![no_main]
#![no_std]

use core::panic::PanicInfo;

use riscv_rt::entry;
use stack_algebra::{Matrix, Vector};

const UART_BASE: usize = 0x1000_0000;
const TEST_EXIT: *mut u32 = 0x0010_0000 as *mut u32;

fn write_byte(byte: u8) {
    unsafe {
        while core::ptr::read_volatile((UART_BASE + 5) as *const u8) & 0x20 == 0 {}
        core::ptr::write_volatile(UART_BASE as *mut u8, byte);
    }
}

fn write_str(message: &str) {
    for byte in message.bytes() {
        write_byte(byte);
    }
}

fn exit_qemu(code: u32) -> ! {
    unsafe {
        core::ptr::write_volatile(TEST_EXIT, code);
    }
    loop {
        core::hint::spin_loop();
    }
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

    write_str("stack-algebra qemu riscv32: PASS\n");
    exit_qemu(0x5555);
}

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {
    write_str("stack-algebra qemu riscv32: FAIL\n");
    exit_qemu(0x3333);
}
