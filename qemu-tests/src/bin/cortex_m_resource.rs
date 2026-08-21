#![no_main]
#![no_std]

use core::panic::PanicInfo;

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};

#[path = "../workloads.rs"]
mod workloads;

const STACK_PAINT: u32 = 0xCCCC_CCCC;

unsafe extern "C" {
    static _stack_end: u8;
    static _stack_start: u8;
}

#[entry]
fn main() -> ! {
    let (workload, object_bytes) = run_selected_workload();
    let stack_used = stack_watermark_used();
    let stack_budget = stack_capacity();
    assert!(stack_used <= stack_budget);
    hprintln!(
        "stack-algebra resource cortex-m: PASS workload={} stack_used={} stack_budget={} object_bytes={}",
        workload,
        stack_used,
        stack_budget,
        object_bytes
    );
    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}

#[cfg(feature = "resource-baseline")]
fn run_selected_workload() -> (&'static str, usize) {
    ("baseline", workloads::baseline())
}

#[cfg(all(not(feature = "resource-baseline"), feature = "resource-dense3"))]
fn run_selected_workload() -> (&'static str, usize) {
    ("dense3", workloads::dense3())
}

#[cfg(all(
    not(feature = "resource-baseline"),
    not(feature = "resource-dense3"),
    feature = "resource-dense6"
))]
fn run_selected_workload() -> (&'static str, usize) {
    ("dense6", workloads::dense6())
}

#[cfg(all(
    not(feature = "resource-baseline"),
    not(feature = "resource-dense3"),
    not(feature = "resource-dense6"),
    feature = "resource-dense15"
))]
fn run_selected_workload() -> (&'static str, usize) {
    ("dense15", workloads::dense15())
}

#[cfg(all(
    not(feature = "resource-baseline"),
    not(feature = "resource-dense3"),
    not(feature = "resource-dense6"),
    not(feature = "resource-dense15"),
    feature = "resource-sparse"
))]
fn run_selected_workload() -> (&'static str, usize) {
    ("sparse", workloads::sparse())
}

#[cfg(all(
    not(feature = "resource-baseline"),
    not(feature = "resource-dense3"),
    not(feature = "resource-dense6"),
    not(feature = "resource-dense15"),
    not(feature = "resource-sparse"),
    feature = "resource-block-sparse"
))]
fn run_selected_workload() -> (&'static str, usize) {
    ("block_sparse", workloads::block_sparse())
}

#[cfg(not(any(
    feature = "resource-baseline",
    feature = "resource-dense3",
    feature = "resource-dense6",
    feature = "resource-dense15",
    feature = "resource-sparse",
    feature = "resource-block-sparse"
)))]
compile_error!("enable exactly one resource-* workload feature");

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
    let _ = hprintln!("stack-algebra resource cortex-m: FAIL: {}", info);
    debug::exit(debug::EXIT_FAILURE);
    loop {}
}
