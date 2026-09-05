#![no_main]
#![no_std]

use core::panic::PanicInfo;
use core::sync::atomic::{compiler_fence, Ordering};

use cortex_m::peripheral::DWT;
use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};

#[path = "../workloads.rs"]
mod workloads;

const ITERATIONS: usize = 32;

#[entry]
fn main() -> ! {
    let mut peripherals = cortex_m::Peripherals::take().expect("core peripherals available once");
    peripherals.DCB.enable_trace();
    assert!(
        DWT::has_cycle_counter(),
        "Cortex-M DWT cycle counter required"
    );
    DWT::unlock();
    peripherals.DWT.enable_cycle_counter();

    hprintln!(
        "stack-algebra cortex-m bench: BEGIN iterations={} counter_bits=32",
        ITERATIONS
    );
    measure("baseline", workloads::baseline);
    measure("dense3", workloads::dense3);
    measure("dense6", workloads::dense6);
    measure("dense6_f64", workloads::dense6_f64);
    measure("dense15", workloads::dense15);
    measure("sparse", workloads::sparse);
    measure("block_sparse", workloads::block_sparse);
    hprintln!("stack-algebra cortex-m bench: PASS");

    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}

fn measure(label: &'static str, mut workload: impl FnMut() -> usize) {
    let mut minimum = u32::MAX;
    let mut object_bytes = 0usize;

    for _ in 0..ITERATIONS {
        compiler_fence(Ordering::SeqCst);
        let start = DWT::cycle_count();
        object_bytes = workload();
        compiler_fence(Ordering::SeqCst);
        let elapsed = DWT::cycle_count().wrapping_sub(start);
        minimum = minimum.min(elapsed);
    }

    hprintln!(
        "stack-algebra cortex-m bench: workload={} cycles_min={} iterations={} object_bytes={}",
        label,
        minimum,
        ITERATIONS,
        object_bytes
    );
}

#[panic_handler]
fn panic(info: &PanicInfo<'_>) -> ! {
    let _ = hprintln!("stack-algebra cortex-m bench: FAIL: {}", info);
    debug::exit(debug::EXIT_FAILURE);
    loop {}
}
