# Platforms and embedded use

`stack-algebra` is designed so the fixed-size core can be used in `no_std`
programs without requiring a heap. This page explains what is currently tested
and, just as importantly, what those tests do **not** prove about a particular
board or application.

## What "supported" means here

There are several useful levels of confidence for an embedded numerical
library:

| Check | What it tells you | What it does not tell you |
| --- | --- | --- |
| Host tests | Numerical behavior and API examples work on the CI host | MCU timing, linker layout, or device-specific behavior |
| Cross-target compile | The crate type-checks for a target without `std` | That a complete firmware image boots or fits |
| QEMU smoke test | Representative algebra runs on an emulated CPU | Real FPU throughput, cache effects, interrupts, peripherals, or exact stack use |
| Real-board measurement | The tested workload behaves as measured on that board/toolchain | Performance on other boards or workloads |

This distinction matters because linear algebra can compile cleanly while still
being a poor fit for a specific firmware stack or timing budget.

## Current target checks

| Target / class | Current repository check | Practical interpretation |
| --- | --- | --- |
| x86-64 host | Full tests, rustdoc, formatting/Clippy, benchmarks | Primary host development and measurement environment |
| `thumbv6m-none-eabi` | `cargo check --no-default-features` | Portable core type-checks for Cortex-M0/M0+-class targets |
| `thumbv7em-none-eabihf` | Cross-compile + Cortex-M4 QEMU smoke test | Representative Cortex-M4F-class execution path is exercised |
| `riscv32imc-unknown-none-elf` | Cross-compile + RISC-V QEMU smoke test | Portable scalar path is exercised on a RISC-V32 target class |
| `aarch64-unknown-none` | Cross-compile + AArch64 QEMU smoke test | AArch64 path and representative kernels are exercised |
| `wasm32-unknown-unknown` | Cross-target compile | Core remains buildable for WASM without a runtime-specific claim |

These checks are useful portability evidence, not a promise of cycle counts or
memory use on a particular STM32, ESP32, or other board.

## Memory placement

A common misunderstanding is that "stack allocated" means every matrix must be
a large local variable. The important property is that the storage is **inline
and bounded**; placement follows the owner.

For example, a matrix or factor can live:

- as a small local value;
- as a field in a long-lived estimator/controller state object;
- in static memory;
- inside another preallocated workspace;
- in caller-owned memory borrowed through `Map` or `StridedMap`.

For large fixed workspaces, long-lived state or static placement may be more
appropriate than a deep call-stack frame.

Many bounded types expose `storage_bytes()` as a `const fn`, which is useful for
budgeting the representation itself:

```rust
use stack_algebra::{Matrix, MatrixBuf};

const STATE_BYTES: usize = Matrix::<15, 1, f32>::storage_bytes();
const COV_BYTES: usize = Matrix::<15, 15, f32>::storage_bytes();
const WORK_BYTES: usize = MatrixBuf::<32, 32, f32>::storage_bytes();
```

This does not replace whole-program stack analysis. Function frames, temporary
values, interrupts, RTOS stacks, and compiler decisions still matter.

## Choosing `f32` or `f64` on embedded targets

`f32` is usually the first choice on MCUs because it uses half the storage of
`f64` and often maps better to the available FPU. That is a starting point, not
a rule.

Use `f64` when the numerical problem needs it and the target can afford the
cost. Conditioning, covariance scale, accumulated error, and solver behavior
can matter more than the arithmetic width alone.

The best decision comes from testing the actual algorithm with representative
inputs on the actual target.

## External and DMA-owned buffers

When a peripheral, driver, generated kernel, or FFI layer owns the data, avoid
copying it merely to satisfy the algebra API. Use:

- `Map` / `MapMut` for contiguous column-major data;
- `StridedMap` / `StridedMapMut` for padded, row-major, interleaved, or
  otherwise strided data;
- `Block` / `BlockMut` for a fixed region inside another matrix.

This keeps data ownership at the system boundary while still allowing dense
operations and factorizations to consume a fixed-shape view.

## What to validate on your board

Before treating a target as production-ready for your application, measure the
things the generic CI cannot know:

1. **Worst-case memory use** for the complete task/thread call graph.
2. **Execution time** for your matrix shapes and solver path.
3. **FPU behavior** for the chosen scalar type and compiler flags.
4. **Interrupt/RTOS interaction** if the operation runs in a real-time task.
5. **Buffer alignment and ownership** at DMA/FFI boundaries.
6. **Numerical behavior** with representative sensor/control/optimization data.

Host and QEMU checks are good at catching portability regressions. They are not
substitutes for this application-level validation.

## Reproducing the repository checks

From the repository root, examples include:

```sh
cargo check --target thumbv7em-none-eabihf --no-default-features
qemu-tests/run_cortex_m.sh

cargo check --target riscv32imc-unknown-none-elf --no-default-features
qemu-tests/run_riscv32.sh

cargo check --target aarch64-unknown-none --no-default-features
qemu-tests/run_aarch64.sh
```

For guidance on selecting matrix/storage types before target validation, see
[Choosing an API](api-usage.md).