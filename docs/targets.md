# Target support and evidence

This page separates what is checked from what is only intended. A successful
host build does not prove that a firmware image executes, and QEMU execution
does not prove peripheral behavior, cycle timing, floating-point throughput, or
stack usage on a particular board.

## Evidence levels

| Level | What it demonstrates | What it does not demonstrate |
| --- | --- | --- |
| Host test | Numerical tests, rustdoc examples, and optional Eigen parity tests pass on the CI host. | MCU instruction selection, peripheral behavior, or device timing. |
| Cross-target compile | The `no_std` crate type-checks for the listed target and linker-facing code remains buildable. | Startup/linker integration, runtime behavior, or resource budgets. |
| QEMU execution | The checked firmware harness starts and completes its algebra smoke tests under the named emulated CPU. | Real hardware peripherals, board clocks, cache/memory effects, exact stack use, and electrical behavior. |
| Real hardware | A maintained board test measures the stated behavior on the stated board/toolchain. | Other boards, compiler flags, or workloads not covered by that test. |

## Current matrix

| Target or class | Compile check | QEMU execution | Real hardware status |
| --- | --- | --- | --- |
| Native host (x86-64 CI runner) | `cargo check`, tests, rustdoc, and Clippy in CI | Not applicable | Host-only measurements; benchmark reports record the runner and commit. |
| `thumbv6m-none-eabi` (Cortex-M0/M0+) | CI `cargo check --target ... --no-default-features` | No dedicated execution gate currently | No board test in this repository. |
| `thumbv7em-none-eabihf` (Cortex-M4F-class) | CI `cargo check --target ... --no-default-features` | `qemu-tests/run_cortex_m.sh` executes the smoke harness on QEMU's MPS2 Cortex-M4 machine | No STM32 board test or cycle claim. |
| `riscv32imc-unknown-none-elf` (ESP32-C3-class ISA) | CI `cargo check --target ... --no-default-features` | `qemu-tests/run_riscv32.sh` executes the scalar smoke harness on QEMU `virt` | No ESP32 board test; this is not evidence for any specific ESP32 peripheral or clock configuration. |
| `aarch64-unknown-none` (AArch64/NEON class) | CI `cargo check --target ... --no-default-features` | `qemu-tests/run_aarch64.sh` executes the AArch64 smoke harness | No board test or NEON cycle claim. |
| `wasm32-unknown-unknown` | CI `cargo check --target ... --no-default-features` | Not currently exercised in a WASM runtime | No browser, WASI, or device integration claim. |

The QEMU harnesses exercise representative matrix operations, decompositions,
and target-selected kernels; they are smoke tests rather than a complete API
or numerical qualification suite. The CI stack-usage gate applies only to the
harnesses and configured linker images, not to every possible user matrix
shape or application call graph.

## Storage and placement

`Matrix<M, N, T>` and the bounded sparse types store their elements inline in
the value. “Inline” describes the representation, not a required placement:
the owner can be a local variable, a `static`, a field in a firmware state
struct, an arena, or a caller-managed external buffer through `Map`/
`StridedMap`. The core does not require `alloc`, but callers remain responsible
for choosing a placement that fits their stack, static-RAM, DMA, and lifetime
budgets.

## Reproducing the checks

From the repository root:

```sh
cargo check --target thumbv7em-none-eabihf --no-default-features
qemu-tests/run_cortex_m.sh
cargo check --target riscv32imc-unknown-none-elf --no-default-features
qemu-tests/run_riscv32.sh
cargo check --target aarch64-unknown-none --no-default-features
qemu-tests/run_aarch64.sh
```

Install the target toolchains and QEMU packages as described by the CI
workflow. For STM32, ESP32, or another MCU, treat the compile/QEMU rows as
portable-core evidence only; add a board-specific smoke test and measurements
before making a hardware-support claim.
