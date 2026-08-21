# Target qualification

`stack-algebra` treats target evidence as a resource contract, not as a claim
that cross-compilation alone proves embedded suitability. The qualification
workflow separates what CI can measure reproducibly from what must be measured
on physical hardware.

## Evidence levels

The project uses four evidence levels:

1. **Cross-compile evidence** — the crate builds for the supported bare-metal
   target triple.
2. **Emulator functional evidence** — representative code executes under QEMU
   and satisfies numerical/stack assertions.
3. **Emulator resource evidence** — isolated workload ELFs report code/static
   size and painted-stack high-water marks. These values are reproducible and
   useful for regression detection, but they are not hardware timing results.
4. **Physical-target evidence** — the same workload definitions execute on a
   named board/MCU at a controlled clock and report DWT cycle counts. This is
   the level required before `0.3` can make a real-device performance claim.

Do not combine these levels into one generic "embedded benchmark" number.

## CI Cortex-M resource report

The Build workflow uses the existing `thumbv7em-none-eabihf` Cortex-M test
configuration and QEMU `mps2-an386` machine. Each workload is compiled as an
isolated release ELF using the `qemu-tests` release profile:

- `opt-level = "z"`;
- LTO enabled;
- one codegen unit;
- debug information retained for inspectability.

Run the same report locally with:

```text
qemu-tests/qualify_cortex_m.sh
```

The script requires the Rust target, QEMU, `llvm-tools-preview`, and
`cargo-binutils`/`rust-size`. CI uploads both `resource-report.tsv` and
`resource-report.md` as the `cortex-m-resource-report` artifact and also writes
the Markdown table into the workflow job summary.

### Reported fields

- `text_bytes`, `data_bytes`, `bss_bytes`: values from `rust-size` for the
  isolated ELF;
- `*_delta`: difference from the baseline resource-probe ELF built with the
  same runtime and output path;
- `flash_bytes`: `text + data` for a simple comparable flash-footprint metric;
- `stack_used`: high-water mark from the `cortex-m-rt` painted stack after the
  workload returns;
- `stack_limit`: CI regression ceiling, currently 8192 bytes;
- `object_bytes`: sum of the principal matrices/factors/output buffers retained
  by the workload at its measurement point.

The ELF deltas are intentionally described as *workload deltas*, not exact
library-only section attribution. Link-time optimization, panic paths, and
formatting/runtime support can affect section layout.

## Qualification workloads

| workload | representative path |
| --- | --- |
| `baseline` | resource-probe runtime with no linear-algebra workload |
| `dense3` | 3x3 `f32` partial-pivot LU factor + solve + residual check |
| `dense6` | 6x6 `f32` matrix product followed by fused linear combination |
| `dense6_f64` | the same 6x6 product/fused path in `f64`, exposing double-precision resource cost on an M4F-class target |
| `dense15` | 15x15 `f32` matrix product followed by fused AXPY output |
| `sparse` | fixed-capacity 4x4 `f32` CSC Cholesky + solve + residual check |
| `block_sparse` | fixed-capacity 2x2-block CSC matrix-vector product |

The definitions live in `qemu-tests/src/workloads.rs` and are shared by the
resource probe and the physical-cycle harness. That prevents the CI resource
workload from drifting away from the code timed on hardware. In particular,
the paired `dense6`/`dense6_f64` cases make single- versus double-precision
resource differences explicit without treating emulator execution time as a
performance measurement.

## Physical Cortex-M cycle harness

`qemu-tests/src/bin/cortex_m_bench.rs` measures the shared workloads with the
Cortex-M DWT `CYCCNT`. Each workload runs 32 times and reports the minimum cycle
count. Semihosting output happens outside the timed region. The harness checks
that the DWT cycle counter exists, enables global trace, removes the DWT
software lock where present, and then enables `CYCCNT` before measurement.

A real board must provide its own `memory.x`; the QEMU memory map is deliberately
not reused as hardware evidence. Build and optionally run the harness with:

```text
MEMORY_X=/path/to/board/memory.x \
RUNNER='your-debugger-command' \
qemu-tests/run_cortex_m_hardware.sh
```

`RUNNER` is optional. When supplied, the script appends the generated ELF path
to the command. The debugger/runner must support the semihosting calls used by
the benchmark binary. `CORTEX_M_TARGET` defaults to
`thumbv7em-none-eabihf` and can be overridden for another compatible target.

For a publishable physical result, record alongside the raw harness output:

- board and exact MCU part number;
- core/flash/bus clock configuration;
- compiler and target triple;
- git commit;
- build profile and `RUSTFLAGS`;
- whether caches/prefetch/TCM are enabled;
- debugger/runner version;
- supply/thermal conditions if they materially affect clocking;
- repeated-run distribution, not just one best value.

Cycle counts should be converted to time only from the *measured/configured*
core clock for that run.

## What is still missing for the 0.3 release gate

The automated report establishes reproducible static-size and stack evidence and
keeps the physical benchmark harness buildable. It does **not** satisfy the
release requirement for a real embedded measurement by itself.

Before calling `0.3` stable, record at least one physical Cortex-M FPU result
using the harness above and check it into the qualification record (or another
versioned release artifact). A second maintained hardware target, preferably a
RISC-V or ESP-class target when available, remains a separate follow-up.

Only after physical data exists should the project make target-performance
comparisons with alternatives such as static nalgebra or CMSIS-DSP, and only for
operations where setup/storage semantics are genuinely comparable.
