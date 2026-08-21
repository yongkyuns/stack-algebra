# Target qualification

`stack-algebra` separates portable-core evidence from physical-device performance evidence. Cross-compilation and QEMU are useful qualification layers, but neither is a substitute for timing code on a named board.

## Evidence levels

1. **Cross-compile evidence** — the crate builds for the target triple.
2. **Emulator functional evidence** — representative code executes under QEMU and satisfies numerical assertions.
3. **Emulator/static resource evidence** — isolated workload ELFs report code/static size and painted-stack high-water marks. These are reproducible regression signals, not hardware timing results.
4. **Physical-target evidence** — the same workload definitions execute on a named board/MCU under a controlled configuration and report measured cycles/timing.

Do not combine these levels into a generic "embedded benchmark" claim.

## Current Cortex-M resource qualification

CI uses `thumbv7em-none-eabihf` and QEMU `mps2-an386`. Representative workloads are compiled as isolated release ELFs using the `qemu-tests` release profile (`opt-level = "z"`, LTO, one codegen unit) and execute with a painted-stack measurement.

Run the same report locally with:

```text
qemu-tests/qualify_cortex_m.sh
```

CI records source/checkout SHA, target, Rust/Cargo, size tooling, QEMU, release profile, input `RUSTFLAGS`, and the configured stack ceiling. The report contains `.text`, `.data`, `.bss`, baseline deltas, flash bytes, painted-stack high-water mark, stack ceiling, and retained workload object bytes.

The measured workloads cover dense 3x3 LU, dense 6x6 `f32`/`f64` product/fused paths, dense 15x15 `f32`, fixed-capacity sparse Cholesky, and block-sparse matvec. Definitions live in `qemu-tests/src/workloads.rs` and are shared with the physical timing harness so the two evidence tiers do not silently drift.

These measurements qualify the tested binaries/resource envelope under the emulator/link configuration. They do not establish exact library-only section attribution or physical MCU timing.

## Physical Cortex-M harness

`qemu-tests/src/bin/cortex_m_bench.rs` measures the shared workloads with Cortex-M DWT `CYCCNT`. The harness runs repeated measurements and keeps semihosting output outside the timed region.

A real board must supply its own memory map. Build and optionally run it with:

```text
MEMORY_X=/path/to/board/memory.x \
RUNNER='your-debugger-command' \
qemu-tests/run_cortex_m_hardware.sh
```

When a physical result is published, record at minimum the board/MCU, clocks, compiler/target, git commit, build flags/profile, cache/prefetch/TCM configuration, debugger/runner, and repeated-run distribution. Convert cycles to time only from the measured/configured core clock.

## 0.3 release policy

A physical Cortex-M measurement is **not required to publish the portable `0.3` library**. The release may rely on the existing host, cross-target, QEMU, and static-resource qualification as long as the documentation remains explicit about the evidence boundary.

Until a named-board result exists, `stack-algebra` must not claim:

- measured STM32/ESP/other MCU execution time or cycle counts;
- board-specific throughput superiority;
- peripheral behavior validated by QEMU;
- exact real-device stack high-water marks inferred from emulator results.

Physical data becomes a prerequisite when making those hardware-specific claims. A second maintained hardware target remains desirable follow-up work, not a release gate.

Only after comparable physical data exists should target-performance comparisons with alternatives such as static nalgebra or CMSIS-DSP be presented as real-device evidence.
