# References

This page lists external projects and documentation used to inform the
implementation, API design, or validation of `stack-algebra`. These references
are not runtime integrations unless a dependency is listed explicitly.

## Implementation dependencies

- [`vectrix-macro`](https://crates.io/crates/vectrix-macro) — matrix and vector
  construction macros.
- [`stride`](https://docs.rs/stride/latest/stride/) — strided view iteration.
- [`num-traits`](https://docs.rs/num-traits/latest/num_traits/) — scalar and
  numeric trait foundations.
- [`approx`](https://docs.rs/approx/latest/approx/) — test assertions for
  floating-point results.

## API and design references

- [Eigen matrix classes](https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html)
  — fixed and dynamic dimensions, storage order, and scalar parameters.
- [Eigen `Map` and strided views](https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html)
  — external buffers and layout-aware views.
- [Eigen sparse matrices](https://eigen.tuxfamily.org/dox/group__TutorialSparse.html)
  — compressed sparse storage and sparse factorization terminology.
- [nalgebra matrices and vectors](https://www.nalgebra.rs/docs/user_guide/vectors_and_matrices/)
  — Rust static/dynamic storage and dimension conventions.
- [nalgebra embedded targets](https://www.nalgebra.rs/docs/user_guide/wasm_and_embedded_targets/)
  — `no_std` and embedded usage considerations.
- [`faer`](https://docs.rs/faer/latest/faer/) — Rust dense and sparse
  algorithms used as a performance reference for runtime-sized workloads.

## Validation references

- [Eigen](https://eigen.tuxfamily.org/) — differential correctness and native
  performance reference for selected operations.
- [`faer` benchmarks and APIs](https://docs.rs/faer/latest/faer/) — independent
  Rust comparison for dense and sparse operations.
- [`nalgebra`](https://docs.rs/nalgebra/latest/nalgebra/) — Rust comparison for
  static matrix APIs and numerical behavior.
- [Rust `no_std` documentation](https://doc.rust-lang.org/embedded-book/intro/install.html)
  — embedded build and target guidance.

The comparison scope, measurement model, and reproduction commands are
documented in [Benchmarking](benchmarking.md). Target evidence and its limits
are documented in [Target support and evidence](targets.md).
