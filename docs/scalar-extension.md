# Custom scalar extensions

`stack-algebra` keeps architecture-specific SIMD backend types private. A
custom scalar opts into public operation-family traits whose default methods use
portable scalar loops.

The contract is intentionally split into three layers:

- `FactorizationScalar` supplies decomposition/update primitives. An empty impl
  selects the portable defaults.
- `MatrixScalar: FactorizationScalar` enables fixed-size matrix multiplication
  and matrix-product accumulation.
- `ReductionScalar: MatrixScalar` enables dot products, norms, and matrix-vector
  products.

This separation keeps factor-update mechanics out of `MatrixScalar` while still
allowing the built-in `f32` and `f64` implementations to select x86 or NEON
kernels at compile time. Custom scalar implementations do not name those private
backend types and do not incur runtime type dispatch.

## Minimal portable scalar

A scalar that supports addition, multiplication, and zero can opt into matrix
products and reductions with empty trait implementations:

```rust
use stack_algebra::{
    FactorizationScalar, Matrix, MatrixScalar, ReductionScalar, Zero,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Fixed(i32);

impl core::ops::Add for Fixed {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl core::ops::Mul for Fixed {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl Zero for Fixed {
    fn zero() -> Self {
        Self(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl FactorizationScalar for Fixed {}
impl MatrixScalar for Fixed {}
impl ReductionScalar for Fixed {}

let a = Matrix::<2, 2, Fixed>::from_rows([
    [Fixed(1), Fixed(2)],
    [Fixed(3), Fixed(4)],
]);
let x = Matrix::<2, 1, Fixed>::from_rows([[Fixed(5)], [Fixed(6)]]);
assert_eq!(
    a * x,
    Matrix::from_rows([[Fixed(17)], [Fixed(39)]])
);
```

Implement only the traits required by the operations the scalar should expose.
`MatrixScalar` requires `FactorizationScalar` because dense algorithms share
portable update primitives even though basic multiplication itself uses only the
matrix-product portion of the contract.

## Dense decompositions

Implementing the three scalar traits is not by itself a promise that every
decomposition is available. Numerical decompositions also impose their own
requirements, typically `Real` plus subtraction, division, ordering, finite
checks, square roots, or related floating-point operations.

For ordinary embedded numerical work, `f32` and `f64` remain the primary scalar
types and receive the maintained optimized paths. The custom-scalar contract is
primarily for portable arithmetic types whose required mathematical operations
are well defined.

## Specialization policy

Do not implement architecture dispatch in downstream scalar types unless there
is a measured need. The default trait methods are the compatibility path. The
built-in floating-point implementations specialize selected methods internally
for maintained target backends.

The project deliberately avoids runtime `TypeId` checks or unsafe type erasure
to distinguish built-in floats from external scalar types. Further reduction of
the public matrix-product/reduction hooks should happen only when a stable-Rust
design preserves code generation and the external extension path.
