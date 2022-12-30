# stack-algebra
A stack-allocated lightweight algebra library for bare-metal applications.

## Overview
This crate provides a stack-allocated matrix type with constant size determined at 
compile time. The primary goal for this library is to be useful in building robotics
applications in rust. This means several things:
1. Target platform is often bare-metal
2. Size of the matrices can usually be defined at compile-time
3. Problem solving does not require large matrices or heavy optimization
4. Users are not experts in rust but often familiar with scientific tools 
(e.g. python or matlab)

Implementing numerical algorithms in rust can be made much more productive and ergonomic
if simple abstractions and necessary algebra routines are available. This library is
a growing collection of addressing those needs. It is heavily based on 
[`vectrix`][vectrix] for core implementations.

## Usage

- `matrix!` macro can be used to create a new matrix
  ```rust
  // 2-by-3 matrix 
  let m = matrix![
      1.0, 2.0, 3.0;
      4.0, 5.0, 6.0; // Semicolon here is optional
  ]; 
  ```

- `vector!` macro can be used to create a row/column vector
  ```rust
  // 1-by-3 row vector
  let r = vector![1.0, 2.0, 3.0]; 

  // 3-by-1 column vector
  let c = vector![1.0; 2.0; 3.0]; 
  ```

- `[i]` or `[(r,c)]` to access individual elements
  ```rust
  let m = matrix![
      1.0, 2.0, 3.0;
      4.0, 5.0, 6.0
  ]; 

  assert_eq!(m[1], 4.0); // Using a single index assumes column-major order
  assert_eq!(m[(1,2)], 6.0);
  ```

- `*`, `/`, `+`, `-` for matrix arithmatics
  ```rust
  let m = matrix![
      1.0, 2.0;
	  3.0, 4.0
  ];

  let exp = matrix![
      2.0, 4.0;
	  6.0, 8.0
  ];

  assert_eq!(m + m, exp); // Add matrices
  
  let exp = matrix![
      2.0, 3.0;
	  4.0, 5.0
  ];

  assert_eq!(m + 1.0, exp); // Add scalar to matrix (note scalar has to be behind the operator)
  ```

- `.T()` for matrix transpose
  ```rust
  let m = matrix![
      1.0, 2.0;
	  3.0, 4.0
  ];

  let exp = matrix![
      1.0, 3.0;
	  2.0, 4.0
  ];

  assert_eq!(m.T(), exp); 

- `.det()` for determinant
  ```rust
    let m = matrix![
	  3.0, 7.0;
	  1.0, -4.0;
	];
    assert_eq!(m.det(), -19.0); 
  ```

- `.inv()` for inverse of a matrix
  ```rust
	let m = matrix![
	  6.0, 2.0, 3.0;
	  1.0, 1.0, 1.0;
	  0.0, 4.0, 9.0;
	];
	let exp = matrix![
	  0.20833333, -0.25, -0.04166667;
		  -0.375,  2.25,      -0.125;
	  0.16666667,  -1.0,  0.16666667;
	];
	assert_relative_eq!(m.inv().unwrap(), exp, max_relative = 1e-6);
  ```

## License

This project is distributed under the terms of both the MIT license and the Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details.

[vectrix]: https://docs.rs/vectrix/latest/vectrix/