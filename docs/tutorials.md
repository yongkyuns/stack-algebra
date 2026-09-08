# Tutorials

Learn `stack-algebra` by running a small program and understanding one part
at a time. You do not need a background in navigation, robotics, or advanced
mathematics. The walkthroughs explain the matrix and statistics terms when
they first appear, and connect the code to numbers you can check yourself.

**Start with fitting a line.** It introduces matrices as tables of numbers
and shows how to ask the library for a result. Then try the Kalman filter,
which updates an estimate as new readings arrive. Each page stands on its
own; expandable sections contain extra mathematical or Rust details that you
can skip on a first reading.

| Walkthrough | The question you will answer |
| --- | --- |
| [Fit a line to a few measurements](tutorial-mapped-least-squares.md) | What straight line best describes five input/output pairs, and how far are the measurements from that line? |
| [Follow a moving object](tutorial-kalman-1d.md) | How can a prediction and an imperfect position reading work together to estimate position and velocity? |

## Before you start

You will run these programs on your computer, not on a microcontroller. All
the input numbers are already in the examples. No sensors, special hardware,
Python setup, or external C++ library is needed.

You need Git and a stable Rust installation, including **Cargo**, Rust's tool
for building and running programs. Open a terminal and check:

```sh
git --version
rustc --version
cargo --version
```

If a command is not found, install the corresponding tool before continuing.
A little familiarity with variables, functions, and arrays will help you
read the Rust, but you do not need to know matrix multiplication or fitting
algorithms in advance. We explain references and other syntax where it is
used.

Copy the repository to a new folder and enter it:

```sh
git clone https://github.com/yongkyuns/stack-algebra.git
cd stack-algebra
```

Run the line-fitting example:

```sh
cargo run --example mapped_least_squares --no-default-features
```

`cargo run` builds and runs a program. `--example mapped_least_squares` picks
the file with that name in the `examples` folder. Cargo may download Rust
dependencies during the first build. The program then prints the fitted line
and a small table of results, explained in the walkthrough.

To run the other example:

```sh
cargo run --example kalman_1d --no-default-features
```

If you already have the repository, use its existing folder instead of
cloning again. Run the commands in the folder containing `Cargo.toml`, the
file describing this Rust package. A “could not find Cargo.toml” message
usually means you are in the wrong folder. You do not need `cargo new` or a
separate application project to run these examples.

The guide follows the development version in this repository. The version
available from the package registry may be older, so installing it with
`cargo add` is not a substitute for this checkout. To identify the source
version you are using, run `git rev-parse HEAD`; include that identifier when
reporting a problem. See [Getting started](getting-started.md) when you are
ready to add the library to your own application.

<details>
<summary>Optional: what do std and no_std mean?</summary>

Rust's standard library, `std`, supplies facilities such as printing to the
terminal. These example programs use it to display their results.
`stack-algebra` can perform its core calculations without that standard
library; this is called `no_std` support and is useful on small devices.

`--no-default-features` runs these examples with the library's default
features disabled. Their algebra uses the no_std core, but the programs still
run normally on your computer and print through std. You do not need an
embedded target or the browser playground.

</details>

Each walkthrough shows excerpts and a complete listing from the actual
example file, not a second implementation. Run the program with Cargo and
read along; the excerpts are not standalone programs to paste together.
The guide includes the code from its own source version. GitHub source links
point to `main`, which can move ahead of a particular copy of the guide.

## Fixed-size dense algebra

A matrix is a table of numbers. `Matrix<M, N, T>` specifies its number of
rows, number of columns, and number type. **Fixed-size** means those row and
column counts are known when the program is compiled. **Dense** means we
store every entry, including entries whose value is zero.

Both walkthroughs explain the table shapes they use. Afterward,
[Getting started](getting-started.md) and [Choosing an API](api-usage.md)
introduce more ways to create, multiply, and solve with matrices. The
[API reference](api-reference.md) gives exact method signatures.

## A two-state Kalman filter

[Follow the moving-object walkthrough](tutorial-kalman-1d.md). Start with a
cart on a track and imperfect position readings, then learn how the program
remembers position and velocity, predicts the next step, and corrects that
prediction. The guide explains uncertainty and works through the first
reading before showing the whole sequence.

The [example source](https://github.com/yongkyuns/stack-algebra/blob/main/examples/kalman_1d.rs)
and [automated tests](https://github.com/yongkyuns/stack-algebra/blob/main/tests/kalman_1d.rs)
remain small. The goal is to understand the library calls through a simple
model, not to build a complete tracking system.

## Views and external buffers

A **buffer** is storage holding some numbers, such as a Rust array. A **view**
lets the library treat those existing numbers as a matrix without creating
another copy of all its entries. The original array continues to own the
data. `Map` is the view used by the line-fitting example below.

### Fit a line from a caller-owned buffer

[Follow the line-fitting walkthrough](tutorial-mapped-least-squares.md) to
find a straight line through the overall pattern of five measurements.
It explains slope and intercept, shows how the data are arranged in memory,
asks the library to solve the problem, and interprets the differences left
over. An experiment with repeated inputs shows why some data cannot determine
a unique line.

Open the [example source](https://github.com/yongkyuns/stack-algebra/blob/main/examples/mapped_least_squares.rs)
or its [automated tests](https://github.com/yongkyuns/stack-algebra/blob/main/tests/mapped_least_squares.rs).
For other memory layouts after this walkthrough, see
[external buffers and views](api-usage.md) and the
[view reference](api-reference.md).

## Dense factorizations

A **factorization** prepares a matrix in a form that makes a calculation,
such as solving equations, easier. You use QR in the line-fitting walkthrough;
you do not need to learn all the other algorithms first.

When choosing an algorithm for your own problem, start with the
[solver guide](api-usage.md). It explains when to use methods such as Cholesky,
LU, QR, or SVD and how to handle a failed solve or reuse earlier work.

## Geometry

Geometry types describe rotations, movements, and changes of coordinate
systems. For example, a rotation can describe turning an object without
changing its size. You do not need these types for either walkthrough.
See [Common use cases](use-cases.md) and
[Capabilities and limits](features.md) when your problem involves geometry.

## Sparse and block-sparse systems

A **sparse** matrix has many zero entries. Sparse storage records the needed
entries and their locations rather than storing a full table. Block-sparse
storage groups entries into small rectangular blocks. These are later topics,
not prerequisites for the tutorials above.

Start with [sparse storage](api-usage.md) and
[sparse use cases](use-cases.md) when you have a problem that needs them.

## Embedded and bounded workflows

**Embedded** programs run inside devices, sometimes with very little memory.
**Bounded** storage reserves room for a maximum size while allowing a smaller
active size. These are reasons to choose explicit storage, but you can learn
and use the library on a desktop computer too.

The [platform guide](targets.md), [capability guide](features.md), and
[use cases](use-cases.md) explain these options. The small teaching examples
are for learning the API, not for measuring application performance.
