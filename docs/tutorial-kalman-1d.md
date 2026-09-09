# Walkthrough: follow a moving object with a Kalman filter

Imagine a small cart moving along a straight track. Once a second, you get a
reading of its position. The readings are a little uneven: a reading might
say `1.2` metres, then `1.8`, then `3.1`. You would like to estimate where the
cart is and how quickly it is moving without treating every reading as exact.

A **Kalman filter** combines two things: a prediction based on what we already
know, and a correction based on the next reading. It also keeps track of how
uncertain those estimates are. This example repeats that predict-and-correct
cycle ten times.

You do not need a background in navigation, filtering, or matrix mathematics.
We will introduce the small tables of numbers as we use them. Familiarity with
Rust variables and functions helps; the less familiar syntax is explained
alongside the code. For a gentler first exercise with the library, start with
[fitting a line](tutorial-mapped-least-squares.md).

Follow [the setup instructions](tutorials.md#before-you-start), then run this
command from the repository folder:

```sh
cargo run --example kalman_1d --no-default-features
```

The program is already complete. The excerpts below come from that
[example file](https://github.com/yongkyuns/stack-algebra/blob/main/examples/kalman_1d.rs);
you do not need to combine them into a new program. There is a
[complete listing](#complete-runnable-example) at the end.

## 1. Decide what to remember between readings

We remember two estimates: **position**, measured in metres, and **velocity**,
measured in metres per second. Velocity includes direction: `+1` means moving
one metre per second along the track, and `-1` means moving the other way.
Together these two numbers are called the **state**. “Two-state” in this
example means two quantities, not two operating modes.

A **matrix** is a rectangular table of numbers. We store the state in a table
with two rows and one column, also called a **column vector**:

```text
[position]
[velocity]
```

The Rust type `Matrix<2, 1, f32>` says exactly that: two rows, one column, with
32-bit floating-point values. `f32` lets us represent numbers with fractional
parts. Because the dimensions are in the type, Rust can check that our matrix
operations have compatible sizes.

We also need a table describing uncertainty. This is the **covariance matrix**:

```text
                         Position error     Velocity error
Position error           position variance  shared uncertainty
Velocity error           shared uncertainty velocity variance
```

A **variance** measures uncertainty using squared units. A larger position
variance means we consider the position estimate less certain. It is not the
actual error, which we do not know. The two off-diagonal entries describe how
errors in position and velocity are related. This relationship will let a
position reading help us adjust velocity too.

The example initializes these tables here:

```rust,noplayground
{{#include ../examples/kalman_1d.rs:initial}}
```

`zeros()` fills the state with zeros. `eye()` creates an **identity matrix**:
ones on the diagonal and zeros elsewhere. Here is what we start with:

```text
state = [0]       covariance = [1  0]
        [0]                    [0  1]
```

This says “start with a position estimate of zero and a velocity estimate of
zero, but do not regard either as certain.” Each variance starts at `1` in
its own squared units, with no initial error correlation. These are chosen
starting assumptions; they do not assert that the cart is actually at rest.
`mut` means the variables can change as readings arrive. The
[`Matrix` reference](api/stack_algebra/struct.Matrix.html) lists the available
operations, but you do not need to read it before continuing.

The example also chooses fixed readings and uncertainty settings:

```rust,noplayground
{{#include ../examples/kalman_1d.rs:inputs}}
```

`DT` is the time between readings: one second. `MEASUREMENT_VARIANCE` describes
how uncertain each position reading is. `ACCELERATION_VARIANCE` allows for
changes in velocity that our simple prediction will not explicitly know.
These constants describe assumptions; the program does not learn them from
the ten readings.

<details>
<summary>Optional: variance and units</summary>

Taking the square root of a variance gives a **standard deviation**, an
uncertainty measure in the original units. A measurement variance of `0.25`
m^2 corresponds to a standard deviation of `0.5` metres; it is not a promise
that every reading is within half a metre of the truth.

Velocity variance has units of (m/s)^2. The shared uncertainty entries have
units of m^2/s. The acceleration setting `0.04` (m/s^2)^2 corresponds to a
standard deviation of `0.2` m/s^2. A variance is not a standard deviation;
using one where the other is expected changes the calculation.

</details>

## 2. Predict where the cart will be next

Our prediction assumes the current velocity continues over the next step:

```text
next position = current position + time step * current velocity
next velocity = current velocity
```

For example, a cart estimated to be at `2` metres and moving at `1` m/s would
be predicted at `3` metres one second later. This rule is the **model**: our
simplified description of how the cart moves, not a new measurement.

The helper puts this rule into a matrix named `transition`:

```text
[1  dt]
[0   1]
```

To multiply it by the state, take each row in turn, multiply matching entries,
and add them. The first row gives `1*position + dt*velocity`. The second gives
`0*position + 1*velocity`. That is all the state multiplication does here.

Here is the prediction helper:

```rust,noplayground
{{#include ../examples/kalman_1d.rs:predict}}
```

[`from_rows`](api/stack_algebra/struct.Matrix.html#method.from_rows) lets us
write the table row by row. The first assignment updates the state. The
second updates uncertainty: it carries the old uncertainty through the same
motion rule and adds uncertainty for possible changes in velocity.

The variable `acceleration` describes how a possible acceleration would affect
both position and velocity; it is **not** an acceleration reading. Multiplying
it by its [`transpose`](api/stack_algebra/struct.Matrix.html#method.transpose)
forms a two-by-two table. A transpose swaps rows and columns, so a two-row
column becomes a two-column row. Scaling that table by the acceleration
variance gives `process_noise`: the new uncertainty to add at each step.

`&mut` in the function arguments lets the helper change the original state
and covariance. `*state` means “the value reached through this reference.”
Between two values, as in `transition * *state`, `*` means multiplication.
The assignments calculate the right-hand side before replacing the value.

**First-step checkpoint:** because our starting velocity estimate is zero,
the predicted state is still zero. The uncertainty does change:

```text
predicted state = [0]       added uncertainty = [0.01  0.02]
                  [0]                           [0.02  0.04]

predicted covariance = [2.01  1.02]
                       [1.02  1.04]
```

The position variance grows from `1` to `2.01`. The off-diagonal entries are
now nonzero: an error in our velocity estimate would also affect where we
predict the cart to be.

<details>
<summary>Optional: connect the code to the usual equations</summary>

The usual names are `F` for `transition`, `P` for `covariance`, and `Q` for
`process_noise`. A superscript `T` means transpose. The prediction is:

```text
state_next = F * state
P_next     = F * P * F^T + Q

G = [dt^2 / 2, dt]^T
Q = G * G^T * acceleration_variance
```

This noise model imagines a random acceleration that stays constant during
each time interval, has average zero, and is independent between intervals.
It is a discrete per-interval model, not a continuous-time noise intensity.
The position-reading errors are assumed independent between samples and
independent of this motion uncertainty.

The small matrix expressions prioritize readability. They are not a claim
that a larger filter should use the same operations or has no temporary
storage.

</details>

## 3. Use the next reading to correct the prediction

Our first position reading is `1.2` metres, while the predicted position is
`0`. The difference is:

```text
difference = measured position - predicted position = 1.2 - 0 = 1.2
```

The code calls this difference the **innovation**. It tells us how much the
new reading disagrees with the prediction. A **scalar** measurement means we
receive one number at a time here: position, not a whole table of readings.

How much of that difference should we use? If the reading is uncertain, we
should not blindly replace our prediction with it. The **gain** tells us how
much to adjust each state entry, based on the uncertainties we have tracked.

```rust,noplayground
{{#include ../examples/kalman_1d.rs:gain}}
```

`state[(0, 0)]` is the first state entry because indexing starts at zero. The
code adds the predicted position variance and the measurement variance. This
sum, `innovation_variance`, describes the uncertainty of their difference
under the independence assumptions above.

It then divides the first covariance column by that sum to get two gain
entries. The first controls the position adjustment. The second controls the
velocity adjustment. Both are read before the covariance changes.

For the first reading:

```text
innovation_variance = 2.01 + 0.25 = 2.26
position gain       = 2.01 / 2.26, about 0.889381
velocity gain       = 1.02 / 2.26, about 0.451327

new position = 0 + 0.889381 * 1.2, about 1.067257 metres
new velocity = 0 + 0.451327 * 1.2, about 0.541593 m/s
```

We move the position estimate about 89 percent of the way toward the reading,
not all the way. Velocity changes too: the cart being farther along than
predicted is evidence that our initial velocity estimate may have been too
low. The covariance supplies the relationship needed for that adjustment.

The last call, [`axpy_in_place`](api/stack_algebra/struct.Matrix.html#method.axpy_in_place),
has a traditional algebra name, but its job is straightforward: multiply each
gain entry by the innovation and add it to the corresponding state entry.
“In place” means the existing state is updated rather than returning a new
state variable.

## 4. Update how uncertain we are

After using a reading, we must update uncertainty as well as the estimated
position and velocity. Otherwise the next step would use an uncertainty
table that no longer describes the calculation we just performed.

The example uses this formula, called the **Joseph form**:

```rust,noplayground
{{#include ../examples/kalman_1d.rs:covariance}}
```

You do not need to derive it to follow the example. There are two
contributions: uncertainty remaining from the prediction, and uncertainty
introduced by using an imperfect measurement. The gain determines how much
each contributes. The temporary table `residual_map` helps apply the first
contribution; it is not the measured position difference.

**After the first correction**, the table is approximately:

```text
[0.222345  0.112832]
[0.112832  0.579646]
```

Position variance is now about `0.222`, down from the predicted `2.01`.
Velocity variance has also decreased. The reading added information, but
neither estimate has become perfectly certain.

![Prediction, reading, and correction at the same instant, on a shared position scale with one-standard-deviation uncertainty bands.](generated/tutorials/kalman-first-update.svg)

This diagram uses intermediate values exported by the actual Rust execution.
The three rows describe the same instant, not three successive positions.
Band widths are the square roots of exported marginal variances; they express
assumed uncertainty, not the cart's size or a measured error. Only position
is measured. Velocity is inferred through the model and covariance.

<details>
<summary>Optional: read the Joseph formula</summary>

`I` is the identity matrix, `K` is the two-entry gain, and `H = [1, 0]`
selects position from the state. `R` is the measurement variance and `P` is
the predicted covariance. The code constructs `J = I - K*H`, then calculates:

```text
P_updated = J * P * J^T + R * K * K^T
```

The last term accounts for measurement uncertainty. The formula is a way to
organize the covariance calculation, not a guarantee against every rounding
or input problem. The tests check the resulting uncertainty table using
small numerical tolerances. The values shown in this walkthrough are rounded;
`f32` calculations may differ in the last digits.

</details>

## 5. Repeat and read the output

The loop repeats the same two actions for each reading: predict first, then
correct. The first reading belongs to time `1` second, not time zero.
The loop body captures copies of the predicted state and covariance for
reporting before the correction changes them:

```rust,noplayground
{{#include ../examples/kalman_1d.rs:cycle}}
```

`update_position` returns the innovation, its variance, and the gain it already
used. Reporting does not compute a second update. In the complete program,
`enumerate()` supplies the step number, starting at zero, so `step + 1`
counts the elapsed one-second intervals. `println!` displays the results;
`{:.3}` means three digits after the decimal point.

```text
{{#include generated/tutorials/kalman_1d.txt}}
```

The output block above is captured from the running example during the
documentation build, rather than maintained as a separate table. The worked
arithmetic earlier on this page describes the checked-in teaching inputs.

Read the first row as: “after one second, the reading was `1.200` metres;
our position estimate is `1.067` metres and our velocity estimate is `0.542`
m/s.” The position estimate need not equal the reading. By the last step,
the estimated velocity is near the roughly one metre per second suggested
by this particular sequence.

![The ten position readings and post-correction position estimates from the running example.](generated/tutorials/kalman-position.svg)

![Post-correction velocity estimates inferred from position readings, with no velocity measurements or ground-truth velocity added.](generated/tutorials/kalman-velocity.svg)

The lines join discrete estimates; they do not represent extra measurements.
Both figures use the [full-precision CSV export](generated/tutorials/kalman_1d.csv).
The [provenance file](generated/tutorials/provenance.json) records the executed
revision, toolchain, commands, source hashes, and output hashes.

These are estimates from chosen assumptions, not proof of the cart's true
motion. A more complicated application needs its own model and checks.

## 6. Change one assumption and see what happens

Open `examples/kalman_1d.rs`. Change `MEASUREMENT_VARIANCE` from `0.25` to `1.0`
and run the same Cargo command. This tells the filter to trust each reading
less, without changing the readings themselves.

Before running, predict what happens to the **first** correction. Its
predicted covariance is unchanged, but the denominator is larger, so the gain
is smaller. The first position estimate should stay closer to the prediction
of zero. Later steps also depend on earlier uncertainty updates, so compare
the first row to isolate the effect.

As a separate experiment, restore that value and set `ACCELERATION_VARIANCE`
to zero. The prediction then adds no new acceleration uncertainty. It does
not make the position readings exact or necessarily improve the estimates.

Restore the original constants before checking the documented output or
running the unchanged reference tests. Changing `DT` changes the assumed
times of the readings too; it is not only a display setting.

This small example assumes valid uncertainty values, ordinary finite input
numbers, nonnegative acceleration variance, and positive measurement variance.
Its helpers do not validate every input. Keep it as a learning example rather
than using it unchanged to handle arbitrary measurements.

## 7. Run the checks

```sh
cargo test --no-default-features --test kalman_1d
```

The five automated tests call the same helpers as the example. They check a
prediction against hand calculations, check a correction, try a reading that
exactly matches the prediction, compare the full sequence with a separately
written calculation, and run a longer sequence. They are kept in a
[separate file](https://github.com/yongkyuns/stack-algebra/blob/main/tests/kalman_1d.rs)
so the example remains readable.

<details>
<summary>Optional: run the other tested build configurations</summary>

`--features std` enables the library's standard-library support. `--release`
turns on compiler optimization; it does not publish a release.

```sh
cargo test --features std --test kalman_1d
cargo test --release --no-default-features --test kalman_1d
```

</details>

## Reproduce the figures

```sh
cargo run --quiet --example kalman_1d --no-default-features -- --csv
python3 scripts/generate_tutorial_assets.py
```

The CSV includes the input settings, initial state and covariance, predicted
and corrected state and covariance, innovation, and gain. See
[how tutorial assets are built](tutorial-assets.md) for the schema and
re-execution checks. No plotting library or second Kalman implementation is
required.

## Complete runnable example

This listing is taken directly from the example file. The `use` line brings
`Matrix` into scope. `pub(crate)` lets the separate tests call the helpers;
`#[cfg(not(test))]` excludes the printing entry point from those tests. The
small [reporting module](https://github.com/yongkyuns/stack-algebra/blob/main/examples/support/tutorial_output.rs)
handles command-line arguments and CSV formatting, not filtering mathematics.
Run the program from the repository using the Cargo command above.

```rust,noplayground
{{#include ../examples/kalman_1d.rs}}
```

Return to the [tutorial overview](tutorials.md), or see
[Choosing an API](api-usage.md) for larger solves, views, and reusable storage.
