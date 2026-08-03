#define EIGEN_DONT_PARALLELIZE
#define EIGEN_NO_DEBUG
#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>

namespace {

constexpr std::size_t kBatchSize = 64;
constexpr std::size_t kSamples = 15;
constexpr auto kMinimumSampleDuration = std::chrono::milliseconds(25);

template <typename Value>
inline Value* opaque(Value* pointer) {
#if defined(__clang__) || defined(__GNUC__)
  asm volatile("" : "+r"(pointer) : : "memory");
#endif
  return pointer;
}

template <typename Scalar, int Dimension>
using Matrix = Eigen::Matrix<Scalar, Dimension, Dimension, Eigen::ColMajor>;

template <typename Scalar, int Dimension>
Matrix<Scalar, Dimension> make_lhs() {
  Matrix<Scalar, Dimension> matrix;
  for (int row = 0; row < Dimension; ++row) {
    for (int column = 0; column < Dimension; ++column) {
      matrix(row, column) = static_cast<Scalar>(row * Dimension + column + 1) / Scalar(17);
    }
  }
  return matrix;
}

template <typename Scalar, int Dimension>
Matrix<Scalar, Dimension> make_rhs() {
  Matrix<Scalar, Dimension> matrix;
  for (int row = 0; row < Dimension; ++row) {
    for (int column = 0; column < Dimension; ++column) {
      matrix(row, column) = static_cast<Scalar>(row + 2 * column + 3) / Scalar(11);
    }
  }
  return matrix;
}

template <typename Scalar, int Dimension>
double sample(std::size_t batches) {
  auto lhs = make_lhs<Scalar, Dimension>();
  auto rhs = make_rhs<Scalar, Dimension>();
  Matrix<Scalar, Dimension> output;

  const auto started = std::chrono::steady_clock::now();
  for (std::size_t batch = 0; batch < batches; ++batch) {
    for (std::size_t iteration = 0; iteration < kBatchSize; ++iteration) {
      auto* lhs_pointer = opaque(&lhs);
      auto* rhs_pointer = opaque(&rhs);
      auto* output_pointer = opaque(&output);
      output_pointer->noalias() = *lhs_pointer * *rhs_pointer;
    }
  }
  const auto elapsed = std::chrono::steady_clock::now() - started;
  opaque(&output);

  const double nanoseconds =
      std::chrono::duration<double, std::nano>(elapsed).count();
  return nanoseconds / static_cast<double>(batches * kBatchSize);
}

template <typename Scalar, int Dimension>
std::size_t calibrated_batches() {
  std::size_t batches = 1;
  while (batches < (1U << 20)) {
    const double nanoseconds_per_operation = sample<Scalar, Dimension>(batches);
    const auto elapsed = std::chrono::duration<double, std::nano>(
        nanoseconds_per_operation * static_cast<double>(batches * kBatchSize));
    if (elapsed >= kMinimumSampleDuration) {
      return batches;
    }
    batches *= 2;
  }
  return batches;
}

template <typename Scalar, int Dimension>
void benchmark_dimension() {
  const std::size_t batches = calibrated_batches<Scalar, Dimension>();
  std::array<double, kSamples> samples{};
  for (double& result : samples) {
    result = sample<Scalar, Dimension>(batches);
  }
  std::sort(samples.begin(), samples.end());

  const double nanoseconds_per_operation = samples[kSamples / 2];
  std::cout << std::setw(3) << Dimension << std::setw(14) << std::fixed
            << std::setprecision(2) << nanoseconds_per_operation * kBatchSize << std::setw(12)
            << nanoseconds_per_operation << '\n';
}

template <typename Scalar>
void benchmark_scalar(const char* scalar_name) {
  std::cout << '\n' << scalar_name << " fixed-size matrix multiplication\n"
            << "  D      ns/batch       ns/op\n";
  benchmark_dimension<Scalar, 2>();
  benchmark_dimension<Scalar, 3>();
  benchmark_dimension<Scalar, 4>();
  benchmark_dimension<Scalar, 6>();
  benchmark_dimension<Scalar, 9>();
  benchmark_dimension<Scalar, 15>();
}

}

int main() {
  std::cout << "Eigen native benchmark: static column-major matrices; "
               "64 multiplications per batch; median of 15 samples.\n";
  benchmark_scalar<float>("f32");
  benchmark_scalar<double>("f64");
}
