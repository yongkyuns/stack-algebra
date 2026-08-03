#define EIGEN_DONT_PARALLELIZE
#define EIGEN_NO_DEBUG
#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string_view>

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

template <typename Scalar, int Rows, int Columns>
using Matrix = Eigen::Matrix<Scalar, Rows, Columns, Eigen::ColMajor>;

template <typename Scalar, int Rows, int Columns>
Matrix<Scalar, Rows, Columns> make_matrix() {
  Matrix<Scalar, Rows, Columns> matrix;
  for (int row = 0; row < Rows; ++row) {
    for (int column = 0; column < Columns; ++column) {
      matrix(row, column) = static_cast<Scalar>(row * Columns + column + 1) / Scalar(17);
    }
  }
  return matrix;
}

template <typename Scalar, int Rows, int Columns>
Matrix<Scalar, Rows, Columns> make_rhs() {
  Matrix<Scalar, Rows, Columns> matrix;
  for (int row = 0; row < Rows; ++row) {
    for (int column = 0; column < Columns; ++column) {
      matrix(row, column) = static_cast<Scalar>(row + 2 * column + 3) / Scalar(11);
    }
  }
  return matrix;
}

template <typename Scalar, int Dimension>
Matrix<Scalar, Dimension, Dimension> make_system() {
  Matrix<Scalar, Dimension, Dimension> matrix;
  for (int row = 0; row < Dimension; ++row) {
    for (int column = 0; column < Dimension; ++column) {
      matrix(row, column) = row == column
                                ? Scalar(Dimension + 1)
                                : static_cast<Scalar>(row + 2 * column + 1) / Scalar(19);
    }
  }
  return matrix;
}

template <typename Operation>
double sample(Operation& operation, std::size_t batches) {
  const auto started = std::chrono::steady_clock::now();
  for (std::size_t batch = 0; batch < batches; ++batch) {
    for (std::size_t iteration = 0; iteration < kBatchSize; ++iteration) {
      operation();
    }
  }
  const auto elapsed = std::chrono::steady_clock::now() - started;
  const double nanoseconds = std::chrono::duration<double, std::nano>(elapsed).count();
  return nanoseconds / static_cast<double>(batches * kBatchSize);
}

template <typename Operation>
std::size_t calibrated_batches(Operation& operation) {
  std::size_t batches = 1;
  while (batches < (1U << 20)) {
    const double nanoseconds_per_operation = sample(operation, batches);
    const auto elapsed = std::chrono::duration<double, std::nano>(
        nanoseconds_per_operation * static_cast<double>(batches * kBatchSize));
    if (elapsed >= kMinimumSampleDuration) {
      return batches;
    }
    batches *= 2;
  }
  return batches;
}

template <typename Operation>
void benchmark_case(const char* name, Operation operation) {
  const std::size_t batches = calibrated_batches(operation);
  std::array<double, kSamples> samples{};
  for (double& result : samples) {
    result = sample(operation, batches);
  }
  std::sort(samples.begin(), samples.end());

  const double nanoseconds_per_operation = samples[kSamples / 2];
  std::cout << std::left << std::setw(24) << name << std::right << std::setw(14) << std::fixed
            << std::setprecision(2) << nanoseconds_per_operation * kBatchSize << std::setw(12)
            << nanoseconds_per_operation << '\n';
}

template <typename Scalar, int Rows, int Shared, int Columns>
void benchmark_product(const char* name) {
  auto lhs = make_matrix<Scalar, Rows, Shared>();
  auto rhs = make_rhs<Scalar, Shared, Columns>();
  Matrix<Scalar, Rows, Columns> output;
  benchmark_case(name, [&] {
    auto* lhs_pointer = opaque(&lhs);
    auto* rhs_pointer = opaque(&rhs);
    auto* output_pointer = opaque(&output);
    output_pointer->noalias() = *lhs_pointer * *rhs_pointer;
  });
  opaque(&output);
}

template <typename Scalar, int Dimension>
void benchmark_lu_factor(const char* name) {
  auto input = make_system<Scalar, Dimension>();
  benchmark_case(name, [&] {
    auto factor = opaque(&input)->partialPivLu();
    opaque(&factor);
  });
}

template <typename Scalar, int Dimension>
void benchmark_lu_solve(const char* name) {
  auto input = make_system<Scalar, Dimension>();
  auto factor = input.partialPivLu();
  auto rhs = make_matrix<Scalar, Dimension, 1>();
  Matrix<Scalar, Dimension, 1> solution;
  benchmark_case(name, [&] {
    auto* factor_pointer = opaque(&factor);
    auto* rhs_pointer = opaque(&rhs);
    auto* solution_pointer = opaque(&solution);
    *solution_pointer = factor_pointer->solve(*rhs_pointer);
  });
  opaque(&solution);
}

template <typename Scalar>
void benchmark_scalar(const char* scalar_name) {
  std::cout << '\n' << scalar_name << " fixed-size operations\n"
            << std::left << std::setw(24) << "operation" << std::right << std::setw(14)
            << "ns/batch" << std::setw(12) << "ns/op\n";
  benchmark_product<Scalar, 2, 2, 2>("matmul 2x2 * 2x2");
  benchmark_product<Scalar, 3, 3, 3>("matmul 3x3 * 3x3");
  benchmark_product<Scalar, 4, 4, 4>("matmul 4x4 * 4x4");
  benchmark_product<Scalar, 6, 6, 6>("matmul 6x6 * 6x6");
  benchmark_product<Scalar, 9, 9, 9>("matmul 9x9 * 9x9");
  benchmark_product<Scalar, 15, 15, 15>("matmul 15x15 * 15x15");
  benchmark_product<Scalar, 2, 3, 2>("matmul 2x3 * 3x2");
  benchmark_product<Scalar, 3, 6, 3>("matmul 3x6 * 6x3");
  benchmark_product<Scalar, 6, 15, 6>("matmul 6x15 * 15x6");
  benchmark_product<Scalar, 3, 3, 1>("matvec 3x3");
  benchmark_product<Scalar, 6, 6, 1>("matvec 6x6");
  benchmark_product<Scalar, 15, 15, 1>("matvec 15x15");
  benchmark_lu_factor<Scalar, 3>("LU factor 3x3");
  benchmark_lu_factor<Scalar, 6>("LU factor 6x6");
  benchmark_lu_factor<Scalar, 15>("LU factor 15x15");
  benchmark_lu_solve<Scalar, 3>("LU solve 3x3");
  benchmark_lu_solve<Scalar, 6>("LU solve 6x6");
  benchmark_lu_solve<Scalar, 15>("LU solve 15x15");
}

}

int main(int argc, char** argv) {
  if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "f32" &&
                   std::string_view(argv[1]) != "f64")) {
    std::cerr << "usage: eigen-native-bench [f32|f64]\n";
    return 1;
  }

  std::cout << "Eigen native benchmark: static column-major matrices; "
               "64 operations per batch; median of 15 samples.\n";
  if (argc == 1 || std::string_view(argv[1]) == "f32") {
    benchmark_scalar<float>("f32");
  }
  if (argc == 1 || std::string_view(argv[1]) == "f64") {
    benchmark_scalar<double>("f64");
  }
}
