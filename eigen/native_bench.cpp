#define EIGEN_DONT_PARALLELIZE
#define EIGEN_NO_DEBUG
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

namespace {

constexpr std::size_t kBatchSize = 64;
constexpr std::size_t kMaxSamples = 15;
constexpr std::size_t kDefaultSamples = 15;
std::string_view g_filter;
std::string_view g_scalar;
std::ofstream g_csv;

std::size_t environment_size(const char* name, std::size_t fallback, std::size_t minimum,
                             std::size_t maximum) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0') {
    return fallback;
  }
  const std::size_t result = static_cast<std::size_t>(parsed);
  return result < minimum || result > maximum ? fallback : result;
}

std::chrono::milliseconds minimum_sample_duration() {
  return std::chrono::milliseconds(
      environment_size("EIGEN_BENCH_MIN_SAMPLE_MS", 25, 1, 1000));
}

std::size_t sample_count() {
  return environment_size("EIGEN_BENCH_SAMPLES", kDefaultSamples, 3, kMaxSamples);
}

void write_csv_field(std::ostream& output, std::string_view value) {
  output << '"';
  for (const char character : value) {
    if (character == '"') {
      output << '"';
    }
    output << character;
  }
  output << '"';
}

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

template <typename Scalar, int Rows, int Columns>
Matrix<Scalar, Rows, Columns> make_tall_system() {
  Matrix<Scalar, Rows, Columns> matrix;
  for (int row = 0; row < Rows; ++row) {
    for (int column = 0; column < Columns; ++column) {
      matrix(row, column) = row == column
                                ? Scalar(Rows + 1)
                                : static_cast<Scalar>(row + 2 * column + 1) / Scalar(19);
    }
  }
  return matrix;
}

template <typename Scalar, int Dimension>
Matrix<Scalar, Dimension, Dimension> make_spd_system() {
  Matrix<Scalar, Dimension, Dimension> matrix;
  for (int row = 0; row < Dimension; ++row) {
    for (int column = 0; column < Dimension; ++column) {
      Scalar value = Scalar(0);
      for (int shared = 0; shared < Dimension; ++shared) {
        const Scalar left = static_cast<Scalar>(shared + 3 * row + 1) / Scalar(23);
        const Scalar right = static_cast<Scalar>(shared + 3 * column + 1) / Scalar(23);
        value += left * right;
      }
      matrix(row, column) = value + (row == column ? Scalar(Dimension) : Scalar(0));
    }
  }
  return matrix;
}

template <typename Scalar, int Dimension>
Matrix<Scalar, Dimension, Dimension> make_ldlt_system() {
  Matrix<Scalar, Dimension, Dimension> matrix;
  for (int row = 0; row < Dimension; ++row) {
    for (int column = 0; column < Dimension; ++column) {
      if (row == column) {
        matrix(row, column) = row % 2 == 0 ? Scalar(-Dimension) : Scalar(Dimension + 1);
      } else {
        matrix(row, column) = static_cast<Scalar>(row + column + 1) / Scalar(29);
      }
    }
  }
  return matrix;
}

template <typename Scalar, int Dimension>
Eigen::SparseMatrix<Scalar> make_sparse_spd_system() {
  Eigen::SparseMatrix<Scalar> matrix(Dimension, Dimension);
  matrix.reserve(2 * Dimension - 1);
  for (int column = 0; column < Dimension; ++column) {
    matrix.insert(column, column) = Scalar(4);
    if (column + 1 < Dimension) {
      matrix.insert(column + 1, column) = Scalar(1);
    }
  }
  matrix.makeCompressed();
  return matrix;
}

template <typename Scalar, int Dimension, int Band>
Eigen::SparseMatrix<Scalar> make_sparse_banded_spd_system() {
  Eigen::SparseMatrix<Scalar> matrix(Dimension, Dimension);
  matrix.reserve((Band + 1) * Dimension);
  for (int column = 0; column < Dimension; ++column) {
    const int end = std::min(Dimension, column + Band + 1);
    for (int row = column; row < end; ++row) {
      matrix.insert(row, column) = row == column ? Scalar(4) : Scalar(1);
    }
  }
  matrix.makeCompressed();
  return matrix;
}

template <typename Scalar, int Dimension>
Eigen::SparseMatrix<Scalar> make_sparse_star_spd_system() {
  Eigen::SparseMatrix<Scalar> matrix(Dimension, Dimension);
  matrix.reserve(2 * Dimension - 1);
  for (int column = 0; column < Dimension; ++column) {
    matrix.insert(column, column) = Scalar(4);
    if (column == 0) {
      for (int row = 1; row < Dimension; ++row) {
        matrix.insert(row, column) = Scalar(1);
      }
    }
  }
  matrix.makeCompressed();
  return matrix;
}

template <typename Scalar, int Dimension>
Eigen::SparseMatrix<Scalar> make_sparse_indefinite_system() {
  Eigen::SparseMatrix<Scalar> matrix(Dimension, Dimension);
  matrix.reserve(2 * Dimension - 1);
  for (int column = 0; column < Dimension; ++column) {
    matrix.insert(column, column) = column % 2 == 0 ? Scalar(4) : Scalar(-3);
    if (column + 1 < Dimension) {
      matrix.insert(column + 1, column) = Scalar(1);
    }
  }
  matrix.makeCompressed();
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
    if (elapsed >= minimum_sample_duration()) {
      return batches;
    }
    batches *= 2;
  }
  return batches;
}

template <typename Operation>
void benchmark_case(const char* name, Operation operation) {
  if (!g_filter.empty() && std::string_view(name).find(g_filter) == std::string_view::npos) {
    return;
  }
  const std::size_t batches = calibrated_batches(operation);
  const std::size_t samples_to_collect = sample_count();
  std::array<double, kMaxSamples> samples{};
  for (std::size_t index = 0; index < samples_to_collect; ++index) {
    samples[index] = sample(operation, batches);
  }
  std::sort(samples.begin(), samples.begin() + samples_to_collect);

  const double nanoseconds_per_operation = samples[samples_to_collect / 2];
  std::cout << std::left << std::setw(24) << name << std::right << std::setw(14) << std::fixed
            << std::setprecision(2) << nanoseconds_per_operation * kBatchSize << std::setw(12)
            << nanoseconds_per_operation << '\n';
  if (g_csv.is_open()) {
    write_csv_field(g_csv, g_scalar);
    g_csv << ',';
    write_csv_field(g_csv, name);
    g_csv << ',' << std::setprecision(17) << nanoseconds_per_operation << ','
          << nanoseconds_per_operation * kBatchSize << ',' << kBatchSize << ','
          << samples_to_collect
          << ',' << batches << '\n';
  }
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
Matrix<Scalar, Dimension, Dimension> make_lower_system() {
  auto matrix = make_system<Scalar, Dimension>();
  for (int row = 0; row < Dimension; ++row) {
    for (int column = row + 1; column < Dimension; ++column) {
      matrix(row, column) = Scalar(0);
    }
  }
  return matrix;
}

template <typename Scalar, int Dimension>
Matrix<Scalar, Dimension, Dimension> make_upper_system() {
  auto matrix = make_system<Scalar, Dimension>();
  for (int row = 1; row < Dimension; ++row) {
    for (int column = 0; column < row; ++column) {
      matrix(row, column) = Scalar(0);
    }
  }
  return matrix;
}

template <typename Scalar, int Dimension>
void benchmark_triangular_solve(const char* name, bool lower) {
  auto input = lower ? make_lower_system<Scalar, Dimension>()
                     : make_upper_system<Scalar, Dimension>();
  auto rhs = make_rhs<Scalar, Dimension, 1>();
  Matrix<Scalar, Dimension, 1> solution;
  benchmark_case(name, [&] {
    auto* input_pointer = opaque(&input);
    auto* rhs_pointer = opaque(&rhs);
    auto* solution_pointer = opaque(&solution);
    if (lower) {
      *solution_pointer = input_pointer->template triangularView<Eigen::Lower>().solve(*rhs_pointer);
    } else {
      *solution_pointer = input_pointer->template triangularView<Eigen::Upper>().solve(*rhs_pointer);
    }
  });
  opaque(&solution);
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
void benchmark_llt_factor(const char* name) {
  auto input = make_spd_system<Scalar, Dimension>();
  benchmark_case(name, [&] {
    auto factor = opaque(&input)->llt();
    opaque(&factor);
  });
}

template <typename Scalar, int Dimension>
void benchmark_sparse_llt_analyze(const char* name) {
  auto input = make_sparse_spd_system<Scalar, Dimension>();
  benchmark_case(name, [&] {
    Eigen::SimplicialLLT<Eigen::SparseMatrix<Scalar>, Eigen::Lower> factor;
    factor.analyzePattern(*opaque(&input));
    opaque(&factor);
  });
}

template <typename Scalar, int Dimension>
void benchmark_sparse_llt_factor(const char* name) {
  auto input = make_sparse_spd_system<Scalar, Dimension>();
  Eigen::SimplicialLLT<Eigen::SparseMatrix<Scalar>, Eigen::Lower> factor;
  factor.analyzePattern(input);
  benchmark_case(name, [&] {
    factor.factorize(*opaque(&input));
    opaque(&factor);
  });
}

template <typename Scalar, int Dimension>
void benchmark_sparse_llt_solve(const char* name) {
  auto input = make_sparse_spd_system<Scalar, Dimension>();
  Eigen::SimplicialLLT<Eigen::SparseMatrix<Scalar>, Eigen::Lower> factor;
  factor.compute(input);
  Matrix<Scalar, Dimension, 1> rhs = make_rhs<Scalar, Dimension, 1>();
  Matrix<Scalar, Dimension, 1> solution;
  benchmark_case(name, [&] {
    solution = factor.solve(*opaque(&rhs));
    opaque(&solution);
  });
}

template <typename Scalar, int Dimension>
void benchmark_ldlt_factor(const char* name) {
  auto input = make_ldlt_system<Scalar, Dimension>();
  benchmark_case(name, [&] {
    auto factor = opaque(&input)->ldlt();
    opaque(&factor);
  });
}

template <typename Scalar, int Rows, int Columns>
void benchmark_norm(const char* name) {
  auto input = make_matrix<Scalar, Rows, Columns>();
  Scalar output{};
  benchmark_case(name, [&] {
    auto* input_pointer = opaque(&input);
    output = input_pointer->norm();
  });
  opaque(&output);
}

template <typename Scalar, int Dimension>
void benchmark_dot(const char* name) {
  using RowVector = Eigen::Matrix<Scalar, 1, Dimension, Eigen::RowMajor>;
  using ColVector = Eigen::Matrix<Scalar, Dimension, 1, Eigen::ColMajor>;
  RowVector lhs;
  ColVector rhs;
  for (int index = 0; index < Dimension; ++index) {
    lhs(index) = static_cast<Scalar>(index + 1) / Scalar(13);
    rhs(index) = static_cast<Scalar>(2 * index + 3) / Scalar(7);
  }
  Scalar output{};
  benchmark_case(name, [&] {
    auto* lhs_pointer = opaque(&lhs);
    auto* rhs_pointer = opaque(&rhs);
    output = lhs_pointer->dot(*rhs_pointer);
  });
  opaque(&output);
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

template <typename Scalar, int Dimension, int Band>
void benchmark_sparse_banded_llt_factor(const char* name) {
  auto input = make_sparse_banded_spd_system<Scalar, Dimension, Band>();
  Eigen::SimplicialLLT<Eigen::SparseMatrix<Scalar>, Eigen::Lower> factor;
  factor.analyzePattern(input);
  benchmark_case(name, [&] {
    factor.factorize(*opaque(&input));
    opaque(&factor);
  });
}

template <typename Scalar, int Dimension, int Band>
void benchmark_sparse_banded_llt_solve(const char* name) {
  auto input = make_sparse_banded_spd_system<Scalar, Dimension, Band>();
  Eigen::SimplicialLLT<Eigen::SparseMatrix<Scalar>, Eigen::Lower> factor;
  factor.compute(input);
  Matrix<Scalar, Dimension, 1> rhs = make_rhs<Scalar, Dimension, 1>();
  Matrix<Scalar, Dimension, 1> solution;
  benchmark_case(name, [&] {
    solution = factor.solve(*opaque(&rhs));
    opaque(&solution);
  });
}

template <typename Scalar, int Dimension>
void benchmark_sparse_star_llt_factor(const char* name) {
  auto input = make_sparse_star_spd_system<Scalar, Dimension>();
  Eigen::SimplicialLLT<Eigen::SparseMatrix<Scalar>, Eigen::Lower> factor;
  factor.analyzePattern(input);
  benchmark_case(name, [&] {
    factor.factorize(*opaque(&input));
    opaque(&factor);
  });
}

template <typename Scalar, int Dimension>
void benchmark_sparse_star_llt_solve(const char* name) {
  auto input = make_sparse_star_spd_system<Scalar, Dimension>();
  Eigen::SimplicialLLT<Eigen::SparseMatrix<Scalar>, Eigen::Lower> factor;
  factor.compute(input);
  Matrix<Scalar, Dimension, 1> rhs = make_rhs<Scalar, Dimension, 1>();
  Matrix<Scalar, Dimension, 1> solution;
  benchmark_case(name, [&] {
    solution = factor.solve(*opaque(&rhs));
    opaque(&solution);
  });
}

template <typename Scalar, int Dimension>
void benchmark_qr_factor(const char* name) {
  auto input = make_system<Scalar, Dimension>();
  auto factor = input.householderQr();
  benchmark_case(name, [&] {
    auto* input_pointer = opaque(&input);
    factor.compute(*input_pointer);
    opaque(&factor);
  });
}

template <typename Scalar, int Dimension>
void benchmark_qr_solve(const char* name) {
  auto input = make_system<Scalar, Dimension>();
  auto factor = input.householderQr();
  auto rhs = make_rhs<Scalar, Dimension, 1>();
  Matrix<Scalar, Dimension, 1> solution;
  benchmark_case(name, [&] {
    auto* factor_pointer = opaque(&factor);
    auto* rhs_pointer = opaque(&rhs);
    auto* solution_pointer = opaque(&solution);
    *solution_pointer = factor_pointer->solve(*rhs_pointer);
  });
  opaque(&solution);
}

template <typename Scalar, int Dimension>
void benchmark_sparse_ldlt_factor(const char* name) {
  auto input = make_sparse_indefinite_system<Scalar, Dimension>();
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>, Eigen::Lower> factor;
  factor.analyzePattern(input);
  benchmark_case(name, [&] {
    factor.factorize(*opaque(&input));
    opaque(&factor);
  });
}

template <typename Scalar, int Dimension>
void benchmark_sparse_ldlt_solve(const char* name) {
  auto input = make_sparse_indefinite_system<Scalar, Dimension>();
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>, Eigen::Lower> factor;
  factor.compute(input);
  Matrix<Scalar, Dimension, 1> rhs = make_rhs<Scalar, Dimension, 1>();
  Matrix<Scalar, Dimension, 1> solution;
  benchmark_case(name, [&] {
    solution = factor.solve(*opaque(&rhs));
    opaque(&solution);
  });
}

template <typename Scalar, int Dimension>
void benchmark_col_piv_qr_factor(const char* name) {
  auto input = make_system<Scalar, Dimension>();
  auto factor = input.colPivHouseholderQr();
  benchmark_case(name, [&] {
    auto* input_pointer = opaque(&input);
    factor.compute(*input_pointer);
    opaque(&factor);
  });
}

template <typename Scalar, int Dimension>
void benchmark_col_piv_qr_solve(const char* name) {
  auto input = make_system<Scalar, Dimension>();
  auto factor = input.colPivHouseholderQr();
  auto rhs = make_rhs<Scalar, Dimension, 1>();
  Matrix<Scalar, Dimension, 1> solution;
  benchmark_case(name, [&] {
    auto* factor_pointer = opaque(&factor);
    auto* rhs_pointer = opaque(&rhs);
    auto* solution_pointer = opaque(&solution);
    *solution_pointer = factor_pointer->solve(*rhs_pointer);
  });
  opaque(&solution);
}

template <typename Scalar, int Rows, int Columns>
void benchmark_tall_qr_factor(const char* name) {
  auto input = make_tall_system<Scalar, Rows, Columns>();
  auto factor = input.householderQr();
  benchmark_case(name, [&] {
    auto* input_pointer = opaque(&input);
    factor.compute(*input_pointer);
    opaque(&factor);
  });
}

template <typename Scalar, int Rows, int Columns>
void benchmark_tall_qr_solve(const char* name) {
  auto input = make_tall_system<Scalar, Rows, Columns>();
  auto factor = input.householderQr();
  auto rhs = make_rhs<Scalar, Rows, 1>();
  Matrix<Scalar, Columns, 1> solution;
  benchmark_case(name, [&] {
    auto* factor_pointer = opaque(&factor);
    auto* rhs_pointer = opaque(&rhs);
    auto* solution_pointer = opaque(&solution);
    *solution_pointer = factor_pointer->solve(*rhs_pointer);
  });
  opaque(&solution);
}

template <typename Scalar, int Rows, int Columns>
void benchmark_tall_svd_factor(const char* name) {
  using Svd = Eigen::JacobiSVD<Matrix<Scalar, Rows, Columns>, Eigen::ComputeThinU | Eigen::ComputeThinV>;
  auto input = make_tall_system<Scalar, Rows, Columns>();
  Svd factor;
  benchmark_case(name, [&] {
    auto* input_pointer = opaque(&input);
    factor.compute(*input_pointer);
    opaque(&factor);
  });
}

template <typename Scalar, int Rows, int Columns>
void benchmark_tall_svd_solve(const char* name) {
  using Svd = Eigen::JacobiSVD<Matrix<Scalar, Rows, Columns>, Eigen::ComputeThinU | Eigen::ComputeThinV>;
  auto input = make_tall_system<Scalar, Rows, Columns>();
  Svd factor(input);
  auto rhs = make_rhs<Scalar, Rows, 1>();
  Matrix<Scalar, Columns, 1> solution;
  benchmark_case(name, [&] {
    auto* factor_pointer = opaque(&factor);
    auto* rhs_pointer = opaque(&rhs);
    auto* solution_pointer = opaque(&solution);
    *solution_pointer = factor_pointer->solve(*rhs_pointer);
  });
  opaque(&solution);
}

template <typename Scalar, int Dimension>
void benchmark_self_adjoint_eigen_factor(const char* name) {
  auto input = make_spd_system<Scalar, Dimension>();
  Eigen::SelfAdjointEigenSolver<Matrix<Scalar, Dimension, Dimension>> factor;
  benchmark_case(name, [&] {
    auto* input_pointer = opaque(&input);
    factor.compute(*input_pointer);
    opaque(&factor);
  });
}

template <typename Scalar, int Dimension>
void benchmark_llt_solve(const char* name) {
  auto input = make_spd_system<Scalar, Dimension>();
  auto factor = input.llt();
  auto rhs = make_rhs<Scalar, Dimension, 1>();
  Matrix<Scalar, Dimension, 1> solution;
  benchmark_case(name, [&] {
    auto* factor_pointer = opaque(&factor);
    auto* rhs_pointer = opaque(&rhs);
    auto* solution_pointer = opaque(&solution);
    *solution_pointer = factor_pointer->solve(*rhs_pointer);
  });
  opaque(&solution);
}

template <typename Scalar, int Dimension>
void benchmark_ldlt_solve(const char* name) {
  auto input = make_ldlt_system<Scalar, Dimension>();
  auto factor = input.ldlt();
  auto rhs = make_rhs<Scalar, Dimension, 1>();
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
  g_scalar = scalar_name;
  std::cout << '\n' << scalar_name << " fixed-size operations\n"
            << std::left << std::setw(24) << "operation" << std::right << std::setw(14)
            << "ns/batch" << std::setw(12) << "ns/op\n";
  benchmark_product<Scalar, 2, 2, 2>("matmul 2x2 * 2x2");
  benchmark_product<Scalar, 3, 3, 3>("matmul 3x3 * 3x3");
  benchmark_product<Scalar, 4, 4, 4>("matmul 4x4 * 4x4");
  benchmark_product<Scalar, 6, 6, 6>("matmul 6x6 * 6x6");
  benchmark_product<Scalar, 8, 8, 8>("matmul 8x8 * 8x8");
  benchmark_product<Scalar, 9, 9, 9>("matmul 9x9 * 9x9");
  benchmark_product<Scalar, 15, 15, 15>("matmul 15x15 * 15x15");
  benchmark_product<Scalar, 16, 16, 16>("matmul 16x16 * 16x16");
  benchmark_product<Scalar, 32, 32, 32>("matmul 32x32 * 32x32");
  benchmark_product<Scalar, 2, 3, 2>("matmul 2x3 * 3x2");
  benchmark_product<Scalar, 3, 6, 3>("matmul 3x6 * 6x3");
  benchmark_product<Scalar, 6, 15, 6>("matmul 6x15 * 15x6");
  benchmark_product<Scalar, 3, 3, 1>("matvec 3x3");
  benchmark_product<Scalar, 6, 6, 1>("matvec 6x6");
  benchmark_product<Scalar, 8, 8, 1>("matvec 8x8");
  benchmark_product<Scalar, 15, 15, 1>("matvec 15x15");
  benchmark_product<Scalar, 16, 16, 1>("matvec 16x16");
  benchmark_product<Scalar, 32, 32, 1>("matvec 32x32");
  benchmark_norm<Scalar, 3, 3>("norm 3x3");
  benchmark_norm<Scalar, 6, 6>("norm 6x6");
  benchmark_norm<Scalar, 8, 8>("norm 8x8");
  benchmark_norm<Scalar, 15, 15>("norm 15x15");
  benchmark_norm<Scalar, 16, 16>("norm 16x16");
  benchmark_norm<Scalar, 32, 32>("norm 32x32");
  benchmark_norm<Scalar, 6, 15>("norm 6x15");
  benchmark_dot<Scalar, 3>("dot 3");
  benchmark_dot<Scalar, 6>("dot 6");
  benchmark_dot<Scalar, 8>("dot 8");
  benchmark_dot<Scalar, 15>("dot 15");
  benchmark_dot<Scalar, 16>("dot 16");
  benchmark_dot<Scalar, 32>("dot 32");
  benchmark_lu_factor<Scalar, 3>("LU factor 3x3");
  benchmark_lu_factor<Scalar, 6>("LU factor 6x6");
  benchmark_lu_factor<Scalar, 8>("LU factor 8x8");
  benchmark_lu_factor<Scalar, 15>("LU factor 15x15");
  benchmark_lu_factor<Scalar, 16>("LU factor 16x16");
  benchmark_lu_factor<Scalar, 32>("LU factor 32x32");
  benchmark_llt_factor<Scalar, 3>("LLT factor 3x3");
  benchmark_llt_factor<Scalar, 6>("LLT factor 6x6");
  benchmark_llt_factor<Scalar, 8>("LLT factor 8x8");
  benchmark_llt_factor<Scalar, 15>("LLT factor 15x15");
  benchmark_llt_factor<Scalar, 16>("LLT factor 16x16");
  benchmark_llt_factor<Scalar, 32>("LLT factor 32x32");
  benchmark_sparse_llt_analyze<Scalar, 3>("Sparse LLT analyze 3x3");
  benchmark_sparse_llt_analyze<Scalar, 6>("Sparse LLT analyze 6x6");
  benchmark_sparse_llt_analyze<Scalar, 15>("Sparse LLT analyze 15x15");
  benchmark_sparse_llt_analyze<Scalar, 32>("Sparse LLT analyze 32x32");
  benchmark_sparse_llt_factor<Scalar, 3>("Sparse LLT factor 3x3");
  benchmark_sparse_llt_factor<Scalar, 6>("Sparse LLT factor 6x6");
  benchmark_sparse_llt_factor<Scalar, 15>("Sparse LLT factor 15x15");
  benchmark_sparse_llt_factor<Scalar, 32>("Sparse LLT factor 32x32");
  benchmark_sparse_banded_llt_factor<Scalar, 15, 2>("Sparse band2 LLT factor 15x15");
  benchmark_sparse_banded_llt_solve<Scalar, 15, 2>("Sparse band2 LLT solve 15x15");
  benchmark_sparse_star_llt_factor<Scalar, 15>("Sparse star LLT factor 15x15");
  benchmark_sparse_star_llt_solve<Scalar, 15>("Sparse star LLT solve 15x15");
  benchmark_sparse_ldlt_factor<Scalar, 15>("Sparse LDLT factor 15x15");
  benchmark_ldlt_factor<Scalar, 3>("LDLT factor 3x3");
  benchmark_ldlt_factor<Scalar, 6>("LDLT factor 6x6");
  benchmark_ldlt_factor<Scalar, 8>("LDLT factor 8x8");
  benchmark_ldlt_factor<Scalar, 15>("LDLT factor 15x15");
  benchmark_ldlt_factor<Scalar, 16>("LDLT factor 16x16");
  benchmark_ldlt_factor<Scalar, 32>("LDLT factor 32x32");
  benchmark_lu_solve<Scalar, 3>("LU solve 3x3");
  benchmark_lu_solve<Scalar, 6>("LU solve 6x6");
  benchmark_lu_solve<Scalar, 8>("LU solve 8x8");
  benchmark_lu_solve<Scalar, 15>("LU solve 15x15");
  benchmark_lu_solve<Scalar, 16>("LU solve 16x16");
  benchmark_lu_solve<Scalar, 32>("LU solve 32x32");
  benchmark_qr_factor<Scalar, 3>("QR factor 3x3");
  benchmark_qr_factor<Scalar, 6>("QR factor 6x6");
  benchmark_qr_factor<Scalar, 8>("QR factor 8x8");
  benchmark_qr_factor<Scalar, 15>("QR factor 15x15");
  benchmark_qr_factor<Scalar, 16>("QR factor 16x16");
  benchmark_qr_factor<Scalar, 32>("QR factor 32x32");
  benchmark_qr_solve<Scalar, 3>("QR solve 3x3");
  benchmark_qr_solve<Scalar, 6>("QR solve 6x6");
  benchmark_qr_solve<Scalar, 8>("QR solve 8x8");
  benchmark_qr_solve<Scalar, 15>("QR solve 15x15");
  benchmark_qr_solve<Scalar, 16>("QR solve 16x16");
  benchmark_qr_solve<Scalar, 32>("QR solve 32x32");
  benchmark_col_piv_qr_factor<Scalar, 3>("ColPiv QR factor 3x3");
  benchmark_col_piv_qr_factor<Scalar, 6>("ColPiv QR factor 6x6");
  benchmark_col_piv_qr_factor<Scalar, 8>("ColPiv QR factor 8x8");
  benchmark_col_piv_qr_factor<Scalar, 15>("ColPiv QR factor 15x15");
  benchmark_col_piv_qr_factor<Scalar, 16>("ColPiv QR factor 16x16");
  benchmark_col_piv_qr_factor<Scalar, 32>("ColPiv QR factor 32x32");
  benchmark_col_piv_qr_solve<Scalar, 3>("ColPiv QR solve 3x3");
  benchmark_col_piv_qr_solve<Scalar, 6>("ColPiv QR solve 6x6");
  benchmark_col_piv_qr_solve<Scalar, 8>("ColPiv QR solve 8x8");
  benchmark_col_piv_qr_solve<Scalar, 15>("ColPiv QR solve 15x15");
  benchmark_col_piv_qr_solve<Scalar, 16>("ColPiv QR solve 16x16");
  benchmark_col_piv_qr_solve<Scalar, 32>("ColPiv QR solve 32x32");
  benchmark_tall_qr_factor<Scalar, 6, 3>("Tall QR factor 6x3");
  benchmark_tall_qr_factor<Scalar, 15, 6>("Tall QR factor 15x6");
  benchmark_tall_qr_factor<Scalar, 32, 8>("Tall QR factor 32x8");
  benchmark_tall_qr_factor<Scalar, 64, 16>("Tall QR factor 64x16");
  benchmark_tall_qr_solve<Scalar, 6, 3>("Tall QR solve 6x3");
  benchmark_tall_qr_solve<Scalar, 15, 6>("Tall QR solve 15x6");
  benchmark_tall_qr_solve<Scalar, 32, 8>("Tall QR solve 32x8");
  benchmark_tall_qr_solve<Scalar, 64, 16>("Tall QR solve 64x16");
  benchmark_tall_svd_factor<Scalar, 6, 3>("Tall SVD factor 6x3");
  benchmark_tall_svd_factor<Scalar, 15, 6>("Tall SVD factor 15x6");
  benchmark_tall_svd_solve<Scalar, 6, 3>("Tall SVD solve 6x3");
  benchmark_tall_svd_solve<Scalar, 15, 6>("Tall SVD solve 15x6");
  benchmark_self_adjoint_eigen_factor<Scalar, 3>("Self-adjoint eigen factor 3x3");
  benchmark_self_adjoint_eigen_factor<Scalar, 6>("Self-adjoint eigen factor 6x6");
  benchmark_self_adjoint_eigen_factor<Scalar, 8>("Self-adjoint eigen factor 8x8");
  benchmark_self_adjoint_eigen_factor<Scalar, 15>("Self-adjoint eigen factor 15x15");
  benchmark_self_adjoint_eigen_factor<Scalar, 16>("Self-adjoint eigen factor 16x16");
  benchmark_self_adjoint_eigen_factor<Scalar, 32>("Self-adjoint eigen factor 32x32");
  benchmark_triangular_solve<Scalar, 3>("Lower triangular solve 3x3", true);
  benchmark_triangular_solve<Scalar, 6>("Lower triangular solve 6x6", true);
  benchmark_triangular_solve<Scalar, 8>("Lower triangular solve 8x8", true);
  benchmark_triangular_solve<Scalar, 15>("Lower triangular solve 15x15", true);
  benchmark_triangular_solve<Scalar, 16>("Lower triangular solve 16x16", true);
  benchmark_triangular_solve<Scalar, 3>("Upper triangular solve 3x3", false);
  benchmark_triangular_solve<Scalar, 6>("Upper triangular solve 6x6", false);
  benchmark_triangular_solve<Scalar, 8>("Upper triangular solve 8x8", false);
  benchmark_triangular_solve<Scalar, 15>("Upper triangular solve 15x15", false);
  benchmark_triangular_solve<Scalar, 16>("Upper triangular solve 16x16", false);
  benchmark_llt_solve<Scalar, 3>("LLT solve 3x3");
  benchmark_llt_solve<Scalar, 6>("LLT solve 6x6");
  benchmark_llt_solve<Scalar, 8>("LLT solve 8x8");
  benchmark_llt_solve<Scalar, 15>("LLT solve 15x15");
  benchmark_llt_solve<Scalar, 16>("LLT solve 16x16");
  benchmark_llt_solve<Scalar, 32>("LLT solve 32x32");
  benchmark_sparse_llt_solve<Scalar, 3>("Sparse LLT solve 3x3");
  benchmark_sparse_llt_solve<Scalar, 6>("Sparse LLT solve 6x6");
  benchmark_sparse_llt_solve<Scalar, 15>("Sparse LLT solve 15x15");
  benchmark_sparse_llt_solve<Scalar, 32>("Sparse LLT solve 32x32");
  benchmark_sparse_ldlt_solve<Scalar, 15>("Sparse LDLT solve 15x15");
  benchmark_ldlt_solve<Scalar, 3>("LDLT solve 3x3");
  benchmark_ldlt_solve<Scalar, 6>("LDLT solve 6x6");
  benchmark_ldlt_solve<Scalar, 8>("LDLT solve 8x8");
  benchmark_ldlt_solve<Scalar, 15>("LDLT solve 15x15");
  benchmark_ldlt_solve<Scalar, 16>("LDLT solve 16x16");
  benchmark_ldlt_solve<Scalar, 32>("LDLT solve 32x32");
}

}

int main(int argc, char** argv) {
  if (argc > 3 || (argc >= 2 && std::string_view(argv[1]) != "f32" &&
                   std::string_view(argv[1]) != "f64")) {
    std::cerr << "usage: eigen-native-bench [f32|f64] [filter]\n";
    return 1;
  }
  if (argc == 3) {
    g_filter = argv[2];
  }

  if (const char* csv_path = std::getenv("EIGEN_BENCH_CSV"); csv_path != nullptr && *csv_path != '\0') {
    g_csv.open(csv_path);
    if (!g_csv) {
      std::cerr << "unable to open EIGEN_BENCH_CSV output: " << csv_path << '\n';
      return 1;
    }
    g_csv << "scalar,operation,ns_per_op,ns_per_batch,batch_size,samples,calibrated_batches\n";
  }

  std::cout << "Eigen native benchmark: static column-major matrices; "
               "64 operations per batch; configurable median sample count.\n";
  if (argc == 1 || std::string_view(argv[1]) == "f32") {
    benchmark_scalar<float>("f32");
  }
  if (argc == 1 || std::string_view(argv[1]) == "f64") {
    benchmark_scalar<double>("f64");
  }
}
