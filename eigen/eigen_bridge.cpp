#include <Eigen/Dense>

#include <cstddef>

template <typename Scalar>
using DynamicMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename Scalar>
using DynamicVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
void add(const Scalar* lhs, const Scalar* rhs, std::size_t rows, std::size_t columns, Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> lhs_map(lhs, rows, columns);
  Eigen::Map<const DynamicMatrix<Scalar>> rhs_map(rhs, rows, columns);
  Eigen::Map<DynamicMatrix<Scalar>> output_map(output, rows, columns);
  output_map.noalias() = lhs_map + rhs_map;
}

template <typename Scalar>
void transpose(const Scalar* input, std::size_t rows, std::size_t columns, Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, rows, columns);
  Eigen::Map<DynamicMatrix<Scalar>> output_map(output, columns, rows);
  output_map.noalias() = input_map.transpose();
}

template <typename Scalar>
Scalar norm(const Scalar* input, std::size_t rows, std::size_t columns) {
  Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, rows, columns);
  return input_map.norm();
}

template <typename Scalar>
Scalar squared_norm(const Scalar* input, std::size_t rows, std::size_t columns) {
  Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, rows, columns);
  return input_map.squaredNorm();
}

template <typename Scalar>
Scalar dot(const Scalar* lhs, const Scalar* rhs, std::size_t size) {
  Eigen::Map<const DynamicVector<Scalar>> lhs_map(lhs, size);
  Eigen::Map<const DynamicVector<Scalar>> rhs_map(rhs, size);
  return lhs_map.dot(rhs_map);
}

template <typename Scalar>
void normalize(const Scalar* input, std::size_t rows, std::size_t columns, Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, rows, columns);
  Eigen::Map<DynamicMatrix<Scalar>> output_map(output, rows, columns);
  output_map.noalias() = input_map / input_map.norm();
}

template <typename Scalar>
void matmul(const Scalar* lhs, const Scalar* rhs, std::size_t rows, std::size_t shared,
            std::size_t columns, Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> lhs_map(lhs, rows, shared);
  Eigen::Map<const DynamicMatrix<Scalar>> rhs_map(rhs, shared, columns);
  Eigen::Map<DynamicMatrix<Scalar>> output_map(output, rows, columns);
  output_map.noalias() = lhs_map * rhs_map;
}

template <typename Scalar>
Scalar determinant(const Scalar* input, std::size_t dimension) {
  Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, dimension, dimension);
  return input_map.determinant();
}

template <typename Scalar>
void inverse(const Scalar* input, std::size_t dimension, Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, dimension, dimension);
  Eigen::Map<DynamicMatrix<Scalar>> output_map(output, dimension, dimension);
  output_map.noalias() = input_map.inverse();
}

template <typename Scalar>
void solve(const Scalar* input, const Scalar* rhs, std::size_t dimension, std::size_t columns,
           Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, dimension, dimension);
  Eigen::Map<const DynamicMatrix<Scalar>> rhs_map(rhs, dimension, columns);
  Eigen::Map<DynamicMatrix<Scalar>> output_map(output, dimension, columns);
  output_map.noalias() = input_map.partialPivLu().solve(rhs_map);
}

template <typename Scalar>
int llt_solve(const Scalar* input, const Scalar* rhs, std::size_t dimension, std::size_t columns,
              Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, dimension, dimension);
  Eigen::Map<const DynamicMatrix<Scalar>> rhs_map(rhs, dimension, columns);
  Eigen::Map<DynamicMatrix<Scalar>> output_map(output, dimension, columns);
  auto factor = input_map.llt();
  if (factor.info() != Eigen::Success) {
    return 0;
  }
  output_map.noalias() = factor.solve(rhs_map);
  return 1;
}

template <typename Scalar>
int ldlt_solve(const Scalar* input, const Scalar* rhs, std::size_t dimension, std::size_t columns,
               Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, dimension, dimension);
  Eigen::Map<const DynamicMatrix<Scalar>> rhs_map(rhs, dimension, columns);
  Eigen::Map<DynamicMatrix<Scalar>> output_map(output, dimension, columns);
  auto factor = input_map.ldlt();
  if (factor.info() != Eigen::Success) {
    return 0;
  }
  output_map.noalias() = factor.solve(rhs_map);
  return 1;
}

#define DEFINE_ORACLE_WRAPPERS(SUFFIX, SCALAR)                                                   \
  extern "C" void sa_eigen_add_##SUFFIX(const SCALAR* lhs, const SCALAR* rhs, std::size_t rows, \
                                           std::size_t columns, SCALAR* output) {                 \
    add(lhs, rhs, rows, columns, output);                                                         \
  }                                                                                                \
  extern "C" void sa_eigen_transpose_##SUFFIX(const SCALAR* input, std::size_t rows,             \
                                                 std::size_t columns, SCALAR* output) {             \
    transpose(input, rows, columns, output);                                                      \
  }                                                                                                \
  extern "C" SCALAR sa_eigen_norm_##SUFFIX(const SCALAR* input, std::size_t rows,                \
                                              std::size_t columns) {                                \
    return norm(input, rows, columns);                                                            \
  }                                                                                                \
  extern "C" SCALAR sa_eigen_squared_norm_##SUFFIX(const SCALAR* input, std::size_t rows,        \
                                                     std::size_t columns) {                         \
    return squared_norm(input, rows, columns);                                                    \
  }                                                                                                \
  extern "C" SCALAR sa_eigen_dot_##SUFFIX(const SCALAR* lhs, const SCALAR* rhs,                  \
                                             std::size_t size) {                                   \
    return dot(lhs, rhs, size);                                                                    \
  }                                                                                                \
  extern "C" void sa_eigen_normalize_##SUFFIX(const SCALAR* input, std::size_t rows,             \
                                                 std::size_t columns, SCALAR* output) {            \
    normalize(input, rows, columns, output);                                                       \
  }                                                                                                \
  extern "C" void sa_eigen_matmul_##SUFFIX(const SCALAR* lhs, const SCALAR* rhs,                 \
                                              std::size_t rows, std::size_t shared,                 \
                                              std::size_t columns, SCALAR* output) {                \
    matmul(lhs, rhs, rows, shared, columns, output);                                               \
  }                                                                                                \
  extern "C" SCALAR sa_eigen_determinant_##SUFFIX(const SCALAR* input, std::size_t dimension) { \
    return determinant(input, dimension);                                                         \
  }                                                                                                \
  extern "C" void sa_eigen_inverse_##SUFFIX(const SCALAR* input, std::size_t dimension,          \
                                               SCALAR* output) {                                    \
    inverse(input, dimension, output);                                                            \
  }                                                                                                \
  extern "C" void sa_eigen_solve_##SUFFIX(const SCALAR* input, const SCALAR* rhs,                \
                                            std::size_t dimension, std::size_t columns,             \
                                            SCALAR* output) {                                       \
    solve(input, rhs, dimension, columns, output);                                                \
  }                                                                                                \
  extern "C" int sa_eigen_llt_solve_##SUFFIX(const SCALAR* input, const SCALAR* rhs,             \
                                                std::size_t dimension, std::size_t columns,         \
                                                SCALAR* output) {                                    \
    return llt_solve(input, rhs, dimension, columns, output);                                     \
  }                                                                                                \
  extern "C" int sa_eigen_ldlt_solve_##SUFFIX(const SCALAR* input, const SCALAR* rhs,            \
                                                 std::size_t dimension, std::size_t columns,        \
                                                 SCALAR* output) {                                   \
    return ldlt_solve(input, rhs, dimension, columns, output);                                    \
  }

DEFINE_ORACLE_WRAPPERS(f32, float)
DEFINE_ORACLE_WRAPPERS(f64, double)
