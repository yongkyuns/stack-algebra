#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SparseCholesky>

#include <cstddef>
#include <new>

template <typename Scalar>
using DynamicMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename Scalar>
using DynamicVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;

template <typename Scalar>
struct SparseLltContext {
  SparseMatrix<Scalar> matrix;
  Eigen::SimplicialLLT<SparseMatrix<Scalar>> factor;
};

template <typename Scalar>
SparseLltContext<Scalar>* sparse_llt_create(const std::size_t* row_indices,
                                            const std::size_t* column_starts,
                                            const Scalar* values, std::size_t dimension,
                                            std::size_t nonzeros) {
  try {
    auto* context = new SparseLltContext<Scalar>();
    context->matrix.resize(static_cast<int>(dimension), static_cast<int>(dimension));
    context->matrix.reserve(static_cast<int>(nonzeros));
    for (std::size_t column = 0; column < dimension; ++column) {
      const std::size_t start = column_starts[column];
      const std::size_t end = column + 1 < dimension ? column_starts[column + 1] : nonzeros;
      for (std::size_t index = start; index < end; ++index) {
        context->matrix.insert(static_cast<int>(row_indices[index]), static_cast<int>(column)) =
            values[index];
      }
    }
    context->matrix.makeCompressed();
    return context;
  } catch (...) {
    return nullptr;
  }
}

template <typename Scalar>
int sparse_llt_analyze(SparseLltContext<Scalar>* context) {
  context->factor.analyzePattern(context->matrix);
  return context->factor.info() == Eigen::Success ? 1 : 0;
}

template <typename Scalar>
int sparse_llt_factorize(SparseLltContext<Scalar>* context) {
  context->factor.factorize(context->matrix);
  return context->factor.info() == Eigen::Success ? 1 : 0;
}

template <typename Scalar>
int sparse_llt_solve(SparseLltContext<Scalar>* context, const Scalar* rhs, std::size_t columns,
                    Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> rhs_map(rhs, context->matrix.rows(), columns);
  Eigen::Map<DynamicMatrix<Scalar>> output_map(output, context->matrix.rows(), columns);
  output_map = context->factor.solve(rhs_map);
  return context->factor.info() == Eigen::Success ? 1 : 0;
}

template <typename Scalar>
int sparse_llt_matvec(SparseLltContext<Scalar>* context, const Scalar* rhs, Scalar* output) {
  Eigen::Map<const DynamicVector<Scalar>> rhs_map(rhs, context->matrix.rows());
  Eigen::Map<DynamicVector<Scalar>> output_map(output, context->matrix.rows());
  output_map.noalias() = context->matrix * rhs_map;
  return 1;
}

template <typename Scalar>
void sparse_llt_destroy(SparseLltContext<Scalar>* context) {
  delete context;
}

template <typename Scalar>
struct SparseLdltContext {
  SparseMatrix<Scalar> matrix;
  Eigen::SimplicialLDLT<SparseMatrix<Scalar>> factor;
};

template <typename Scalar>
SparseLdltContext<Scalar>* sparse_ldlt_create(const std::size_t* row_indices,
                                              const std::size_t* column_starts,
                                              const Scalar* values, std::size_t dimension,
                                              std::size_t nonzeros) {
  try {
    auto* context = new SparseLdltContext<Scalar>();
    context->matrix.resize(static_cast<int>(dimension), static_cast<int>(dimension));
    context->matrix.reserve(static_cast<int>(nonzeros));
    for (std::size_t column = 0; column < dimension; ++column) {
      const std::size_t start = column_starts[column];
      const std::size_t end = column + 1 < dimension ? column_starts[column + 1] : nonzeros;
      for (std::size_t index = start; index < end; ++index) {
        context->matrix.insert(static_cast<int>(row_indices[index]), static_cast<int>(column)) =
            values[index];
      }
    }
    context->matrix.makeCompressed();
    return context;
  } catch (...) {
    return nullptr;
  }
}

template <typename Scalar>
int sparse_ldlt_analyze(SparseLdltContext<Scalar>* context) {
  context->factor.analyzePattern(context->matrix);
  return context->factor.info() == Eigen::Success ? 1 : 0;
}

template <typename Scalar>
int sparse_ldlt_factorize(SparseLdltContext<Scalar>* context) {
  context->factor.factorize(context->matrix);
  return context->factor.info() == Eigen::Success ? 1 : 0;
}

template <typename Scalar>
int sparse_ldlt_solve(SparseLdltContext<Scalar>* context, const Scalar* rhs, std::size_t columns,
                      Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> rhs_map(rhs, context->matrix.rows(), columns);
  Eigen::Map<DynamicMatrix<Scalar>> output_map(output, context->matrix.rows(), columns);
  output_map = context->factor.solve(rhs_map);
  return context->factor.info() == Eigen::Success ? 1 : 0;
}

template <typename Scalar>
void sparse_ldlt_destroy(SparseLdltContext<Scalar>* context) {
  delete context;
}

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
void qr_solve(const Scalar* input, const Scalar* rhs, std::size_t rows, std::size_t columns,
              std::size_t rhs_columns, Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, rows, columns);
  Eigen::Map<const DynamicMatrix<Scalar>> rhs_map(rhs, rows, rhs_columns);
  Eigen::Map<DynamicMatrix<Scalar>> output_map(output, columns, rhs_columns);
  output_map.noalias() = input_map.householderQr().solve(rhs_map);
}

template <typename Scalar>
void col_piv_qr_solve(const Scalar* input, const Scalar* rhs, std::size_t rows,
                      std::size_t columns, std::size_t rhs_columns, Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, rows, columns);
  Eigen::Map<const DynamicMatrix<Scalar>> rhs_map(rhs, rows, rhs_columns);
  Eigen::Map<DynamicMatrix<Scalar>> output_map(output, columns, rhs_columns);
  output_map.noalias() = input_map.colPivHouseholderQr().solve(rhs_map);
}

template <typename Scalar>
void svd_singular_values(const Scalar* input, std::size_t rows, std::size_t columns,
                         Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, rows, columns);
  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> output_map(output, columns);
  using Svd = Eigen::JacobiSVD<DynamicMatrix<Scalar>>;
  output_map = Svd(input_map, Eigen::ComputeThinU | Eigen::ComputeThinV).singularValues();
}

template <typename Scalar>
void svd_solve(const Scalar* input, const Scalar* rhs, std::size_t rows, std::size_t columns,
               std::size_t rhs_columns, Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, rows, columns);
  Eigen::Map<const DynamicMatrix<Scalar>> rhs_map(rhs, rows, rhs_columns);
  Eigen::Map<DynamicMatrix<Scalar>> output_map(output, columns, rhs_columns);
  using Svd = Eigen::JacobiSVD<DynamicMatrix<Scalar>>;
  output_map.noalias() = Svd(input_map, Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs_map);
}

template <typename Scalar>
void self_adjoint_eigenvalues(const Scalar* input, std::size_t dimension, Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, dimension, dimension);
  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> output_map(output, dimension);
  output_map = Eigen::SelfAdjointEigenSolver<DynamicMatrix<Scalar>>(input_map).eigenvalues();
}

template <typename Scalar>
void self_adjoint_eigenvectors(const Scalar* input, std::size_t dimension, Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, dimension, dimension);
  Eigen::Map<DynamicMatrix<Scalar>> output_map(output, dimension, dimension);
  output_map = Eigen::SelfAdjointEigenSolver<DynamicMatrix<Scalar>>(input_map).eigenvectors();
}

template <typename Scalar>
void triangular_solve(const Scalar* input, const Scalar* rhs, std::size_t dimension,
                      std::size_t columns, bool lower, Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, dimension, dimension);
  Eigen::Map<const DynamicMatrix<Scalar>> rhs_map(rhs, dimension, columns);
  Eigen::Map<DynamicMatrix<Scalar>> output_map(output, dimension, columns);
  if (lower) {
    output_map.noalias() = input_map.template triangularView<Eigen::Lower>().solve(rhs_map);
  } else {
    output_map.noalias() = input_map.template triangularView<Eigen::Upper>().solve(rhs_map);
  }
}

template <typename Scalar>
void triangular_mul(const Scalar* input, const Scalar* rhs, std::size_t dimension,
                    std::size_t columns, bool lower, Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, dimension, dimension);
  Eigen::Map<const DynamicMatrix<Scalar>> rhs_map(rhs, dimension, columns);
  Eigen::Map<DynamicMatrix<Scalar>> output_map(output, dimension, columns);
  if (lower) {
    output_map.noalias() = input_map.template triangularView<Eigen::Lower>() * rhs_map;
  } else {
    output_map.noalias() = input_map.template triangularView<Eigen::Upper>() * rhs_map;
  }
}

template <typename Scalar>
void quaternion_rotation(const Scalar* quaternion, Scalar* matrix_output, Scalar* vector_output,
                         const Scalar* vector_input) {
  Eigen::Quaternion<Scalar> rotation(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
  Eigen::Map<DynamicMatrix<Scalar>> matrix_map(matrix_output, 3, 3);
  Eigen::Map<const DynamicVector<Scalar>> vector_map(vector_input, 3);
  Eigen::Map<DynamicVector<Scalar>> output_map(vector_output, 3);
  rotation.normalize();
  matrix_map = rotation.toRotationMatrix();
  output_map = rotation * vector_map;
}

template <typename Scalar>
void isometry_transform(const Scalar* quaternion, const Scalar* translation,
                        const Scalar* point, Scalar* matrix_output, Scalar* point_output) {
  Eigen::Quaternion<Scalar> rotation(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
  Eigen::Transform<Scalar, 3, Eigen::Isometry> transform =
      Eigen::Transform<Scalar, 3, Eigen::Isometry>::Identity();
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  Eigen::Map<const Vector3> translation_map(translation);
  Eigen::Map<const Vector3> point_map(point);
  Eigen::Map<DynamicMatrix<Scalar>> matrix_map(matrix_output, 4, 4);
  Eigen::Map<Vector3> output_map(point_output);
  rotation.normalize();
  transform.linear() = rotation.toRotationMatrix();
  transform.translation() = translation_map;
  matrix_map = transform.matrix();
  output_map = transform * point_map;
}

template <typename Scalar>
void affine_transform(const Scalar* matrix, const Scalar* point, Scalar* point_output) {
  using Matrix4 = Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor>;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  Eigen::Map<const Matrix4> matrix_map(matrix);
  Eigen::Map<const Vector3> point_map(point);
  Eigen::Map<Vector3> output_map(point_output);
  Eigen::Transform<Scalar, 3, Eigen::Affine> transform(matrix_map);
  output_map = transform * point_map;
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

template <typename Scalar>
struct DenseLdltContext {
  DynamicMatrix<Scalar> matrix;
  Eigen::LDLT<DynamicMatrix<Scalar>> factor;
};

template <typename Scalar>
DenseLdltContext<Scalar>* dense_ldlt_create(const Scalar* input, std::size_t dimension) {
  try {
    auto* context = new DenseLdltContext<Scalar>();
    Eigen::Map<const DynamicMatrix<Scalar>> input_map(input, dimension, dimension);
    context->matrix = input_map;
    return context;
  } catch (...) {
    return nullptr;
  }
}

template <typename Scalar>
int dense_ldlt_factorize(DenseLdltContext<Scalar>* context) {
  context->factor.compute(context->matrix);
  return context->factor.info() == Eigen::Success ? 1 : 0;
}

template <typename Scalar>
int dense_ldlt_solve(DenseLdltContext<Scalar>* context, const Scalar* rhs,
                     std::size_t columns, Scalar* output) {
  Eigen::Map<const DynamicMatrix<Scalar>> rhs_map(rhs, context->matrix.rows(), columns);
  Eigen::Map<DynamicMatrix<Scalar>> output_map(output, context->matrix.rows(), columns);
  output_map.noalias() = context->factor.solve(rhs_map);
  return context->factor.info() == Eigen::Success ? 1 : 0;
}

template <typename Scalar>
void dense_ldlt_destroy(DenseLdltContext<Scalar>* context) {
  delete context;
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
  extern "C" void sa_eigen_qr_solve_##SUFFIX(const SCALAR* input, const SCALAR* rhs,             \
                                                std::size_t rows, std::size_t columns,              \
                                                std::size_t rhs_columns, SCALAR* output) {           \
    qr_solve(input, rhs, rows, columns, rhs_columns, output);                                      \
  }                                                                                                \
  extern "C" void sa_eigen_col_piv_qr_solve_##SUFFIX(                                             \
      const SCALAR* input, const SCALAR* rhs, std::size_t rows, std::size_t columns,              \
      std::size_t rhs_columns, SCALAR* output) {                                                    \
    col_piv_qr_solve(input, rhs, rows, columns, rhs_columns, output);                               \
  }                                                                                                \
  extern "C" void sa_eigen_svd_singular_values_##SUFFIX(                                           \
      const SCALAR* input, std::size_t rows, std::size_t columns, SCALAR* output) {                \
    svd_singular_values(input, rows, columns, output);                                              \
  }                                                                                                \
  extern "C" void sa_eigen_svd_solve_##SUFFIX(                                                     \
      const SCALAR* input, const SCALAR* rhs, std::size_t rows, std::size_t columns,              \
      std::size_t rhs_columns, SCALAR* output) {                                                    \
    svd_solve(input, rhs, rows, columns, rhs_columns, output);                                      \
  }                                                                                                \
  extern "C" void sa_eigen_self_adjoint_eigenvalues_##SUFFIX(                                      \
      const SCALAR* input, std::size_t dimension, SCALAR* output) {                                 \
    self_adjoint_eigenvalues(input, dimension, output);                                             \
  }                                                                                                \
  extern "C" void sa_eigen_self_adjoint_eigenvectors_##SUFFIX(                                      \
      const SCALAR* input, std::size_t dimension, SCALAR* output) {                                 \
    self_adjoint_eigenvectors(input, dimension, output);                                             \
  }                                                                                                \
  extern "C" void sa_eigen_lower_triangular_solve_##SUFFIX(                                         \
      const SCALAR* input, const SCALAR* rhs, std::size_t dimension, std::size_t columns,          \
      SCALAR* output) {                                                                             \
    triangular_solve(input, rhs, dimension, columns, true, output);                                 \
  }                                                                                                \
  extern "C" void sa_eigen_upper_triangular_solve_##SUFFIX(                                         \
      const SCALAR* input, const SCALAR* rhs, std::size_t dimension, std::size_t columns,          \
      SCALAR* output) {                                                                             \
    triangular_solve(input, rhs, dimension, columns, false, output);                                \
  }                                                                                                \
  extern "C" void sa_eigen_lower_triangular_mul_##SUFFIX(                                           \
      const SCALAR* input, const SCALAR* rhs, std::size_t dimension, std::size_t columns,          \
      SCALAR* output) {                                                                             \
    triangular_mul(input, rhs, dimension, columns, true, output);                                   \
  }                                                                                                \
  extern "C" void sa_eigen_upper_triangular_mul_##SUFFIX(                                           \
      const SCALAR* input, const SCALAR* rhs, std::size_t dimension, std::size_t columns,          \
      SCALAR* output) {                                                                             \
    triangular_mul(input, rhs, dimension, columns, false, output);                                  \
  }                                                                                                \
  extern "C" void sa_eigen_quaternion_rotation_##SUFFIX(                                           \
      const SCALAR* quaternion, SCALAR* matrix_output, SCALAR* vector_output,                     \
      const SCALAR* vector_input) {                                                                \
    quaternion_rotation(quaternion, matrix_output, vector_output, vector_input);                   \
  }                                                                                                \
  extern "C" void sa_eigen_isometry_transform_##SUFFIX(                                            \
      const SCALAR* quaternion, const SCALAR* translation, const SCALAR* point,                   \
      SCALAR* matrix_output, SCALAR* point_output) {                                               \
    isometry_transform(quaternion, translation, point, matrix_output, point_output);               \
  }                                                                                                \
  extern "C" void sa_eigen_affine_transform_##SUFFIX(                                              \
      const SCALAR* matrix, const SCALAR* point, SCALAR* point_output) {                            \
    affine_transform(matrix, point, point_output);                                                 \
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

#define DEFINE_DENSE_LDLT_WRAPPERS(SUFFIX, SCALAR)                                                \
  extern "C" void* sa_eigen_dense_ldlt_create_##SUFFIX(const SCALAR* input,                     \
                                                         std::size_t dimension) {                  \
    return dense_ldlt_create(input, dimension);                                                    \
  }                                                                                                 \
  extern "C" int sa_eigen_dense_ldlt_factorize_##SUFFIX(void* opaque) {                          \
    return dense_ldlt_factorize(static_cast<DenseLdltContext<SCALAR>*>(opaque));                  \
  }                                                                                                 \
  extern "C" int sa_eigen_dense_ldlt_solve_##SUFFIX(                                             \
      void* opaque, const SCALAR* rhs, std::size_t columns, SCALAR* output) {                     \
    return dense_ldlt_solve(static_cast<DenseLdltContext<SCALAR>*>(opaque), rhs, columns, output); \
  }                                                                                                 \
  extern "C" void sa_eigen_dense_ldlt_destroy_##SUFFIX(void* opaque) {                            \
    dense_ldlt_destroy(static_cast<DenseLdltContext<SCALAR>*>(opaque));                            \
  }

DEFINE_DENSE_LDLT_WRAPPERS(f32, float)
DEFINE_DENSE_LDLT_WRAPPERS(f64, double)

#define DEFINE_SPARSE_LLT_WRAPPERS(SUFFIX, SCALAR)                                               \
  extern "C" void* sa_eigen_sparse_llt_create_##SUFFIX(                                         \
      const std::size_t* row_indices, const std::size_t* column_starts, const SCALAR* values,     \
      std::size_t dimension, std::size_t nonzeros) {                                               \
    return sparse_llt_create(row_indices, column_starts, values, dimension, nonzeros);             \
  }                                                                                                 \
  extern "C" int sa_eigen_sparse_llt_analyze_##SUFFIX(void* opaque) {                             \
    return sparse_llt_analyze(static_cast<SparseLltContext<SCALAR>*>(opaque));                     \
  }                                                                                                 \
  extern "C" int sa_eigen_sparse_llt_factorize_##SUFFIX(void* opaque) {                          \
    return sparse_llt_factorize(static_cast<SparseLltContext<SCALAR>*>(opaque));                  \
  }                                                                                                 \
  extern "C" int sa_eigen_sparse_llt_solve_##SUFFIX(                                              \
      void* opaque, const SCALAR* rhs, std::size_t columns, SCALAR* output) {                     \
    return sparse_llt_solve(static_cast<SparseLltContext<SCALAR>*>(opaque), rhs, columns, output); \
  }                                                                                                 \
  extern "C" int sa_eigen_sparse_llt_matvec_##SUFFIX(                                             \
      void* opaque, const SCALAR* rhs, SCALAR* output) {                                            \
    return sparse_llt_matvec(static_cast<SparseLltContext<SCALAR>*>(opaque), rhs, output);          \
  }                                                                                                 \
  extern "C" void sa_eigen_sparse_llt_destroy_##SUFFIX(void* opaque) {                           \
    sparse_llt_destroy(static_cast<SparseLltContext<SCALAR>*>(opaque));                            \
  }

DEFINE_SPARSE_LLT_WRAPPERS(f32, float)
DEFINE_SPARSE_LLT_WRAPPERS(f64, double)

#define DEFINE_SPARSE_LDLT_WRAPPERS(SUFFIX, SCALAR)                                               \
  extern "C" void* sa_eigen_sparse_ldlt_create_##SUFFIX(                                        \
      const std::size_t* row_indices, const std::size_t* column_starts, const SCALAR* values,     \
      std::size_t dimension, std::size_t nonzeros) {                                               \
    return sparse_ldlt_create(row_indices, column_starts, values, dimension, nonzeros);            \
  }                                                                                                 \
  extern "C" int sa_eigen_sparse_ldlt_analyze_##SUFFIX(void* opaque) {                            \
    return sparse_ldlt_analyze(static_cast<SparseLdltContext<SCALAR>*>(opaque));                  \
  }                                                                                                 \
  extern "C" int sa_eigen_sparse_ldlt_factorize_##SUFFIX(void* opaque) {                         \
    return sparse_ldlt_factorize(static_cast<SparseLdltContext<SCALAR>*>(opaque));                \
  }                                                                                                 \
  extern "C" int sa_eigen_sparse_ldlt_solve_##SUFFIX(                                             \
      void* opaque, const SCALAR* rhs, std::size_t columns, SCALAR* output) {                    \
    return sparse_ldlt_solve(static_cast<SparseLdltContext<SCALAR>*>(opaque), rhs, columns, output); \
  }                                                                                                 \
  extern "C" void sa_eigen_sparse_ldlt_destroy_##SUFFIX(void* opaque) {                          \
    sparse_ldlt_destroy(static_cast<SparseLdltContext<SCALAR>*>(opaque));                         \
  }

DEFINE_SPARSE_LDLT_WRAPPERS(f32, float)
DEFINE_SPARSE_LDLT_WRAPPERS(f64, double)
