#include "Eigen/Dense"
template<typename Scalar> using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
template<typename Scalar> using RowVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
template<typename Scalar> using ColVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
