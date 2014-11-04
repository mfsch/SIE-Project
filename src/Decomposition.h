#include <iostream>
#include "Eigen/Sparse"

template<typename Scalar> class Decomposition {

public:
    /*
     * The constructor does all the work for the decomposition.
     * The N largest eigenvectors are saved internally.
     * Caution: The matrix X is passed by reference and will be changed
     * by the algorithm, as the mean will be subtracted.
     */
    Decomposition(Matrix<Scalar> &X, int M) {
        
        std::cout << "Subtracting mean... " << std::flush;
        subtract_mean(X);
        std::cout << "done" << X.mean() << std::endl;

        lanczos(X, M);

    }

private:

    RowVector<Scalar> mean_;
    Matrix<Scalar> eigenvectors_;
    Matrix<Scalar> eigenvalues_;

    void subtract_mean(Matrix<Scalar> &X) {
        mean_ = X.colwise().mean();
        X.rowwise() -= mean_;
    }

    /*
     * This function uses the Lanczos method to find the M largest
     * eigenvalues and the corresponding eigenvectors of the matrix
     * X.T*X (dimensions NxN)
     */
    void lanczos(Matrix<Scalar> &X, int M, int max_it = 20) {

        int N = X.cols();
        int Nr = X.rows();

        // Hessenberg matrix
        Matrix<Scalar> H(max_it+1, max_it+1);

        // Matrix for subspace vectors
        Matrix<Scalar> V(N, max_it+1);

        // initial vector for Krylov subspace
        V.col(0).setOnes();
        V.col(0) /= V.col(0).norm();

        // iterations
        Scalar alpha, beta;
        ColVector<Scalar> w(N);
        ColVector<Scalar> alphas(max_it);
        for (int i=0; i<max_it; i++) {

            w = X.transpose() * (X * V.col(i)) / (Nr-1);
            alpha = V.col(i).transpose() * w;
            w -= V.col(i) * alpha;
            if (i) w -= V.col(i-1) * beta;

            // reorthogonalization
            alphas.topRows(i+1) = V.leftCols(i+1).transpose() * w;
            w -= V.leftCols(i+1) * alphas.topRows(i+1);
            alpha += alphas(i);

            beta = w.norm();
            V.col(i+1) = w / beta;

            H(i,i) = alpha;
            H(i,i+1) = beta;
            H(i+1,i) = beta;

            // test orthogonality
            Scalar orth = (V.leftCols(i+2).transpose()*V.leftCols(i+2) - Eigen::MatrixXf::Identity(i+2, i+2)).norm();
            std::cout << "orthogonality norm: " << orth << std::endl;

            //H.cornerUpperLeft(i, i);
            if (i>=M) {
                //std::cout << H.topLeftCorner(i+1,i+1) << std::endl;
                Eigen::SelfAdjointEigenSolver< Matrix<Scalar> > eigensolver(H.topLeftCorner(i+1,i+1));

                eigenvalues_  = eigensolver.eigenvalues().bottomRows(M);
                eigenvectors_ = V.leftCols(i+1) * eigensolver.eigenvectors().rightCols(M);


                //Matrix<Scalar> A = eigenvectors_ * eigenvalues_.asDiagonal();
                    //(X.transpose() * (X * eigenvectors_));
                //std::cout << A.rows() << "x" << A.cols() << std::endl;


                std::cout << "eigenvector norm: " << (X.transpose() * (X * eigenvectors_) / (Nr-1) - eigenvectors_ * eigenvalues_.asDiagonal()).norm() << std::endl;
            }
        }
    }
};
