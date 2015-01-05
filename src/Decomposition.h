#pragma once
#include <iostream>
#include <Eigen/Sparse>
#include <mpi.h>
#include "mpi_helper.h"


template<typename Scalar> class Decomposition {

public:

    /*
     * The constructor does all the work for the decomposition.
     * The N largest eigenvectors are saved internally.
     * Caution: The matrix X is passed by reference and will be changed
     * by the algorithm, as the mean will be subtracted.
     */
    Decomposition(Matrix<Scalar> &X, const int M, const int global_rows) {
        
        if (!mpi_rank_) std::cout << "Subtracting mean... " << std::flush;
        subtract_mean(X, global_rows);
        if (!mpi_rank_) std::cout << "done" << std::endl;

        lanczos(X, M, global_rows);
    }

    /*
     * This function returns the M largest eigenvalues.
     */
    RowVector<Scalar> eigenvalues() {
        return eigenvalues_.reverse(); // return largest to smallest
    }

    /*
     * This function extracts the j-th eigenvector.
     */
    RowVector<Scalar> eigenvector(const int j) {
        return eigenvectors_.col(j).transpose();
    }

    /*
     * This function projects the data on the j-th eigenvector.
     */
    ColVector<Scalar> projection(const Matrix<Scalar> &X, const int i) {
        int j = eigenvectors_.cols()-1-i; // EVs are in increasing order
        return X * eigenvectors_.col(j);
    }


private:

    // C++11 allows for dynamic initialization of class members
    const int mpi_size_ = MPI::COMM_WORLD.Get_size();
    const int mpi_rank_ = MPI::COMM_WORLD.Get_rank();

    RowVector<Scalar> mean_;
    ColVector<Scalar> eigenvalues_;
    Matrix<Scalar> eigenvectors_;


    void subtract_mean(Matrix<Scalar> &X, const int global_rows) {
        int N = X.cols();
        RowVector<Scalar> local_sum = X.colwise().sum();
        RowVector<Scalar> global_sum(N);
        MPI::COMM_WORLD.Allreduce(local_sum.data(), global_sum.data(), N,
                mpi_helper<Scalar>().type, MPI::SUM);
        mean_ = global_sum / global_rows;
        X.rowwise() -= mean_;
    }


    /*
     * This function uses the Lanczos method to find the M largest
     * eigenvalues and the corresponding eigenvectors of the matrix
     * X.T*X (dimensions NxN).
     *
     * Proper PCA should divide this matrix by (#rows-1), but as this
     * doesn't change the eigenvectors, we can omit this.
     */
    void lanczos(const Matrix<Scalar> &X, const int M, const int global_rows,
            const int max_it = 20, const Scalar tolerance = 1e-6) {

        // saved for convenience
        int N = X.cols();

        // Hessenberg matrix
        Matrix<Scalar> H(max_it+1, max_it+1);
        H.setZero();

        // matrix for subspace vectors
        Matrix<Scalar> V(N, max_it+1);
        V.col(0).setOnes(); // initial vector for Krylov subspace
        V.col(0) /= V.col(0).norm();

        // define sizes for eigenvalues and -vectors
        eigenvalues_  = ColVector<Scalar>(M);
        eigenvectors_ = Matrix<Scalar>(N,M);

        // variables for iterations are defined outside of loop
        Scalar alpha, beta;
        ColVector<Scalar> w_local(N);
        ColVector<Scalar> w(N);
        ColVector<Scalar> r_local(N);
        ColVector<Scalar> r(N);
        ColVector<Scalar> alphas(max_it);

        for (int i=0; i<max_it; i++) {

            // Lanczos iterations
            w_local = X.transpose() * (X * V.col(i)) / global_rows;
            MPI::COMM_WORLD.Allreduce(w_local.data(), w.data(), N,
                    mpi_helper<Scalar>().type, MPI::SUM);

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

#if DEBUG
            // test orthogonality
            Scalar orth = (V.leftCols(i+2).transpose()*V.leftCols(i+2) -
                    Matrix<Scalar>::Identity(i+2, i+2)).norm();
            if (!mpi_rank_) std::cout << "Orthogonality norm: " << orth << std::endl;
#endif

            // only find eigenvectors after subspace is large enough
            if (i>=M) {
                Eigen::SelfAdjointEigenSolver< Matrix<Scalar> >
                        eigensolver(H.topLeftCorner(i+1,i+1));
                eigenvalues_  = eigensolver.eigenvalues().bottomRows(M);
                eigenvectors_ = V.leftCols(i+1) *
                        eigensolver.eigenvectors().rightCols(M);

                // test eigenvector convergence
                Scalar error = 0;
                for (int k=0; k<M; k++) {
                    r_local = X.transpose() * (X * eigenvectors_.col(k)) / global_rows;
                    MPI::COMM_WORLD.Allreduce(r_local.data(), r.data(), N,
                            mpi_helper<Scalar>().type, MPI::SUM);
                    Scalar this_error = (r - eigenvectors_.col(k) *
                            eigenvalues_.row(k)).norm();
                    if (this_error > error) error = this_error;
                }
                error /= global_rows; // make errors independent of data size
                if (!mpi_rank_) std::cout << "Largest error norm: " << error
                        << std::endl;

                if (error < tolerance) {
                    if (!mpi_rank_) std::cout << "Reached tolerance level " << tolerance
                            << "." << std::endl;
                    return;
                }
            }
        }
        if (!mpi_rank_) std::cout << "Reached maximum number of iterations without convergence." << std::endl;
    }
};
