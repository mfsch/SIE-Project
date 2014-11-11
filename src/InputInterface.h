#pragma once
#include <iostream>
#include <boost/iostreams/device/mapped_file.hpp>
#include <Eigen/Dense>
#include <mpi.h>
#include "matrix_definition.h"

template<typename Scalar> class InputInterface {

public:
    InputInterface(std::vector<int> dimensions, std::vector<bool> reduced) {

        // save information about mpi
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);

        // make sure both vectors are the same length
        if (dimensions.size() != reduced.size()) {
            if (!mpi_rank_) std::cerr << "ERROR: The number of dimensions for the argument '--dimensions' and '--reduced' must be the same." << std::endl;
            exit(1);
        }
        ND_ = dimensions.size();

        dims_ = dimensions;
        reduced_ = reduced;
        row_map_ = std::vector<int>(ND_);
        col_map_ = std::vector<int>(ND_);

        set_up_maps();
        dump_maps();
    }

    Matrix<Scalar> read(std::string file_name) {

        if (!mpi_rank_) std::cout << "Input File: " << file_name << std::endl;
        boost::iostreams::mapped_file_source file(file_name);

        if (file.is_open()) {

            // raise error if sizes do not match
            size_t n_bytes = NR_ * NC_ * sizeof(Scalar);
            if (file.size() != n_bytes) {
                if (!mpi_rank_) std::cerr << "ERROR: File does not match specified size (expected " << n_bytes << "B, file has " << file.size() << "B)." << std::endl;
                file.close();
                exit(1);
            }

            // create matrix and load data from file
            if (!mpi_rank_) std::cout << "File opened successfully." << std::endl;
            const Scalar *data = reinterpret_cast<const Scalar*>(file.data());
            if (!mpi_rank_) std::cout << "Reordering matrix... " << std::flush;
            Matrix<Scalar> matrix = load_matrix(data);
            if (!mpi_rank_) std::cout << "done" << std::endl;
            file.close();
            return matrix;

        } else {
            if (!mpi_rank_) std::cerr << "ERROR: Could not open file: " << file_name << std::endl;
            exit(1);
        }
    }


private:
    int mpi_size_;
    int mpi_rank_;
    int ND_; // number of dimensions
    int NR_; // number of rows
    int NC_; // number of columns
    int N_;  // total number of values
    std::vector<int>  dims_;    // length of dimensions
    std::vector<bool> reduced_; // whether dimensions are reduced (along columns)
    std::vector<int>  row_map_; // interelement distances along rows
    std::vector<int>  col_map_; // interelement distances along column

    /*
     * Convenience function for development, no real use.
     */
    void dump_maps() {
        if (mpi_rank_) return; // do nothing unless rank 0
        std::cout << "row map:\t";
        for (int i=0; i<ND_; i++) { std::cout << row_map_[i] << "\t"; }
        std::cout << std::endl;
        std::cout << "col map:\t";
        for (int i=0; i<ND_; i++) { std::cout << col_map_[i] << "\t"; }
        std::cout << std::endl;
    }

    void set_up_maps() {

        int row_interelement_distance = 1;
        int col_interelement_distance = 1;

        // go through dimensions, fastest varying first
        for (int i=0; i<ND_; i++) {
            if (reduced_[i]) {
                //place along columns
                row_map_[i] = 0;
                col_map_[i] = col_interelement_distance;
                col_interelement_distance *= dims_[i];
            } else {
                // place along rows
                row_map_[i] = row_interelement_distance;
                col_map_[i] = 0;
                row_interelement_distance *= dims_[i];
            }

        }

        // set matrix size
        NR_ = row_interelement_distance;
        NC_ = col_interelement_distance;
        N_  = NR_*NC_;
        if (!mpi_rank_) std::cout << "Matrix size: " << NR_ << "x" << NC_ << std::endl;
    }

    
    /*
     * observations along columns
     * observations are reduced
     * -> columns are reduced
     */
    Matrix<Scalar> load_matrix(const Scalar *data) {
        Matrix<Scalar> matrix(NR_, NC_);

        bool carry;
        std::vector<int> dim_index(ND_, 0);
        int row_index, col_index;
        for (int i=0; i<N_; i++) {
        //for (int i=0; i<200000; i++) { // use this line for subset of values
            row_index = 0;
            col_index = 0;
            carry = true;
            for (int d=0; d<ND_; d++) {
                row_index += dim_index[d] * row_map_[d];
                col_index += dim_index[d] * col_map_[d];
                // increment dim_index for next iteration
                if (carry) {
                    dim_index[d]++;
                    if (dim_index[d] == dims_[d]) {
                        dim_index[d] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            matrix(row_index, col_index) = data[i];
        }
        //std::cout << matrix.bottomRightCorner(10,10) << std::endl;
        return matrix;
    }
};
