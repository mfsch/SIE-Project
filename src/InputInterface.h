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

        /*
         * TODO for parallelization:
         * - adapt code for reading to only read subset 
         */

        std::vector<int> pd = process_distribution();
        std::vector<int> dim_index = indices_along_dimensions(pd);
        set_up_ranges(dim_index, pd);
        set_up_maps();

        // dump some information for debugging
        if (!mpi_rank_) {
            std::cout << "Process distribution:";
            for (int i=0;i<ND_;i++) std::cout << " " << pd[i];
            std::cout << std::endl << std::flush;
        }
        std::cout << "Indices for rank " << mpi_rank_ << ":  ";
        for (int i=0;i<ND_;i++) std::cout << " " << dim_index[i];
        std::cout << std::endl << std::flush;
        dump_ranges();
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
    std::vector<int>  start_; // interelement distances along rows
    std::vector<int>  count_; // interelement distances along rows
    std::vector<int>  row_map_; // interelement distances along rows
    std::vector<int>  col_map_; // interelement distances along column

    /*
     * Convenience function for development, no real use.
     */
    void dump_ranges() {
        std::cout << "Ranges for rank " << mpi_rank_ << ":";
        for (int d=0; d<ND_; d++) {
            std::cout << " " << start_[d] << "-" << start_[d] + count_[d] - 1;
        }
        std::cout << std::endl << std::flush;
    }

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

    /*
     * This function builds a vector with the number of MPI processes along
     * each dimension. The product of this vector is equal to the total
     * number of MPI processes. We assume this number to be a power of 2.
     */
    std::vector<int> process_distribution() {
        int N = mpi_size_;
        int d = 0;
        std::vector<int> pd(ND_,1);
        while (N>1) {
            if (!reduced_[d]) {// do not distribute reduced dimensions
                pd[d] *= 2;
                if (N%2) {
                    if (!mpi_rank_) std::cerr << "ERROR: The number of MPI processes must be divisible by 2." << std::endl;
                    exit(1);
                }
                N /= 2;
            }
            d = (d+1)%ND_;
        }
        return pd;
    }

    std::vector<int> indices_along_dimensions(std::vector<int> pd) {
        std::vector<int> dim_index(ND_,0);
        int r = mpi_rank_;
        int s = mpi_size_;
        for (int d=0; d<ND_; d++) {
            //if (reduced_[d]) continue;
            s /= pd[d];
            dim_index[d] = r / s;
            r = r % s;
        }
        return dim_index;
    }

    /*
     * Here, we decide which part of the data we read in the
     * current MPI process, based on its rank. We make an extra
     * effort to distribute it as evenly as possible as the actual
     * number of values each process reads is the product of the
     * counts. If we just always give the remainder to the last
     * rank, this process will end up with a lot more data and the
     * computation is not very balanced anymore.
     */
    void set_up_ranges(std::vector<int> dim_index, std::vector<int> pd) {

        count_ = std::vector<int>(ND_);
        start_ = std::vector<int>(ND_);

        for (int d=0; d<ND_; d++) {
            count_[d] = dims_[d] / pd[d];
            start_[d] = count_[d] * dim_index[d];

            // distribute remainder
            int remainder = dims_[d] % pd[d];
            if (dim_index[d] < remainder) {
                count_[d] += 1;
                start_[d] += dim_index[d];
            } else {
                start_[d] += remainder;
            }
        }
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
