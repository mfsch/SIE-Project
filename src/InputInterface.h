#pragma once
#include <iostream>
#include <boost/iostreams/device/mapped_file.hpp>
#include <Eigen/Dense>
#include <mpi.h>
#include "matrix_definition.h"

template<typename Scalar> class InputInterface {

public:

    int global_rows; // number of rows


    InputInterface(const std::vector<int> &dimensions, 
            const std::vector<bool> &reduced) {

        // make sure both vectors are the same length
        if (dimensions.size() != reduced.size()) {
            if (!mpi_rank_) std::cerr << "ERROR: The number of "
                    << "dimensions for the argument '--dimensions' and "
                    << "'--reduced' must be the same." << std::endl;
            exit(1);
        }
        ND_ = dimensions.size();

        // prepare data for reading values
        set_up_ranges(dimensions, reduced);
        set_up_maps(dimensions, reduced);

        // dump some information for debugging
        //dump_info();
    }


    Matrix<Scalar> read(const std::string &file_name) {
        // TODO: look at possibility of using Eigen Map for this array

        if (!mpi_rank_) std::cout << "Input File: " << file_name << std::endl;
        boost::iostreams::mapped_file_source file(file_name);

        if (file.is_open()) {

            // raise error if sizes do not match
            size_t n_bytes = N_global_ * sizeof(Scalar);
            if (file.size() != n_bytes) {
                if (!mpi_rank_) std::cerr << "ERROR: File does not match "
                        << "specified size (expected " << n_bytes << "B, file "
                        << "has " << file.size() << "B)." << std::endl;
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
            if (!mpi_rank_) std::cerr << "ERROR: Could not open file: "
                    << file_name << std::endl;
            exit(1);
        }
    }


private:

    // C++11 allows for dynamic initialization of class members
    const int mpi_size_ = MPI::COMM_WORLD.Get_size();
    const int mpi_rank_ = MPI::COMM_WORLD.Get_rank();

    // set up in constructor
    int ND_; // number of dimensions

    // set up in set_up_ranges()
    std::vector<int>  start_; // interelement distances along rows
    std::vector<int>  count_; // interelement distances along rows

    // set up in set_up_maps()
    int NR_; // number of rows
    int NC_; // number of columns
    int N_;  // total number of values
    int N_global_;  // total number of values
    std::vector<int>  global_map_; // interelement distances in original data
    std::vector<int>  row_map_; // interelement distances along rows
    std::vector<int>  col_map_; // interelement distances along column


    /*
     * Dump information on data ranges and maps.
     * Convenience function for development, no real use.
     */
    void dump_info() {
        // data ranges
        std::cout << "Ranges for rank " << mpi_rank_ << ":";
        for (int d=0; d<ND_; d++) std::cout << "\t" << start_[d] << "-"
            << start_[d] + count_[d] - 1;
        std::cout << std::endl;
        MPI::COMM_WORLD.Barrier();

        // maps
        std::cout << "Row map for rank " << mpi_rank_ << ":";
        for (int d=0; d<ND_; d++) { std::cout << "\t" << row_map_[d]; }
        std::cout << std::endl;
        MPI::COMM_WORLD.Barrier();
        std::cout << "Col map for rank " << mpi_rank_ << ":";
        for (int d=0; d<ND_; d++) { std::cout << "\t" << col_map_[d]; }
        std::cout << std::endl;
        MPI::COMM_WORLD.Barrier();
    }


    /*
     * Dump information on process distribution.
     * Convenience function for development, no real use.
     */
    void dump_pd_info(const std::vector<int> &pd,
            const std::vector<int> &dim_index) {
        // process distribution
        if (!mpi_rank_) {
            std::cout << "Process distribution:";
            for (int i=0;i<ND_;i++) std::cout << "\t" << pd[i];
            std::cout << std::endl;
        }
        MPI::COMM_WORLD.Barrier();

        // dimension indices
        std::cout << "Indices for rank " << mpi_rank_ << ":";
        for (int d=0;d<ND_;d++) std::cout << "\t" << dim_index[d];
        std::cout << std::endl;
        MPI::COMM_WORLD.Barrier();
    }


    /*
     * This function builds a vector with the number of MPI processes along
     * each dimension. The product of this vector is equal to the total
     * number of MPI processes. We assume this number to be a power of 2.
     */
    std::vector<int> process_distribution(const std::vector<bool> &reduced) {
        int N = mpi_size_;
        int d = 0;
        std::vector<int> pd(ND_,1);
        while (N>1) {
            if (!reduced[d]) {// do not distribute reduced dimensions
                pd[d] *= 2;
                if (N%2) {
                    if (!mpi_rank_) std::cerr << "ERROR: The number of MPI "
                            << "processes must be divisible by 2." << std::endl;
                    exit(1);
                }
                N /= 2;
            }
            d = (d+1)%ND_;
        }
        return pd;
    }


    /*
     * This function assigns an index for each dimension to the
     * current MPI process. This is later used to decide which part
     * of the data should be read.
     */
    std::vector<int> indices_along_dimensions(const std::vector<int> &pd) {
        std::vector<int> dim_index(ND_,0);
        int r = mpi_rank_;
        int s = mpi_size_;
        for (int d=0; d<ND_; d++) {
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
    void set_up_ranges(const std::vector<int> &dimensions, const std::vector<bool> &reduced) {

        // define process distribution and get index for current rank
        std::vector<int> pd = process_distribution(reduced);
        std::vector<int> dim_index = indices_along_dimensions(pd);

        //dump_pd_info(pd, dim_index); // dump some debug info

        count_ = std::vector<int>(ND_);
        start_ = std::vector<int>(ND_);

        for (int d=0; d<ND_; d++) {
            count_[d] = dimensions[d] / pd[d];
            start_[d] = count_[d] * dim_index[d];

            // distribute remainder
            int remainder = dimensions[d] % pd[d];
            if (dim_index[d] < remainder) {
                count_[d] += 1;
                start_[d] += dim_index[d];
            } else {
                start_[d] += remainder;
            }
        }
    }


    /*
     * Here, we set up maps that will be used for reordering the
     * data. Each map contains, for each dimension, the distance
     * between two elements that are subsequent along this
     * dimension. One map is for the distances in the original data
     * array, one is for the distances along columns in the reordered
     * data, and the last is for the distances along rows in this
     * array. By building these maps, the actual reordering process
     * becomes quite easy.
     */
    void set_up_maps(const std::vector<int> &dimensions, const std::vector<bool> &reduced) {

        global_map_ = std::vector<int>(ND_);
        row_map_ = std::vector<int>(ND_);
        col_map_ = std::vector<int>(ND_);

        int global_interelement_distance = 1;
        int row_interelement_distance = 1;
        int col_interelement_distance = 1;
        global_rows = 1; // public member variable

        // go through dimensions, fastest varying first
        for (int d=0; d<ND_; d++) {
            global_map_[d] = global_interelement_distance;
            global_interelement_distance *= dimensions[d];
            if (reduced[d]) {
                //place along columns
                row_map_[d] = 0;
                col_map_[d] = col_interelement_distance;
                col_interelement_distance *= count_[d];
            } else {
                // place along rows
                row_map_[d] = row_interelement_distance;
                col_map_[d] = 0;
                row_interelement_distance *= count_[d];
                global_rows *= dimensions[d];
            }
        }

        // set matrix size
        N_global_ = global_interelement_distance;
        NR_ = row_interelement_distance;
        NC_ = col_interelement_distance;
        N_  = NR_*NC_;
        std::cout << "Matrix size for rank " << mpi_rank_ << ": " << NR_
                << "x" << NC_ << std::endl;
    }

    
    /*
     * This function selects the subset of the data assigned to the
     * current MPI process and reorders it according to the command
     * line options. It uses the information built up in the constructor.
     */
    Matrix<Scalar> load_matrix(const Scalar *data) {

        Matrix<Scalar> matrix(NR_, NC_);

        bool global_carry, local_carry;
        int global_index, row_index, col_index;

        // initialize global dimension indices
        std::vector<int> global_dim_index(ND_, 0);
        for (int d=0; d<ND_; d++) {
            global_dim_index[d] = start_[d];
        }

        // initialize local dimension indices
        std::vector<int> local_dim_index(ND_, 0);

        // loop over all values of the current mpi rank
        for (int i=0; i<N_; i++) {

            // set indices to zero, these are built in the next for loop
            global_index = 0;
            row_index = 0;
            col_index = 0;

            // set carries to true so the last dimension is incremented
            global_carry = true;
            local_carry = true;

            // loop over dimensions, fastest varying is the first
            for (int d=0; d<ND_; d++) {

                // add contribution of current dimension to indices
                global_index += global_dim_index[d] * global_map_[d];
                row_index += local_dim_index[d] * row_map_[d];
                col_index += local_dim_index[d] * col_map_[d];

                // increment global index for next iteration
                if (global_carry) {
                    global_dim_index[d]++;
                    if (global_dim_index[d] == start_[d] + count_[d]) {
                        global_dim_index[d] = start_[d];
                    } else {
                        global_carry = false;
                    }
                }

                // increment local index for next iteration
                if (local_carry) {
                    local_dim_index[d]++;
                    if (local_dim_index[d] == count_[d]) {
                        local_dim_index[d] = 0;
                    } else {
                        local_carry = false;
                    }
                }
            }

            // copy value to memory
            matrix(row_index, col_index) = data[global_index];
        }
        return matrix;
    }
};
