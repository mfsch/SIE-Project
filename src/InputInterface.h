#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <mpi.h>
#include "InputMap.h"
#include "matrix_definition.h"

template<typename InputType, typename Scalar> class InputInterface {

public:

    int global_rows; // number of rows


    InputInterface(std::vector<int> dimensions,
            std::vector<bool> reduced, const int fields) {

        // make sure both vectors are the same length
        if (dimensions.size() != reduced.size()) {
            if (!mpi_rank_) std::cerr << "ERROR: The number of "
                    << "dimensions for the argument '--dimensions' and "
                    << "'--reduced' must be the same." << std::endl;
            exit(1);
        }

        // add additional "dimension" for fields
        dimensions.push_back(fields);
        reduced.push_back(false);

        // prepare data for reading values
        set_up_ranges(dimensions, reduced);
        set_up_maps(dimensions, reduced);

        // dump some information for debugging
        //dump_info();
    }


    Matrix<Scalar> read(const std::vector<std::string> &filenames, bool multiple) {

        // initialize matrix
        Matrix<Scalar> matrix(NR_, NC_);
        matrix.setZero(); // TODO: remove this for better performance

        // unpack filenames and open first file
        std::vector< std::vector<std::string> > files
                = unpack_filenames(filenames, multiple);
        InputMap<InputType> current_data(files[0][0]);
        //bool reload_file = true;
        int part_index = 0;
        int field_index = 0;
        int file_offset = 0;

        // here, local corresponds to the current MPI process whereas
        // global corresponds to the combined data of all processes
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

            // switch to new file if necessary
            if (global_index >= file_offset + current_data.size()) {
                // TODO: add checks for file size

                // increment field/part index
                part_index++;
                if (part_index == files[field_index].size()) {
                    part_index = 0;
                    field_index++;
                }

                if (field_index == files.size()) {
                    if (!mpi_rank_) std::cerr << "ERROR: Trying to access more "
                        << "information than available in specified files." << std::endl;
                    exit(1);
                }

                file_offset += current_data.size();
                current_data = InputMap<InputType>(files[field_index][part_index]);
            }

            // copy value to memory
            //std::cout << "loading value " << global_index-file_offset << std::endl;
            //std::cout << "writing value (" << row_index << ", " << col_index << ")" << std::endl;
            matrix(row_index, col_index) = static_cast<Scalar>(current_data[global_index-file_offset]);
        }
        return matrix;
    }

private:

    // C++11 allows for dynamic initialization of class members
    const int mpi_size_ = MPI::COMM_WORLD.Get_size();
    const int mpi_rank_ = MPI::COMM_WORLD.Get_rank();

    // set up in set_up_ranges()
    int ND_;                  // number of dimensions
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
            // do not distribute reduced dimensions or dimension for fields
            if (!reduced[d] && d != ND_-1) {
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

        // initialize member variables
        ND_ = dimensions.size();
        count_ = std::vector<int>(ND_);
        start_ = std::vector<int>(ND_);

        // define process distribution and get index for current rank
        std::vector<int> pd = process_distribution(reduced);
        std::vector<int> dim_index = indices_along_dimensions(pd);

        //dump_pd_info(pd, dim_index); // dump some debug info

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


    std::vector< std::vector<std::string> > unpack_filenames(const std::vector<std::string> &filenames, const bool multiple) {

        // initialize return variable
        std::vector< std::vector<std::string> > files(filenames.size());

        //for (auto fn = filenames.begin(); fn != filenames.end(); ++fn) {
        for (int i=0; i<filenames.size(); i++) {

            if (multiple) { // unpack multiple file names

                // find path prefix
                std::string prefix = "";
                size_t pos = filenames[i].find_last_of("/\\");
                if (pos != std::string::npos) {
                    prefix = filenames[i].substr(0, pos+1);
                }

                // read filenames from file
                std::ifstream ifs(filenames[i]); // open file to read filenames
                if (ifs.is_open()) {
                    std::string file;
                    while (getline(ifs, file)) {
                        files[i].push_back(prefix + file);
                    }
                } else {
                    if (!mpi_rank_) std::cerr << "ERROR: Could not open file: "
                            << filenames[i] << std::endl;
                    exit(1);
                }
            } else {
                files[i].push_back(filenames[i]);
            }
        }
        return files;
    }


    Eigen::Map< const ColVector<InputType> > open_file(std::string &filename) {

        if (!mpi_rank_) std::cout << "Opening data input file: " << filename << std::endl;
        boost::iostreams::mapped_file_source file(filename);

        if (file.is_open()) {

            // get size of data
            int number_of_values = file.size() / sizeof(InputType);
            if (file.size() % sizeof(InputType)) {
                if (!mpi_rank_) std::cerr << "ERROR: Input file '" << filename
                        << "' does not seem to contain values of the correct type."
                        << std::endl;
                exit(1);
            }

            // create matrix and load data from file
            if (!mpi_rank_) std::cout << "File " << filename <<" opened successfully ("
                    << number_of_values << " values)." << std::endl;
            const InputType *data = reinterpret_cast<const InputType*>(file.data());
            Eigen::Map< const ColVector<InputType> > mapped_data(data, number_of_values);

            return mapped_data;

        } else {
            if (!mpi_rank_) std::cerr << "ERROR: Could not open file: "
                    << filename << std::endl;
            exit(1);
        }
    }
};
