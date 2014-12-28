#pragma once
#include <algorithm> // swap
#include <iostream>
#include <fstream> // ifstream
#include <boost/iostreams/device/mapped_file.hpp>
#include <Eigen/Dense>
#include <mpi.h>

template<typename InputType> class InputMap {

private:
    int mpi_rank_;
    boost::iostreams::mapped_file_source file_;
    const InputType* data_pointer_;
    int size_;

public:
    InputMap(const std::string &filename) {

        mpi_rank_ = MPI::COMM_WORLD.Get_rank();

#if DEBUG
        if (!mpi_rank_) std::cout << "Opening data input file: " << filename << std::endl;
#endif

        file_ = boost::iostreams::mapped_file_source(filename);

        if (file_.is_open()) {

            // get size of data
            size_ = file_.size() / sizeof(InputType);
            if (file_.size() % sizeof(InputType)) {
                std::cerr << "ERROR: Input file '" << filename
                        << "' does not seem to contain values of the correct type."
                        << std::endl;
                exit(1);
            }

            // create matrix and load data from file
#if DEBUG
            if (!mpi_rank_) std::cout << "File " << filename <<" opened successfully ("
                    << size_ << " values)." << std::endl;
#endif
            data_pointer_ = reinterpret_cast<const InputType*>(file_.data());
            //data_ = Map<InputType>(data_pointer, number_of_values);

        } else {
            std::cerr << "ERROR: Could not open file: " << filename << std::endl;
            exit(1);
        }
    }

    ~InputMap() {
        file_.close();
    }

    InputMap(const InputMap& other)
        : mpi_rank_(other.mpi_rank_),
          file_(other.file_),
          data_pointer_(other.data_pointer_),
          size_(other.size_) {
    }

    InputMap& operator=(InputMap other) {
        std::swap(this->file_, other.file_);
        std::swap(this->data_pointer_, other.data_pointer_);
        std::swap(this->size_, other.size_);
        return *this;
    }

    const InputType& operator[](const int i) {
        if (i >= size_) {
            std::cerr << "ERROR: Trying to access index " << i
                    << " when there are only " << size_ << " values." << std::endl;
            exit(1);
        }
        return data_pointer_[i];
    }

    int size() {
        return size_;
    }
};
