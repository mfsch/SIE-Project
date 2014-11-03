#pragma once
#include <iostream>
//#include "Eigen/Dense"

template<class Scalar>
class InputInterface {

public:
    InputInterface(std::vector<int> dimensions, std::vector<bool> reduced) {

        // make sure both vectors are the same length
        if (dimensions.size() != reduced.size()) {
            std::cerr << "ERROR: The number of dimensions for the argument '--dimensions' and '--reduced' must be the same." << std::endl;
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

private:
    int ND_; // number of dimensions
    int NR_; // number of rows
    int NC_; // number of columns
    std::vector<int>  dims_;    // length of dimensions
    std::vector<bool> reduced_; // whether dimensions are reduced (along columns)
    std::vector<int>  row_map_; // interelement distances along rows
    std::vector<int>  col_map_; // interelement distances along column

    /*
     * Convenience function for development, no real use.
     */
    void dump_maps() {
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
        std::cout << "Matrix size: " << NR_ << "x" << NC_ << std::endl;
    }


};
