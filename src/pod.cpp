// Author: Manuel Schmid
// License: see LICENSE file

#include <iostream> // cout
#include <boost/program_options.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include "Eigen/Dense"
#include "matrix_definition.h"
#include "InputInterface.h"

int main(int argc, char *argv[]) {

    // type for data input
    using Scalar = float;

    // program options
    std::string file_name;
    std::vector<int> dimensions;
    std::vector<bool> reduced;
    namespace po = boost::program_options;
    po::options_description po_desc("Program Options:");
    po_desc.add_options()
        ("help,h", "Show this help message.")
        ("file,f", po::value<std::string>(&file_name)->required(), "Input file.")
        ("dimensions,d", po::value< std::vector<int> >(&dimensions)->multitoken()->required(),
            "Length of dimensions, e.g. '256 256 65 200'")
        ("reduced,r", po::value< std::vector<bool> >(&reduced)->multitoken()->required(),
            "Which dimensions will be reduced in the POD, e.g. '1 0 0 1'")
        ;
    po::variables_map po_vm;
    po::store(po::command_line_parser(argc, argv).options(po_desc).run(), po_vm);

    if (po_vm.count("help")) {
        std::cout << po_desc << std::endl;
        return 0;
    }

    // po:notify tests for valid arguments, so do this after help
    po::notify(po_vm);

    // set up input file interface
    InputInterface<Scalar> input(dimensions, reduced);
    Matrix<Scalar> matrix = input.read(file_name);

    std::getchar(); // wait for ENTER

    return 0;
}