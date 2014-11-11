// Author: Manuel Schmid
// License: see LICENSE file

#include <iostream> // cout
#include <boost/program_options.hpp>
#include <mpi.h>
#include "matrix_definition.h"
#include "InputInterface.h"
#include "Decomposition.h"

int main(int argc, char *argv[]) {

    // type for data input
    using Scalar = float;

    // initialize mpi
    MPI_Init(&argc,&argv);
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

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
        if (!mpi_rank) std::cout << po_desc << std::endl;
        return 0;
    }

    // po:notify tests for valid arguments, so do this after help
    po::notify(po_vm);

    // set up input file interface
    InputInterface<Scalar> input(dimensions, reduced);
    Matrix<Scalar> matrix = input.read(file_name);

    // get N largest eigenvectors
    Decomposition<Scalar> pod(matrix, 5);

    // finalize mpi
    MPI_Finalize();

    // program has run successfully
    return 0;
}
