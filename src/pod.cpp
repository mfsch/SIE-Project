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
    using InputType = float;
    using Scalar = double;

    // initialize mpi
    MPI::Init(argc, argv);

    // program options
    std::vector<std::string> filenames;
    bool multiple;
    bool reduce_variables;
    std::vector<int> dimensions;
    std::vector<bool> reduced;
    std::string output_prefix;
    namespace po = boost::program_options;
    po::options_description po_desc("Program Options:");
    po_desc.add_options()
        ("help,h", "Show this help message.")
        ("files,f", po::value< std::vector<std::string> >(&filenames)->multitoken()->required(), "Input files, one file per variable.")
        ("reduce-variables", po::value<bool>(&reduce_variables), "Treat the different variables as reduced dimensions.")
        ("multiple,m", po::value<bool>(&multiple), "Use multiple files per variable. When this is set, the input files have to be text files with one filename per line. The files will then be concatenated along the last dimension.")
        ("dimensions,d", po::value< std::vector<int> >(&dimensions)->multitoken()->required(),
            "Length of dimensions, e.g. '256 256 65 200'")
        ("reduced,r", po::value< std::vector<bool> >(&reduced)->multitoken()->required(),
            "Which dimensions will be reduced in the POD, e.g. '1 0 0 1'")
        ("output-prefix,o", po::value<std::string>(&output_prefix)->required(), "Input files, one file per variable.")
        ;
    po::variables_map po_vm;
    po::store(po::command_line_parser(argc, argv).options(po_desc).run(), po_vm);

    // only first mpi process prints help, then all exit
    if (po_vm.count("help")) {
        if (!MPI::COMM_WORLD.Get_rank()) std::cout << po_desc << std::endl;
        return 0;
    }

    // po:notify tests for valid arguments, so do this after help
    po::notify(po_vm);

    // set up input file interface
    InputInterface<InputType, Scalar> input(dimensions, reduced, filenames.size());
    MPI::COMM_WORLD.Barrier(); // for aesthetic output only
    Matrix<Scalar> matrix = input.read(filenames, multiple);

    // find M largest eigenvectors
    int M = 5;
    Decomposition<Scalar> pod(matrix, M, input.global_rows);

    // print eigenvalues
    if (!MPI::COMM_WORLD.Get_rank()) {
        std::cout << "Eigenvalues:\t";
        ColVector<Scalar> ev = pod.eigenvalues();
        for (int i=0; i<M; i++) {
            std::cout << ev[i] << "\t";
        }
        std::cout << std::endl;
    }

    // project data
    for (int i=0; i<M; i++) {
        ColVector<Scalar> v = pod.projection(matrix, i);
        std::string output_filename = output_prefix + "_" + std::to_string(i+1) + ".out";
        input.write_col(v, output_filename);
    }

    // finalize mpi
    MPI::Finalize();
    return 0;
}
