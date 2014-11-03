#include <boost/program_options.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include "Eigen/Dense"
#include <iostream> // cout

int main(int argc, char *argv[]) {

    // type for data input
    using scalar = float;

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

    // open memory mapped file
    std::cout << "Input File: " << file_name << std::endl;
    //boost::iostreams::mapped_file_params file_params(file_name);
    boost::iostreams::mapped_file_source file(file_name);
    size_t n_entries = 256*256*65*200;
    size_t n_bytes = n_entries * sizeof(scalar);
    //file.open(file_name);

    if (file.is_open()) {
        const scalar *data = reinterpret_cast<const scalar*>(file.data());
        std::cout << "file size: " << file.size() << ", should be " << n_bytes << std::endl;
        std::cout << "first values:" << std::endl;
        for (int i=200000; i<200010; i++) {
            std::cout << data[i] << std::endl;
        }

        std::getchar();
        file.close();
    } else {
        std::cerr << "ERROR: Could not open file: " << file_name << std::endl;
        exit(1);
    }

    return 0;
}
