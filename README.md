Flexible, parallel POD for large data sets
==========================================

*This project has been started as part of an SIE project at [EPFL](http://www.epfl.ch) with the title ‘POD of Wall-Bounded Turbulent Flows’ in the fall semester of the academic year 2014/15.*


Setup
-----

    git clone https://github.com/mfsch/SIE-Project.git
    cd SIE-Project

    mkdir build && cd build
    cmake ..
    make

### Dependencies

Apart from the C++ standard library, the code only depends on two more libraries. [Eigen](http://eigen.tuxfamily.org/) is used for all linear algebra operations. To avoid dependency issues, it is included in the ‘eigen/’ subfolder of this repository. [Boost](http://www.boost.org/) is used for the memory-mapped files. This is expected to be installed on the system and CMake will attempt to find.

The tests are based on the [Google C++ Testing Framework](https://code.google.com/p/googletest/). This is automatically cloned from the SVN repository by CMake.


Usage
-----

    Program Options::
      -h [ --help ]              Show this help message.
      -M [ --modes ] arg         Number of eigenvalues and -vectors that are 
                                 calculated (default: 1).
      -i [ --input-files ] arg   Input files, one file per variable.
      -o [ --output-prefix ] arg Prefix for output files.
      -d [ --dimensions ] arg    Length of dimensions, e.g. '256 256 65 200'
      -r [ --reduced ] arg       Which dimensions will be reduced in the POD, e.g. 
                                 '1 0 0 1'
      -m [ --multiple ]          Use multiple files per variable. When this is set,
                                 the input files have to be text files with one 
                                 filename per line. The files will then be 
                                 concatenated along the last dimension.
      --reduce-variables         Treat the different variables as reduced 
                                 dimensions.

### Input Files

The input files are raw binaries. No precautions are taken with respect to endianness. By default, the data type is expected to be `float`, but this can be changed at the beginning of the file `pod.cpp`. The dimensions can be in any order, as the data is reordered according to the `--reduced` argument.

The argument `--input-files` is followed by a list of files, each specifying the data for one variable (u, v, T, ...). If more than one data file is needed for each variable, a text file with one file name per line can be passed instead. In this case, the `--multiple` flag needs to be set.

### Output Files

The program writes one file for each of the M eigenvectors that have been calculated. In each file, it writes the projection of the data onto this eigenvector. These files therefore have all dimensions that haven’t been reduced. If several variables are analyzed together, the files contain the data of all variables, as if ‘variables’ had been added as a new slowest-changing dimension.

### Specifying dimensions

The argument `--dimensions` is a list of integers with the number of entries in each dimension, from the fastest to the slowest changing dimension. If multiple files per variable are used, i.e. if the `--multiple` flag is set, the files are concatenated along the last, slowest changing dimension. In this case, the total number of entries has to be specified. The program will automatically switch to the next file once one file has been read.

### Reduced dimensions

The argument `--reduced` is a list of boolean values specifying which dimensions should be treated as ‘reduced’ dimensions. These are the dimensions that are placed along the rows of the data matrix X. In the context of PCA, this corresponds to the *variables* as opposed to the *observations*. The eigenvectors will have these dimensions and the projections on the eigenvectors will have all other dimensions.

With the flag `--reduce-variables`, the data of different variables will be concatenated along rows, i.e. the variables are treated as ‘reduced’ dimension.


Implementation details
----------------------

The code has two main parts. The first part (mainly the class InputInterface) is responsible for reading the data from multiple files and reorganizing it into a 2D matrix. The second part (mainly the class Decomposition) does a partial eigendecomposition of the covariance matrix, i.e. it calculates the M largest eigenvalues and their eigenvectors.

### Data reorganization

The data to be analyzed by POD is typically multi-dimensional. A 3D simulation will produce four-dimensional (x, y, z, and t) output files. In addition, we can combine different variables (e.g. u, v, T, p...) and analyze them together. In this case, the variables are like an additional dimension of the data.

POD is based on two-dimensional data. We can think of them as *x* and *t* and they correspond to the rows and columns of the data matrix. In the context of PCA, we can think of them as *variables* and *observations*, where each observation contains a value for all of the variables. Each of the original dimensions can be treated as part of *x* and *t*, where *x* is placed along the columns and *t* is placed along the rows of the data matrix. The eigenvectors then have the length of *t* whereas the projections have the length of *x*. In the code, the dimensions treated as *t* are called *reduced* dimensions.

In addition, the data has to be split up between the different MPI processes. This is done along columns, i.e. each MPI rank gets some of the rows of the data matrix. This has the advantage that in order to calculate X'\*X\*v, each rank can do this computation locally and the result is simply summed up with an MPI Allreduce. The way the data is currently partitioned, it is split along all dimensions that are not *reduced*, execpt for the variables, which are never split. This might not actually be necessary and splitting along just one of these dimensions might be enough. Such a change might simplify the code a bit, so it might be worth giving it a try.

#### Selection of data range

Each MPI rank has to decide which part of the data is read. As mentioned above, this could probably also be done in a simpler way, by only splitting the data along one dimension. However, the current implementation splits the data along all dimensions that are not *reduced*.

In a first step, each dimension is assigned a number of processes, i.e. the number of times it is split. The product of these numbers has to be equal to the total number of MPI processes. The code assumes the total number of MPI processes to be a power of two, and each dimension is assigned a number of processes that is a power of two. First, all *reduced* dimensions are assigned the number 1. The other dimensions start with 1 as well, then they get in turn multiplied by 2 while the number of remaining processes is divided by two. This stops when the remaining processes are down to 1.

In a second step, each process is assigned assigned an index along each of the dimensions, starting with \[0 0 ... 0\], \[1 0 .. 0\], and so on. This basically corresponds to a conversion of the MPI rank to a number with a different basis for each digit. This number is then used to select the range of entries that are read along each dimension.

#### Reordering the data

The data is reordered with the help of several *maps*. These are short lists of integers that represent, for each dimension, the distance between two consecutive elements along this dimension. One such map is created for the input data. In this case, the number simply represents the product of the length of all faster changing dimensions in the input array. The second map is for the rows of the restructured data. There, the number is the product of the length of all faster changing *reduced* dimensions. The dimensions that aren’t reduced get an entry of 0. The third map is for the columns of the restructured data. There, the number is the product of the length of all faster changing dimensions that are not reduced. The length here is not the total length but rather the number of entries that have been assigned to the current MPI process.

With these maps, the actual reordering of the data becomes quite simple. All we need to do is keep an index along each dimension. We can then find the corresponding index of the input array as well as the row and column of the restructured matrix by multiplying the index along each dimension with the corresponding map and calculating their sum.


### Eigendecomposition
The M largest eigenvalues and their eigenvectors are calculated using the Lanczos method. This is an iterative method converging to the largest eigenvalues first. This has the advantage that it can be stopped as soon as the first M eigenvalues are approximated to an acceptable tolerance. In addition, the Lanczos method never requires the actual matrix A but only its application A\*x. This way, the matrix X'X used for POD never has to be computed explicitly.

*to be completed*


To-do list
----------

* Use exceptions for errors.
* Add more information to README.


License
-------
This project is distributed under the MIT license. See the accompanying [license file](LICENSE) or [this online copy](http://opensource.org/licenses/MIT) for details.

The Eigen library that is contained in this repository for convenience is primarily MPL2 licensed. Refer to [the Eigen license notice](eigen/COPYING.README) for more details.
