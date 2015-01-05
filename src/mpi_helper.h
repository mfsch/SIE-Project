#pragma once

// MPI type helper
template<typename T> struct mpi_helper {};
template<> struct mpi_helper<float>  { const MPI::Datatype type = MPI::FLOAT; };
template<> struct mpi_helper<double> { const MPI::Datatype type = MPI::DOUBLE; };
