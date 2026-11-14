// Process-level parallelism (MPI) behind a small interface; without MPI all functions are trivial.
//
// Model: k-points are distributed over the processes (k-point ik belongs to process ik mod size). Every
// process holds the wavefunctions of its own k-points only; all other data (density, potentials, density
// matrices, eigenvalues) are replicated. Quantities summed over k-points are reduced with par::sum, and
// values that steer the control flow are taken from the root process (par::bcast) so that all processes
// take identical decisions. Threads (OpenMP) work inside each process.
#pragma once
#include "base/common.hpp"

namespace lsda::par {

void init(int* argc, char*** argv);
void finalize();
[[noreturn]] void abort(int code);

int rank();
int size();
inline bool root() { return rank() == 0; }
inline int owner(int ik) { return ik % size(); }
inline bool owns(int ik) { return owner(ik) == rank(); }

// element-wise reduction over all processes, result on every process (in place)
void sum(double* x, size_t n);
void sum(cplx* x, size_t n);
void sum(int* x, size_t n);
void max(double* x, size_t n);
// copy from the root process to all others
void bcast(double* x, size_t n);
void bcast(cplx* x, size_t n);
void bcast(int* x, size_t n);
// point-to-point transfer (blocking)
void send(const double* x, size_t n, int dest, int tag);
void recv(double* x, size_t n, int source, int tag);
void send(const cplx* x, size_t n, int dest, int tag);
void recv(cplx* x, size_t n, int source, int tag);

}  // namespace lsda::par
