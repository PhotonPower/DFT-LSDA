#include "base/parallel.hpp"

#include <cstdlib>

#ifdef LSDA_USE_MPI
#include <mpi.h>
#endif

namespace lsda {
void set_logging(bool enabled);  // common.cpp
}

namespace lsda::par {

#ifdef LSDA_USE_MPI

namespace {
int g_rank = 0, g_size = 1;
// MPI counts are ints: transfer long arrays in pieces
constexpr size_t chunk = size_t{1} << 28;

template <class F>
void pieces(size_t n, F&& f) {
  for (size_t off = 0; off < n; off += chunk) f(off, static_cast<int>(std::min(chunk, n - off)));
}
void check(int err, const char* what) {
  if (err != MPI_SUCCESS) fail(std::string("MPI error in ") + what);
}
}  // namespace

void init(int* argc, char*** argv) {
  int provided = 0;
  check(MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided), "MPI_Init_thread");
  check(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank), "MPI_Comm_rank");
  check(MPI_Comm_size(MPI_COMM_WORLD, &g_size), "MPI_Comm_size");
  set_logging(g_rank == 0);  // only the root process writes the log
}
void finalize() { MPI_Finalize(); }
void abort(int code) {
  MPI_Abort(MPI_COMM_WORLD, code);
  std::exit(code);
}
int rank() { return g_rank; }
int size() { return g_size; }

void sum(double* x, size_t n) {
  if (g_size == 1) return;
  pieces(n, [&](size_t off, int cnt) { check(MPI_Allreduce(MPI_IN_PLACE, x + off, cnt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), "MPI_Allreduce"); });
}
void sum(cplx* x, size_t n) { sum(reinterpret_cast<double*>(x), 2 * n); }
void sum(int* x, size_t n) {
  if (g_size == 1) return;
  pieces(n, [&](size_t off, int cnt) { check(MPI_Allreduce(MPI_IN_PLACE, x + off, cnt, MPI_INT, MPI_SUM, MPI_COMM_WORLD), "MPI_Allreduce"); });
}
void max(double* x, size_t n) {
  if (g_size == 1) return;
  pieces(n, [&](size_t off, int cnt) { check(MPI_Allreduce(MPI_IN_PLACE, x + off, cnt, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD), "MPI_Allreduce"); });
}
void bcast(double* x, size_t n) {
  if (g_size == 1) return;
  pieces(n, [&](size_t off, int cnt) { check(MPI_Bcast(x + off, cnt, MPI_DOUBLE, 0, MPI_COMM_WORLD), "MPI_Bcast"); });
}
void bcast(cplx* x, size_t n) { bcast(reinterpret_cast<double*>(x), 2 * n); }
void bcast(int* x, size_t n) {
  if (g_size == 1) return;
  pieces(n, [&](size_t off, int cnt) { check(MPI_Bcast(x + off, cnt, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast"); });
}
void send(const double* x, size_t n, int dest, int tag) {
  pieces(n, [&](size_t off, int cnt) { check(MPI_Send(x + off, cnt, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD), "MPI_Send"); });
}
void recv(double* x, size_t n, int source, int tag) {
  pieces(n, [&](size_t off, int cnt) {
    check(MPI_Recv(x + off, cnt, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "MPI_Recv");
  });
}
void send(const cplx* x, size_t n, int dest, int tag) { send(reinterpret_cast<const double*>(x), 2 * n, dest, tag); }
void recv(cplx* x, size_t n, int source, int tag) { recv(reinterpret_cast<double*>(x), 2 * n, source, tag); }

#else  // ---- serial build ---------------------------------------------------------------------------

void init(int*, char***) {}
void finalize() {}
void abort(int code) { std::exit(code); }
int rank() { return 0; }
int size() { return 1; }
void sum(double*, size_t) {}
void sum(cplx*, size_t) {}
void sum(int*, size_t) {}
void max(double*, size_t) {}
void bcast(double*, size_t) {}
void bcast(cplx*, size_t) {}
void bcast(int*, size_t) {}
void send(const double*, size_t, int, int) { fail("par::send called in a serial build"); }
void recv(double*, size_t, int, int) { fail("par::recv called in a serial build"); }
void send(const cplx*, size_t, int, int) { fail("par::send called in a serial build"); }
void recv(cplx*, size_t, int, int) { fail("par::recv called in a serial build"); }

#endif

}  // namespace lsda::par
