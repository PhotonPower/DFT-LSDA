/* Stand-in for <mpi.h> for machines without an MPI library; testing only, never for production runs.
 *
 *   cmake -DLSDA_MPI=ON -DLSDA_MPI_MOCK=ON ...
 *   LSDA_MOCK_NP=3 ./lsda input.in        (default: one process)
 *
 * It provides the calls used by src/base/parallel.cpp with the prototypes of the MPI standard. With
 * LSDA_MOCK_NP > 1 the program forks into that many processes at MPI_Init_thread; every process other than
 * the root is connected to the root by a socket pair. Because all processes execute the same sequence of
 * collective operations, a plain byte stream per connection is sufficient:
 *   MPI_Allreduce : each process sends its buffer to the root, the root combines and returns the result
 *   MPI_Bcast     : the root sends, the others receive (root must be process 0)
 *   MPI_Send/Recv : only from a non-root process to the root
 * This exercises the distribution of k-points over processes for real, but it is not an MPI implementation. */
#ifndef LSDA_MOCK_MPI_H
#define LSDA_MOCK_MPI_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;
typedef struct { int MPI_SOURCE, MPI_TAG, MPI_ERROR; } MPI_Status;
#define MPI_COMM_WORLD 0
#define MPI_SUCCESS 0
#define MPI_THREAD_FUNNELED 1
#define MPI_DOUBLE 8 /* the datatype handles are the element sizes */
#define MPI_INT 4
#define MPI_SUM 1
#define MPI_MAX 2
#define MPI_IN_PLACE ((void*)1)
#define MPI_STATUS_IGNORE ((MPI_Status*)0)
#define MOCK_MAX_NP 16

static int mock_rank = 0, mock_size = 1;
static int mock_fd[MOCK_MAX_NP];    /* root: connection to process r; other processes: mock_fd[0] leads to the root */
static pid_t mock_pid[MOCK_MAX_NP];

static inline void mock_write(int fd, const void* buf, size_t n) {
  const char* p = (const char*)buf;
  while (n > 0) {
    const ssize_t k = write(fd, p, n);
    if (k <= 0) _exit(98); /* peer has gone */
    p += k;
    n -= (size_t)k;
  }
}
static inline void mock_read(int fd, void* buf, size_t n) {
  char* p = (char*)buf;
  while (n > 0) {
    const ssize_t k = read(fd, p, n);
    if (k <= 0) _exit(98);
    p += k;
    n -= (size_t)k;
  }
}

static inline int MPI_Init_thread(int* argc, char*** argv, int required, int* provided) {
  (void)argc; (void)argv;
  *provided = required;
  const char* np = getenv("LSDA_MOCK_NP");
  int n = np ? atoi(np) : 1;
  if (n < 1) n = 1;
  if (n > MOCK_MAX_NP) n = MOCK_MAX_NP;
  mock_size = n;
  fflush(stdout);
  for (int r = 1; r < n; ++r) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) abort();
    const pid_t pid = fork();
    if (pid < 0) abort();
    if (pid == 0) { /* child = process r */
      mock_rank = r;
      for (int q = 1; q < r; ++q) close(mock_fd[q]);
      close(sv[0]);
      mock_fd[0] = sv[1];
      break;
    }
    close(sv[1]);
    mock_fd[r] = sv[0];
    mock_pid[r] = pid;
  }
  return MPI_SUCCESS;
}
static inline int MPI_Finalize(void) {
  if (mock_rank == 0)
    for (int r = 1; r < mock_size; ++r) {
      int status = 0;
      waitpid(mock_pid[r], &status, 0);
    }
  return MPI_SUCCESS;
}
static inline int MPI_Comm_rank(MPI_Comm comm, int* rank) { (void)comm; *rank = mock_rank; return MPI_SUCCESS; }
static inline int MPI_Comm_size(MPI_Comm comm, int* size) { (void)comm; *size = mock_size; return MPI_SUCCESS; }
static inline int MPI_Abort(MPI_Comm comm, int errorcode) { (void)comm; _exit(errorcode); }

static inline int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  (void)comm;
  const size_t bytes = (size_t)count * (size_t)datatype;
  const void* src = sendbuf == MPI_IN_PLACE ? recvbuf : sendbuf;
  if (mock_size == 1) {
    if (src != recvbuf) memcpy(recvbuf, src, bytes);
    return MPI_SUCCESS;
  }
  if (mock_rank != 0) {
    mock_write(mock_fd[0], src, bytes);
    mock_read(mock_fd[0], recvbuf, bytes);
    return MPI_SUCCESS;
  }
  void* acc = malloc(bytes ? bytes : 1);
  void* tmp = malloc(bytes ? bytes : 1);
  memcpy(acc, src, bytes);
  for (int r = 1; r < mock_size; ++r) {
    mock_read(mock_fd[r], tmp, bytes);
    for (int i = 0; i < count; ++i) {
      if (datatype == MPI_DOUBLE) {
        double* a = (double*)acc + i;
        const double b = ((double*)tmp)[i];
        *a = op == MPI_SUM ? *a + b : (b > *a ? b : *a);
      } else {
        int* a = (int*)acc + i;
        const int b = ((int*)tmp)[i];
        *a = op == MPI_SUM ? *a + b : (b > *a ? b : *a);
      }
    }
  }
  memcpy(recvbuf, acc, bytes);
  for (int r = 1; r < mock_size; ++r) mock_write(mock_fd[r], acc, bytes);
  free(acc);
  free(tmp);
  return MPI_SUCCESS;
}
static inline int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
  (void)comm;
  if (root != 0) abort();
  const size_t bytes = (size_t)count * (size_t)datatype;
  if (mock_rank == 0)
    for (int r = 1; r < mock_size; ++r) mock_write(mock_fd[r], buffer, bytes);
  else
    mock_read(mock_fd[0], buffer, bytes);
  return MPI_SUCCESS;
}
static inline int MPI_Send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
  (void)tag; (void)comm;
  if (dest != 0 || mock_rank == 0) abort(); /* only process -> root is supported */
  mock_write(mock_fd[0], buf, (size_t)count * (size_t)datatype);
  return MPI_SUCCESS;
}
static inline int MPI_Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status* status) {
  (void)tag; (void)comm; (void)status;
  if (mock_rank != 0 || source <= 0 || source >= mock_size) abort();
  mock_read(mock_fd[source], buf, (size_t)count * (size_t)datatype);
  return MPI_SUCCESS;
}
#endif
