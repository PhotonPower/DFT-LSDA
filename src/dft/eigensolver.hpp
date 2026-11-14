// Iterative eigensolver for the generalized problem H x = eps S x.
// All-band LOBPCG (locally optimal block preconditioned conjugate gradient) with soft locking,
// kinetic (Teter-Payne-Allan) preconditioning and an S-orthonormal search basis [X, W, P].
#pragma once
#include "dft/hamiltonian.hpp"

namespace lsda {

struct EigensolverResult {
  int iterations = 0;
  double max_residual = 0;  // over the required bands
  int not_converged = 0;
};

// X (npw x nb) holds the start vectors on entry and the Ritz vectors on exit; eps the Ritz values.
// tol[n] is the residual-norm target ||H x - eps S x|| of band n; bands are iterated until they reach it.
// Only the lowest n_required bands enter the convergence statistics of the result.
// If PX is given it receives the projections beta^H X of the final vectors.
EigensolverResult lobpcg(const Hamiltonian& h, CMatrix& X, std::vector<double>& eps, const std::vector<double>& tol,
                         int n_required, int max_iter, CMatrix* PX = nullptr);

// Projected preconditioned conjugate gradient (Vecharynski, Yang, Pask, J. Comput. Phys. 290, 73 (2015)):
// Rayleigh-Ritz steps in span{x_j, w_j, p_j} for sub-blocks of `block_size` neighbouring bands,
// S-orthonormalization by Cholesky QR, and a full Rayleigh-Ritz step only every `rr_period` iterations
// and at the end. All matrix elements are evaluated with the PAW projections (no S x or nonlocal H x
// vectors), which keeps the dense linear algebra small. block_size = 1 is the cheapest variant,
// block_size >= number of bands is equivalent to LOBPCG.
// If px_valid is set, *PX holds the projections beta^H X of the start vectors (saves one projection).
// The convergence statistics refer to the residuals before the last update (an upper bound in practice).
EigensolverResult ppcg(const Hamiltonian& h, CMatrix& X, std::vector<double>& eps, const std::vector<double>& tol,
                       int n_required, int max_iter, CMatrix* PX = nullptr, int block_size = 8, int rr_period = 5,
                       bool px_valid = false);

// Rayleigh-Ritz in the span of the m >= nkeep columns of X (which may be linearly dependent);
// on exit X holds the nkeep lowest Ritz vectors (S-orthonormal). Used to turn atomic orbitals into
// start wavefunctions. Returns false if the span has fewer than nkeep independent directions.
bool subspace_start(const Hamiltonian& h, CMatrix& X, int nkeep);

}  // namespace lsda
