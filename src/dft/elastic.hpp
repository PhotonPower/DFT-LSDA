// Elastic constants from the stress response to small homogeneous strains:
//   C_ij = d sigma_i / d eps_j   (Voigt notation: 1..6 = xx, yy, zz, yz, xz, xy; engineering shear strains).
// Atomic positions are relaxed in every strained cell (relaxed-ion constants). Strain patterns that are
// related by a symmetry operation of the crystal are computed only once.
#pragma once
#include "dft/calculation.hpp"
#include "dft/relax.hpp"

namespace lsda {

struct ElasticResult {
  bool stopped = false;
  int n_strained = 0;       // strained calculations carried out
  double asymmetry = 0;     // largest |C_ij - C_ji| before symmetrization (Ha/bohr^3)
  Mat3 stress0{};           // stress of the unstrained cell
  RMatrix C;                // 6 x 6, Ha/bohr^3
};

ElasticResult run_elastic(const Crystal& crystal, const Params& params, const RelaxParams& rp, double delta);
void write_elastic_results(const std::string& path, const Crystal& c, const ElasticResult& r);

}  // namespace lsda
