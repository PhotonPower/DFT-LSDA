// Occupation numbers for metals: smearing functions and Fermi-level search.
#pragma once
#include <optional>

#include "base/common.hpp"

namespace lsda {

enum class Smearing { FermiDirac, Gaussian, MethfesselPaxton1, Cold, Tetrahedron };
Smearing parse_smearing(const std::string& name);

// occupation f(x) in [0,1] and the entropy-like function s(x), x = (mu - eps)/width;
// the contribution of a state to -TS is  width * s(x) * (spin degeneracy) * (k weight)
double smearing_occupation(Smearing type, double x);
double smearing_entropy(Smearing type, double x);

struct Occupations {
  std::vector<std::vector<double>> f;  // [ik * nspin + s][band], including the spin degeneracy for nspin = 1
  double efermi[2] = {0, 0};           // per spin (identical unless the moment is fixed)
  double entropy_term = 0;             // -TS (Ha)
  double band_energy = 0;              // sum of f * w * eps
  double n_spin[2] = {0, 0};           // electrons per spin channel
};

// Tetrahedron method with Bloechl's correction (Bloechl, Jepsen, Andersen, PRB 49, 16223 (1994)).
struct TetraGrid {
  int n[3] = {0, 0, 0};
  std::vector<int> map;  // full-grid point (i0 * n1 + i1) * n2 + i2 -> irreducible k-point
  Mat3 B{};              // reciprocal lattice (selects the shortest diagonal of the microcells)
};
// Integration weights of the four corners of one tetrahedron (corner energies e, any order) for the
// Fermi level ef; vt = volume fraction of the tetrahedron. A filled tetrahedron gives w = vt/4 each.
void tetrahedron_weights(const double e[4], double ef, double vt, bool bloechl, double w[4]);

// eig[ik * nspin + s][band]; kweight sums to 1. For Smearing::Tetrahedron `tetra` is required. If fixed_moment is set (nspin = 2), the two spin
// channels get separate Fermi levels such that N_up - N_dn = fixed_moment.
Occupations compute_occupations(const std::vector<std::vector<double>>& eig, const std::vector<double>& kweight,
                                int nspin, double nelec, Smearing type, double width,
                                std::optional<double> fixed_moment, const TetraGrid* tetra = nullptr);

}  // namespace lsda
