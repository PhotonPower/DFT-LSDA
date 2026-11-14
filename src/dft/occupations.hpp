// Occupation numbers for metals: smearing functions and Fermi-level search.
#pragma once
#include <optional>

#include "base/common.hpp"

namespace lsda {

enum class Smearing { FermiDirac, Gaussian, MethfesselPaxton1, Cold };
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

// eig[ik * nspin + s][band]; kweight sums to 1. If fixed_moment is set (nspin = 2), the two spin
// channels get separate Fermi levels such that N_up - N_dn = fixed_moment.
Occupations compute_occupations(const std::vector<std::vector<double>>& eig, const std::vector<double>& kweight,
                                int nspin, double nelec, Smearing type, double width,
                                std::optional<double> fixed_moment);

}  // namespace lsda
