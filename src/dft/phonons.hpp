// Harmonic phonons at the Gamma point of the given (super)cell from finite displacements.
// Only symmetry-inequivalent atoms are displaced; the remaining force constants follow from
//   Phi(g(b), g(a)) = R Phi(b, a) R^T   for every symmetry operation g = (R, t).
// The frequencies give the vibrational free energy of a supercell, e.g. for defect formation entropies.
#pragma once
#include "dft/calculation.hpp"

namespace lsda {

struct PhononParams {
  double displacement = 0.02;  // bohr; central differences +-h
  std::string prefix = "lsda";
  bool restart = false;        // reuse displacements already stored in <prefix>.phonons.forces
};

struct PhononResult {
  bool stopped = false;                 // interrupted by the wall-time limit
  int n_displacements = 0;              // force calculations (without the reference structure)
  double asr_violation = 0;             // largest |sum_b Phi(b, a)| before the acoustic sum rule is imposed (Ha/bohr^2)
  double asymmetry = 0;                 // largest |Phi - Phi^T| before symmetrization
  double max_force_reference = 0;       // residual force of the undisplaced structure
  std::vector<double> frequency_thz;    // 3N modes, ascending; imaginary modes as negative numbers
  RMatrix force_constants;              // 3N x 3N, Ha/bohr^2
};

// standard atomic weight (amu) of element Z; fails for unknown elements
double atomic_mass(int Z);

// masses: amu per atom
PhononResult run_phonons(const Crystal& crystal, const Params& params, const PhononParams& pp, const std::vector<double>& masses);
void write_phonon_results(const std::string& path, const Crystal& c, const std::vector<double>& masses, const PhononResult& r);

}  // namespace lsda
