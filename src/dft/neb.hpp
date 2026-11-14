// Minimum-energy paths and migration barriers: nudged elastic band method with improved tangents
// (Henkelman, Jonsson, J. Chem. Phys. 113, 9978 (2000)) and climbing image (ibid. 113, 9901),
// relaxed with the FIRE algorithm (Bitzek et al., PRL 97, 170201 (2006)).
#pragma once
#include <functional>

#include "dft/calculation.hpp"

namespace lsda {

struct NebParams {
  int n_images = 5;         // intermediate images (the two end points are fixed)
  double spring = 0.1;      // spring constant (Ha/bohr^2)
  bool climb = true;        // climbing image: the highest image converges to the saddle point
  int max_steps = 200;
  double force_tol = 1e-3;  // Ha/bohr, largest NEB force on an atom
  double max_step = 0.2;    // bohr, largest displacement of an atom per step
  std::string prefix = "lsda";
  bool restart = false;     // continue from <prefix>.neb.pos and the image checkpoints
};

struct NebResult {
  bool converged = false, stopped = false;
  int steps = 0, saddle = 0;                 // index of the highest image
  double max_force = 0;
  std::vector<double> energy;                // free energies of all images including the end points
  std::vector<double> distance;              // cumulative path length (bohr)
  std::vector<std::vector<Vec3>> frac;       // fractional coordinates of all images
};

// Generic band optimizer. R holds the Cartesian positions of all images (first and last are fixed and their
// energies must be set in `energy`). evaluate(i, R_i, E_i, F_i) computes energy and forces of image i and
// returns false to interrupt the optimization (result.stopped). on_step is called after every evaluation of
// the whole band, before the images are moved.
using BandCoords = std::vector<Vec3>;
NebResult optimize_band(std::vector<BandCoords>& R, std::vector<double>& energy,
                        const std::function<bool(int, const BandCoords&, double&, BandCoords&)>& evaluate,
                        const NebParams& np, const std::function<void(int step)>& on_step = nullptr, bool verbose = true,
                        double energy_unit = hartree_in_ev);

// `initial` holds the first end point; frac_final the last one (same atom order; coordinates are not
// wrapped, so that the path is defined by the given values).
NebResult run_neb(const Crystal& initial, const std::vector<Vec3>& frac_final, const Params& params, const NebParams& np);

void write_neb_results(const std::string& path, const Crystal& c, const NebResult& r);

}  // namespace lsda
