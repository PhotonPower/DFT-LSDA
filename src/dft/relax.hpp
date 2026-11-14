// Structure optimization at fixed cell: BFGS on the Cartesian atomic coordinates.
#pragma once
#include <functional>

#include "dft/calculation.hpp"

namespace lsda {

struct RelaxParams {
  int max_steps = 50;
  double force_tol = 5e-4;  // Ha/bohr (~0.026 eV/A)
  double max_step = 0.3;    // bohr, largest displacement of an atom per step
  double hessian0 = 0.3;    // initial curvature (Ha/bohr^2)
  // called after every force evaluation (e.g. to write results and checkpoints)
  std::function<void(const Calculation&, const std::vector<Vec3>& forces, bool scf_converged)> on_step;
};

struct RelaxResult {
  bool converged = false;
  int steps = 0;
  double max_force = 0;
  std::vector<Vec3> forces;
};

RelaxResult relax(Calculation& calc, const RelaxParams& rp);

}  // namespace lsda
