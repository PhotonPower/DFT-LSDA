// Structure optimization at fixed cell: BFGS on the Cartesian atomic coordinates.
#pragma once
#include <functional>
#include <memory>

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
  bool stopped = false;  // interrupted by the wall-time limit
  int steps = 0;
  double max_force = 0;
  std::vector<Vec3> forces;
};

RelaxResult relax(Calculation& calc, const RelaxParams& rp);

// Variable-cell optimization: lattice vectors and atomic positions are relaxed until the stress equals
// -pressure * 1 within stress_tol. The plane-wave basis is rebuilt for every cell at constant cutoff;
// with a smoothed cutoff (ecut_smooth) the energy is a differentiable function of the cell and the stress
// its exact derivative. Wavefunctions and density are carried over between the cells.
struct CellRelaxParams {
  int max_steps = 30;
  double stress_tol = 0.05 / 29421.02648438959;  // Ha/bohr^3 (0.05 GPa)
  double pressure = 0.0;                          // target pressure (Ha/bohr^3)
  double modulus = 300.0 / 29421.02648438959;     // elastic modulus for the initial Hessian (Ha/bohr^3)
  double max_strain = 0.03;                       // largest strain component per step
  std::function<void(const Calculation&, const std::vector<Vec3>& forces, const Mat3& stress)> on_step;
};

struct CellRelaxResult {
  bool converged = false;
  bool stopped = false;
  int steps = 0;
  Mat3 stress{};
  std::vector<Vec3> forces;
};

CellRelaxResult relax_cell(std::unique_ptr<Calculation>& calc, const RelaxParams& rp, const CellRelaxParams& cp);

}  // namespace lsda
