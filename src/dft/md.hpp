// Born-Oppenheimer molecular dynamics: velocity-Verlet integration of the ionic motion on the
// self-consistent energy surface (NVE), optionally with a Berendsen thermostat.
#pragma once
#include "dft/calculation.hpp"

namespace lsda {

struct MdParams {
  int steps = 100;
  double dt_fs = 2.0;            // time step (fs)
  double temperature = 300.0;    // initial temperature (K), also the target of the thermostat
  std::string thermostat = "none";  // none | berendsen
  double tau_fs = 100.0;         // relaxation time of the thermostat (fs)
  unsigned seed = 4711;
  std::string prefix = "lsda";
  bool restart = false;          // continue from <prefix>.md.restart
};

struct MdResult {
  bool stopped = false;
  int steps = 0;
  double drift = 0;              // largest deviation of the conserved energy from its first value (Ha)
  double mean_temperature = 0;
};

// masses in amu (one per atom). The calculation must have been set up without symmetry.
MdResult run_md(Calculation& calc, const std::vector<double>& masses, const MdParams& mp);

}  // namespace lsda
