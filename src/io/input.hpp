// Input file reader. Format: "key = value" lines, '#' comments and blocks
//   begin lattice / species / atoms ... end
// See README.md for the list of keywords.
#pragma once
#include "crystal/crystal.hpp"
#include "dft/calculation.hpp"
#include "dft/relax.hpp"

namespace lsda {

struct Input {
  std::string task = "scf";  // scf | forces | relax
  std::string prefix = "lsda";
  bool restart = false;      // continue from <prefix>.chk (positions, density, density matrices)
  Crystal crystal;
  Params params;
  RelaxParams relax;
};

Input read_input(const std::string& path);

// machine-readable summary of a finished calculation (JSON)
void write_results(const std::string& path, const Calculation& calc, bool converged, const std::vector<Vec3>* forces);

}  // namespace lsda
