// Input file reader. Format: "key = value" lines, '#' comments and blocks
//   begin lattice / species / atoms ... end
// See README.md for the list of keywords.
#pragma once
#include "crystal/crystal.hpp"
#include "dft/calculation.hpp"
#include "dft/elastic.hpp"
#include "dft/md.hpp"
#include "dft/neb.hpp"
#include "dft/phonons.hpp"
#include "dft/relax.hpp"

namespace lsda {

struct Input {
  std::string task = "scf";  // scf | forces | stress | relax | cellrelax | neb | phonons | elastic | bands | md
  bool stress = false;       // compute the stress tensor after the final SCF (implied by task = stress)
  std::string prefix = "lsda";
  bool write_density = false;  // <prefix>.rho.cube (and .mag.cube) after the final SCF
  // band structure (task = bands): path through the listed points, kpath_points steps per segment
  std::vector<std::pair<std::string, Vec3>> kpath;
  int kpath_points = 20;
  bool dos = false;          // write <prefix>.dos after the final SCF
  double dos_width = 0.0;    // Gaussian width (Ha); 0: the smearing width
  int dos_points = 2001;
  bool restart = false;      // continue from <prefix>.chk (positions, density, density matrices)
  Crystal crystal;
  Params params;
  RelaxParams relax;
  CellRelaxParams cell;
  NebParams neb;
  MdParams md;
  PhononParams phonons;
  // task = phonons with phonon_supercell: the input cell is primitive; force constants from the supercell,
  // dispersion along kpath and frequencies on a q-mesh (vibrational thermodynamics, quasi-harmonic approximation)
  int phonon_supercell[3] = {1, 1, 1};
  int phonon_qgrid[3] = {8, 8, 8};
  double strain = 0.005;     // strain amplitude for task = elastic
  std::vector<Vec3> frac_final;  // second end point of a NEB path (block atoms_final)
};

Input read_input(const std::string& path);

// machine-readable summary of a finished calculation (JSON)
void write_results(const std::string& path, const Calculation& calc, bool converged, const std::vector<Vec3>* forces,
                   const Mat3* stress = nullptr);

}  // namespace lsda
