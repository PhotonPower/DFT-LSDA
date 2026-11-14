// Crystal structure, space-group symmetry and Brillouin-zone sampling.
#pragma once
#include "base/common.hpp"
#include "base/linalg.hpp"
#include "paw/setup.hpp"

namespace lsda {

struct Species {
  std::string name;   // label used in the input (e.g. "Ni", "Fe1")
  std::string file;   // PAW dataset
  double mass = 0;    // atomic mass (amu) for phonons; 0: standard atomic weight of the element
  PawSetup setup;
};

struct Atom {
  int species = 0;
  Vec3 frac{};        // fractional coordinates
  double magmom = 0;  // initial magnetic moment (mu_B)
};

struct Crystal {
  Mat3 A{};  // rows: lattice vectors a_i (bohr)
  Mat3 B{};  // rows: reciprocal vectors b_i with a_i . b_j = 2 pi delta_ij
  double volume = 0;
  // Lattice that defines which plane waves belong to the basis (|k+G|^2/2 <= ecut is evaluated with it).
  // Normally identical to A/B; a different reference lattice keeps the basis set fixed when the cell is
  // strained (constant-basis energies, e.g. for finite-difference tests of the stress tensor).
  Mat3 A_cut{}, B_cut{};
  std::vector<Species> species;
  std::vector<Atom> atoms;
  // Further sets of fractional positions (same atom order) that every symmetry operation has to map onto
  // themselves with the same permutation of atoms, e.g. the end points of a transition path.
  std::vector<std::vector<Vec3>> sym_constraints;

  void set_lattice(const Mat3& a);              // also resets the basis reference lattice
  void set_basis_lattice(const Mat3& a);
  Vec3 cart(const Vec3& frac) const;          // r = sum_i f_i a_i
  Vec3 frac_from_cart(const Vec3& r) const;
  Vec3 atom_cart(int ia) const { return cart(atoms[ia].frac); }
  const PawSetup& setup(int ia) const { return species[atoms[ia].species].setup; }
  double n_electrons() const;
  int natoms() const { return static_cast<int>(atoms.size()); }
};

// Space-group operation x' = W x + t (fractional coordinates)
struct SymOp {
  int W[3][3];
  Vec3 t{};
  Mat3 R{};                   // Cartesian rotation matrix
  int Wk[3][3];               // action on fractional k / integer G vectors: (W^{-1})^T
  std::vector<int> atom_map;  // image of each atom
};

// All symmetry operations of the crystal (atoms distinguished by species and initial moment).
std::vector<SymOp> find_symmetry(const Crystal& c, bool enabled, double tol = 1e-5);

struct KPoint {
  Vec3 frac{};
  double weight = 0;
};

// Monkhorst-Pack grid k = (i + shift)/n reduced with the symmetry operations and time reversal.
// Operations that do not map the grid onto itself are removed from `ops`.
std::vector<KPoint> make_kpoints(const int n[3], const Vec3& shift, std::vector<SymOp>& ops);

}  // namespace lsda
