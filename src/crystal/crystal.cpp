#include "crystal/crystal.hpp"

#include <cmath>

namespace lsda {

void Crystal::set_lattice(const Mat3& a) {
  A = a;
  volume = std::abs(det3(A));
  LSDA_CHECK(volume > 1e-8, "lattice vectors are linearly dependent");
  const Mat3 inv = inverse3(A);  // A * inv = 1  ->  columns of inv are b_j / 2 pi
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) B[i][j] = 2 * pi * inv[j][i];
  A_cut = A;
  B_cut = B;
}

void Crystal::set_basis_lattice(const Mat3& a) {
  Crystal ref;
  ref.set_lattice(a);
  A_cut = ref.A;
  B_cut = ref.B;
}

Vec3 Crystal::cart(const Vec3& f) const {
  Vec3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r[j] += f[i] * A[i][j];
  return r;
}

Vec3 Crystal::frac_from_cart(const Vec3& r) const {
  Vec3 f{};
  for (int i = 0; i < 3; ++i) f[i] = dot(B[i], r) / (2 * pi);
  return f;
}

double Crystal::n_electrons() const {
  double n = 0.0;
  for (const Atom& a : atoms) n += species[a.species].setup.n_valence;
  return n;
}

}  // namespace lsda
