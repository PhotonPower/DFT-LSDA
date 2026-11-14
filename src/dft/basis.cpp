#include "dft/basis.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace lsda {

void fft_dimensions(const Crystal& c, double gmax, int n[3]) {
  for (int d = 0; d < 3; ++d) {
    const int mmax = static_cast<int>(std::floor(gmax * norm(c.A_cut[d]) / (2 * pi) + 1e-10));
    n[d] = good_fft_size(2 * mmax + 1);
  }
}

GSphere make_gsphere(const Crystal& c, double ecut_dense, const FFT3D& fine, const FFT3D& coarse) {
  const double g2max = 2.0 * ecut_dense, gmax = std::sqrt(g2max);
  int mmax[3];
  for (int d = 0; d < 3; ++d) {
    mmax[d] = static_cast<int>(std::floor(gmax * norm(c.A_cut[d]) / (2 * pi) + 1e-10));
    LSDA_CHECK(2 * mmax[d] + 1 <= fine.n(d), "fine FFT grid too small for the density cutoff");
  }
  struct Entry { int m[3]; double g2; };
  std::vector<Entry> list;
  for (int m1 = -mmax[0]; m1 <= mmax[0]; ++m1)
    for (int m2 = -mmax[1]; m2 <= mmax[1]; ++m2)
      for (int m3 = -mmax[2]; m3 <= mmax[2]; ++m3) {
        double g2 = 0.0, g2cut = 0.0;  // actual length and length in the reference lattice (selection)
        for (int j = 0; j < 3; ++j) {
          const double g = m1 * c.B[0][j] + m2 * c.B[1][j] + m3 * c.B[2][j];
          const double gc = m1 * c.B_cut[0][j] + m2 * c.B_cut[1][j] + m3 * c.B_cut[2][j];
          g2 += g * g;
          g2cut += gc * gc;
        }
        if (g2cut <= g2max) list.push_back({{m1, m2, m3}, g2});
      }
  std::stable_sort(list.begin(), list.end(), [](const Entry& a, const Entry& b) { return a.g2 < b.g2; });
  GSphere s;
  s.ng = static_cast<int>(list.size());
  s.m.resize(3 * s.ng);
  s.g.resize(3 * s.ng);
  s.g2.resize(s.ng);
  s.idx_fine.resize(s.ng);
  s.idx_coarse.resize(s.ng);
  s.lookup.assign(fine.size(), -1);
  for (int i = 0; i < s.ng; ++i) {
    const int* m = list[i].m;
    bool in_coarse = true;
    for (int d = 0; d < 3; ++d) {
      s.m[3 * i + d] = m[d];
      s.g[3 * i + d] = m[0] * c.B[0][d] + m[1] * c.B[1][d] + m[2] * c.B[2][d];
      if (2 * std::abs(m[d]) >= coarse.n(d)) in_coarse = false;
    }
    s.g2[i] = list[i].g2;
    s.idx_fine[i] = fine.index(m[0], m[1], m[2]);
    s.idx_coarse[i] = in_coarse ? static_cast<long>(coarse.index(m[0], m[1], m[2])) : -1;
    s.lookup[s.idx_fine[i]] = i;
  }
  return s;
}

PWBasis make_pw_basis(const Crystal& c, const KPoint& k, double ecut, const FFT3D& coarse, double ecut_smooth) {
  PWBasis b;
  b.kfrac = k.frac;
  b.weight = k.weight;
  for (int j = 0; j < 3; ++j) b.kcart[j] = k.frac[0] * c.B[0][j] + k.frac[1] * c.B[1][j] + k.frac[2] * c.B[2][j];
  Vec3 kcut{};  // k-point in the reference lattice of the cutoff
  for (int j = 0; j < 3; ++j) kcut[j] = k.frac[0] * c.B_cut[0][j] + k.frac[1] * c.B_cut[1][j] + k.frac[2] * c.B_cut[2][j];
  const double gmax = std::sqrt(2.0 * ecut) + norm(kcut);
  int mmax[3];
  for (int d = 0; d < 3; ++d) mmax[d] = static_cast<int>(std::floor(gmax * norm(c.A_cut[d]) / (2 * pi) + 1e-10));
  for (int m1 = -mmax[0]; m1 <= mmax[0]; ++m1)
    for (int m2 = -mmax[1]; m2 <= mmax[1]; ++m2)
      for (int m3 = -mmax[2]; m3 <= mmax[2]; ++m3) {
        double q[3], q2 = 0.0, q2cut = 0.0;
        for (int j = 0; j < 3; ++j) {
          q[j] = b.kcart[j] + m1 * c.B[0][j] + m2 * c.B[1][j] + m3 * c.B[2][j];
          const double qc = kcut[j] + m1 * c.B_cut[0][j] + m2 * c.B_cut[1][j] + m3 * c.B_cut[2][j];
          q2 += q[j] * q[j];
          q2cut += qc * qc;
        }
        if (0.5 * q2cut > ecut) continue;
        LSDA_CHECK(2 * std::abs(m1) < coarse.n(0) && 2 * std::abs(m2) < coarse.n(1) && 2 * std::abs(m3) < coarse.n(2),
                   "coarse FFT grid too small for the wavefunction cutoff");
        b.idx.push_back(coarse.index(m1, m2, m3));
        b.mill.insert(b.mill.end(), {m1, m2, m3});
        double t = 0.5 * q2, dt = 1.0;
        if (ecut_smooth > 0.0) {
          // the smoothing refers to the same quantity that selects the basis (reference lattice if present)
          const double x = std::max((ecut - 0.5 * q2cut) / ecut_smooth, 1e-4);
          if (x < 1.0) {
            const double p = x * x * (3.0 + x * (1.0 + x * (-6.0 + 3.0 * x)));
            const double dp = x * (6.0 + x * (3.0 + x * (-24.0 + 15.0 * x)));
            dt = 1.0 / p + t * dp / (p * p * ecut_smooth);
            t /= p;
          }
        }
        b.kin.push_back(t);
        b.dkin.push_back(dt);
        b.kg.insert(b.kg.end(), q, q + 3);
      }
  b.npw = static_cast<int>(b.idx.size());
  return b;
}

}  // namespace lsda
