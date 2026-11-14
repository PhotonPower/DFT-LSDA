#include "dft/occupations.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace lsda {

Smearing parse_smearing(const std::string& name) {
  if (name == "fermi" || name == "fermi-dirac" || name == "fd") return Smearing::FermiDirac;
  if (name == "gauss" || name == "gaussian") return Smearing::Gaussian;
  if (name == "mp" || name == "mp1" || name == "methfessel-paxton") return Smearing::MethfesselPaxton1;
  if (name == "cold" || name == "mv" || name == "marzari-vanderbilt") return Smearing::Cold;
  if (name == "tetrahedron" || name == "tetra" || name == "bloechl") return Smearing::Tetrahedron;
  fail("unknown smearing '" + name + "' (fermi, gauss, mp1, cold, tetrahedron)");
}

double smearing_occupation(Smearing type, double x) {
  const double sqrtpi = std::sqrt(pi);
  switch (type) {
    case Smearing::FermiDirac:
      return x > 40 ? 1.0 : (x < -40 ? 0.0 : 1.0 / (1.0 + std::exp(-x)));
    case Smearing::Gaussian:
      return 0.5 * std::erfc(-x);
    case Smearing::MethfesselPaxton1:
      return 0.5 * std::erfc(-x) + x * std::exp(-std::min(x * x, 200.0)) / (2.0 * sqrtpi);
    case Smearing::Cold: {
      const double xp = x - 1.0 / std::sqrt(2.0);
      return 0.5 * std::erf(xp) + std::exp(-std::min(xp * xp, 200.0)) / std::sqrt(2.0 * pi) + 0.5;
    }
    case Smearing::Tetrahedron:
      return x > 0 ? 1.0 : 0.0;
  }
  return 0.0;
}

double smearing_entropy(Smearing type, double x) {
  const double sqrtpi = std::sqrt(pi);
  switch (type) {
    case Smearing::FermiDirac: {
      if (std::abs(x) > 36) return 0.0;
      const double f = 1.0 / (1.0 + std::exp(-x));
      return f * std::log(f) + (1.0 - f) * std::log(1.0 - f);
    }
    case Smearing::Gaussian:
      return -std::exp(-std::min(x * x, 200.0)) / (2.0 * sqrtpi);
    case Smearing::MethfesselPaxton1:
      return (2.0 * x * x - 1.0) * std::exp(-std::min(x * x, 200.0)) / (4.0 * sqrtpi);
    case Smearing::Cold: {
      const double xp = x - 1.0 / std::sqrt(2.0);
      return xp * std::exp(-std::min(xp * xp, 200.0)) / std::sqrt(2.0 * pi);
    }
    case Smearing::Tetrahedron:
      return 0.0;
  }
  return 0.0;
}

namespace {

// number of electrons in the given spin channels for chemical potential mu
double count(const std::vector<std::vector<double>>& eig, const std::vector<double>& kw, int nspin, int s0, int s1,
             double degeneracy, Smearing type, double width, double mu) {
  double n = 0.0;
  for (size_t ik = 0; ik < kw.size(); ++ik)
    for (int s = s0; s < s1; ++s)
      for (double e : eig[ik * nspin + s]) n += degeneracy * kw[ik] * smearing_occupation(type, (mu - e) / width);
  return n;
}

double find_mu(const std::vector<std::vector<double>>& eig, const std::vector<double>& kw, int nspin, int s0, int s1,
               double degeneracy, Smearing type, double width, double target) {
  double lo = 1e300, hi = -1e300, capacity = 0.0;
  for (size_t ik = 0; ik < kw.size(); ++ik)
    for (int s = s0; s < s1; ++s) {
      for (double e : eig[ik * nspin + s]) {
        lo = std::min(lo, e);
        hi = std::max(hi, e);
      }
      capacity += degeneracy * kw[ik] * eig[ik * nspin + s].size();
    }
  LSDA_CHECK(target <= capacity + 1e-9, "not enough bands for the number of electrons (increase nbands)");
  lo -= 20 * width + 1.0;
  hi += 20 * width + 1.0;
  for (int it = 0; it < 200; ++it) {
    const double mid = 0.5 * (lo + hi);
    if (count(eig, kw, nspin, s0, s1, degeneracy, type, width, mid) < target)
      lo = mid;
    else
      hi = mid;
    if (hi - lo < 1e-14) break;
  }
  return 0.5 * (lo + hi);
}

}  // namespace

void tetrahedron_weights(const double ein[4], double ef, double vt, bool bloechl, double w[4]) {
  // sort the corners; equal energies are split by a tiny amount to avoid vanishing denominators
  int idx[4] = {0, 1, 2, 3};
  std::sort(idx, idx + 4, [&](int a, int b) { return ein[a] < ein[b]; });
  double e[4];
  for (int i = 0; i < 4; ++i) e[i] = ein[idx[i]];
  const double tiny = 1e-10 * (1.0 + std::abs(e[0]) + std::abs(e[3]));
  for (int i = 1; i < 4; ++i)
    if (e[i] - e[i - 1] < tiny) e[i] = e[i - 1] + tiny;
  const double e21 = e[1] - e[0], e31 = e[2] - e[0], e41 = e[3] - e[0], e32 = e[2] - e[1], e42 = e[3] - e[1], e43 = e[3] - e[2];
  double ws[4] = {0, 0, 0, 0}, dos = 0.0;
  const double q = 0.25 * vt;
  if (ef <= e[0]) {
    // empty
  } else if (ef < e[1]) {
    const double x = ef - e[0], c = q * x * x * x / (e21 * e31 * e41);
    ws[0] = c * (4.0 - x * (1.0 / e21 + 1.0 / e31 + 1.0 / e41));
    ws[1] = c * x / e21;
    ws[2] = c * x / e31;
    ws[3] = c * x / e41;
    dos = vt * 3.0 * x * x / (e21 * e31 * e41);
  } else if (ef < e[2]) {
    const double c1 = q * (ef - e[0]) * (ef - e[0]) / (e41 * e31);
    const double c2 = q * (ef - e[0]) * (ef - e[1]) * (e[2] - ef) / (e41 * e32 * e31);
    const double c3 = q * (ef - e[1]) * (ef - e[1]) * (e[3] - ef) / (e42 * e32 * e41);
    ws[0] = c1 + (c1 + c2) * (e[2] - ef) / e31 + (c1 + c2 + c3) * (e[3] - ef) / e41;
    ws[1] = c1 + c2 + c3 + (c2 + c3) * (e[2] - ef) / e32 + c3 * (e[3] - ef) / e42;
    ws[2] = (c1 + c2) * (ef - e[0]) / e31 + (c2 + c3) * (ef - e[1]) / e32;
    ws[3] = (c1 + c2 + c3) * (ef - e[0]) / e41 + c3 * (ef - e[1]) / e42;
    dos = vt / (e31 * e41) * (3.0 * e21 + 6.0 * (ef - e[1]) - 3.0 * (e31 + e42) * (ef - e[1]) * (ef - e[1]) / (e32 * e42));
  } else if (ef < e[3]) {
    const double x = e[3] - ef, c = q * x * x * x / (e41 * e42 * e43);
    ws[0] = q - c * x / e41;
    ws[1] = q - c * x / e42;
    ws[2] = q - c * x / e43;
    ws[3] = q - c * (4.0 - x * (1.0 / e41 + 1.0 / e42 + 1.0 / e43));
    dos = vt * 3.0 * x * x / (e41 * e42 * e43);
  } else {
    for (double& v : ws) v = q;
  }
  if (bloechl && dos != 0.0) {  // correction for the curvature of the bands: dw_i = D_T(ef)/40 sum_j (e_j - e_i)
    const double esum = e[0] + e[1] + e[2] + e[3];
    for (int i = 0; i < 4; ++i) ws[i] += dos / 40.0 * (esum - 4.0 * e[i]);
  }
  for (int i = 0; i < 4; ++i) w[idx[i]] = ws[i];
}

namespace {

// corner indices (irreducible k-points) of all tetrahedra of the grid
std::vector<std::array<int, 4>> make_tetrahedra(const TetraGrid& g) {
  const int n0 = g.n[0], n1 = g.n[1], n2 = g.n[2];
  LSDA_CHECK(static_cast<int>(g.map.size()) == n0 * n1 * n2, "tetrahedron method: inconsistent k-point grid map");
  // microcell diagonal of shortest length: flip mask f, diagonal from corner f to its opposite corner
  int best = 0;
  double dmin = 1e300;
  for (int f = 0; f < 4; ++f) {
    Vec3 d{};
    for (int a = 0; a < 3; ++a) {
      const double sign = (f >> a) & 1 ? -1.0 : 1.0;
      for (int j = 0; j < 3; ++j) d[j] += sign * g.B[a][j] / g.n[a];
    }
    if (norm(d) < dmin - 1e-12) {
      dmin = norm(d);
      best = f;
    }
  }
  const int perm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  std::vector<std::array<int, 4>> tets;
  tets.reserve(static_cast<size_t>(6) * n0 * n1 * n2);
  for (int i0 = 0; i0 < n0; ++i0)
    for (int i1 = 0; i1 < n1; ++i1)
      for (int i2 = 0; i2 < n2; ++i2)
        for (const auto& p : perm) {
          std::array<int, 4> t;
          int c[3] = {0, 0, 0};
          for (int v = 0; v < 4; ++v) {
            if (v > 0) c[p[v - 1]] = 1;
            int m[3];
            const int base[3] = {i0, i1, i2};
            for (int a = 0; a < 3; ++a) m[a] = (base[a] + (((best >> a) & 1) ? 1 - c[a] : c[a])) % g.n[a];
            t[v] = g.map[(m[0] * n1 + m[1]) * n2 + m[2]];
          }
          tets.push_back(t);
        }
  return tets;
}

Occupations tetrahedron_occupations(const std::vector<std::vector<double>>& eig, const std::vector<double>& kweight, int nspin,
                                    double nelec, const TetraGrid& grid, double degeneracy) {
  const auto tets = make_tetrahedra(grid);
  const double vt = 1.0 / static_cast<double>(tets.size()), deg = degeneracy > 0.0 ? degeneracy : (nspin == 1 ? 2.0 : 1.0);
  const int nk = static_cast<int>(kweight.size()), nb = static_cast<int>(eig[0].size());
  auto accumulate = [&](double ef, bool bloechl, std::vector<std::vector<double>>* wout) {
    double total = 0.0;
    if (wout) wout->assign(eig.size(), std::vector<double>(nb, 0.0));
    for (int s = 0; s < nspin; ++s)
      for (int n = 0; n < nb; ++n)
        for (const auto& t : tets) {
          double e[4], w[4];
          for (int v = 0; v < 4; ++v) e[v] = eig[static_cast<size_t>(t[v]) * nspin + s][n];
          if (std::min({e[0], e[1], e[2], e[3]}) >= ef) continue;
          tetrahedron_weights(e, ef, vt, bloechl, w);
          for (int v = 0; v < 4; ++v) {
            total += deg * w[v];
            if (wout) (*wout)[static_cast<size_t>(t[v]) * nspin + s][n] += w[v];
          }
        }
    return total;
  };
  double lo = 1e300, hi = -1e300;
  for (const auto& e : eig)
    for (double v : e) {
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
  LSDA_CHECK(nelec <= deg * nb * (nspin == 2 ? 2 : 1) + 1e-9, "not enough bands for the number of electrons (increase nbands)");
  for (int it = 0; it < 200 && hi - lo > 1e-13; ++it) {
    const double mid = 0.5 * (lo + hi);
    if (accumulate(mid, false, nullptr) < nelec)
      lo = mid;
    else
      hi = mid;
  }
  Occupations occ;
  occ.efermi[0] = occ.efermi[1] = 0.5 * (lo + hi);
  std::vector<std::vector<double>> w;
  accumulate(occ.efermi[0], true, &w);
  occ.f.assign(eig.size(), std::vector<double>(nb, 0.0));
  for (int ik = 0; ik < nk; ++ik)
    for (int s = 0; s < nspin; ++s) {
      const size_t ks = static_cast<size_t>(ik) * nspin + s;
      for (int n = 0; n < nb; ++n) {
        occ.f[ks][n] = deg * w[ks][n] / kweight[ik];  // occupation such that kweight * f is the integration weight
        occ.band_energy += kweight[ik] * occ.f[ks][n] * eig[ks][n];
        occ.n_spin[s] += kweight[ik] * occ.f[ks][n];
      }
    }
  if (nspin == 1) occ.n_spin[0] = occ.n_spin[1] = 0.5 * occ.n_spin[0];
  return occ;
}

}  // namespace

Occupations compute_occupations(const std::vector<std::vector<double>>& eig, const std::vector<double>& kweight,
                                int nspin, double nelec, Smearing type, double width,
                                std::optional<double> fixed_moment, const TetraGrid* tetra, double degeneracy) {
  if (type == Smearing::Tetrahedron) {
    LSDA_CHECK(tetra != nullptr, "tetrahedron method: k-point grid information missing");
    LSDA_CHECK(!fixed_moment, "tetrahedron method: fixed_moment is not supported");
    return tetrahedron_occupations(eig, kweight, nspin, nelec, *tetra, degeneracy);
  }
  Occupations occ;
  const double deg = degeneracy > 0.0 ? degeneracy : (nspin == 1 ? 2.0 : 1.0);
  if (nspin == 2 && fixed_moment) {
    occ.efermi[0] = find_mu(eig, kweight, 2, 0, 1, 1.0, type, width, 0.5 * (nelec + *fixed_moment));
    occ.efermi[1] = find_mu(eig, kweight, 2, 1, 2, 1.0, type, width, 0.5 * (nelec - *fixed_moment));
  } else {
    occ.efermi[0] = occ.efermi[1] = find_mu(eig, kweight, nspin, 0, nspin, deg, type, width, nelec);
  }
  occ.f.resize(eig.size());
  for (size_t ik = 0; ik < kweight.size(); ++ik)
    for (int s = 0; s < nspin; ++s) {
      const auto& e = eig[ik * nspin + s];
      auto& f = occ.f[ik * nspin + s];
      f.resize(e.size());
      for (size_t n = 0; n < e.size(); ++n) {
        const double x = (occ.efermi[s] - e[n]) / width;
        f[n] = deg * smearing_occupation(type, x);
        occ.entropy_term += width * deg * kweight[ik] * smearing_entropy(type, x);
        occ.band_energy += kweight[ik] * f[n] * e[n];
        occ.n_spin[s] += kweight[ik] * f[n];
      }
    }
  if (nspin == 1) occ.n_spin[0] = occ.n_spin[1] = 0.5 * occ.n_spin[0];
  return occ;
}

}  // namespace lsda
