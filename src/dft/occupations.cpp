#include "dft/occupations.hpp"

#include <cmath>

namespace lsda {

Smearing parse_smearing(const std::string& name) {
  if (name == "fermi" || name == "fermi-dirac" || name == "fd") return Smearing::FermiDirac;
  if (name == "gauss" || name == "gaussian") return Smearing::Gaussian;
  if (name == "mp" || name == "mp1" || name == "methfessel-paxton") return Smearing::MethfesselPaxton1;
  if (name == "cold" || name == "mv" || name == "marzari-vanderbilt") return Smearing::Cold;
  fail("unknown smearing '" + name + "' (fermi, gauss, mp1, cold)");
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

Occupations compute_occupations(const std::vector<std::vector<double>>& eig, const std::vector<double>& kweight,
                                int nspin, double nelec, Smearing type, double width,
                                std::optional<double> fixed_moment) {
  Occupations occ;
  const double deg = nspin == 1 ? 2.0 : 1.0;
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
