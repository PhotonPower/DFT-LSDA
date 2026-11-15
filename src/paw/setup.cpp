#include <cstdio>
#include <cstdlib>
#include "paw/setup.hpp"

#include <cmath>

#include "paw/xml.hpp"

namespace lsda {

PawSetup PawSetup::load(const std::string& path, int ntheta) {
  const auto root = parse_xml_file(path);
  LSDA_CHECK(root->name == "paw_dataset" || root->name == "paw_setup", "not a PAW-XML dataset: " + path);
  PawSetup s;
  s.path = path;
  const XmlNode* atom = root->child("atom");
  LSDA_CHECK(atom, "PAW-XML: missing <atom>");
  s.symbol = atom->str("symbol");
  s.Z = atom->num("Z");
  s.n_core = atom->num("core");
  s.n_valence = atom->num("valence");
  if (const XmlNode* x = root->child("xc_functional")) s.xc_name = x->str("type") + "/" + x->str("name");
  if (const XmlNode* x = root->child("ae_energy")) s.e_total_ae = x->num("total");
  if (const XmlNode* x = root->child("core_energy")) s.e_kin_core = x->num("kinetic");
  if (const XmlNode* x = root->child("pw_ecut")) {
    s.ecut_low = x->num("low", x->num("low_cutoff", 0.0));
    s.ecut_high = x->num("high", x->num("high_cutoff", 0.0));
  }

  // radial grid (a single grid is supported)
  const auto grids = root->children_named("radial_grid");
  LSDA_CHECK(grids.size() == 1, "PAW-XML: datasets with several radial grids are not supported yet");
  const std::string grid_id = grids[0]->str("id");
  LSDA_CHECK(grids[0]->child("values") && grids[0]->child("derivatives"), "PAW-XML: radial grid without values");
  s.grid.r = grids[0]->child("values")->values();
  s.grid.dr = grids[0]->child("derivatives")->values();
  const size_t n = s.grid.r.size();
  LSDA_CHECK(n > 10 && s.grid.dr.size() == n, "PAW-XML: inconsistent radial grid");
  auto radial = [&](const XmlNode* node, const char* what) {
    LSDA_CHECK(node, std::string("PAW-XML: missing <") + what + ">");
    LSDA_CHECK(node->str("grid") == grid_id, "PAW-XML: unknown radial grid");
    std::vector<double> v = node->values();
    LSDA_CHECK(v.size() == n, std::string("PAW-XML: wrong number of values in <") + what + ">");
    return v;
  };

  // valence states / partial waves
  const XmlNode* states = root->child("valence_states");
  LSDA_CHECK(states, "PAW-XML: missing <valence_states>");
  std::vector<std::string> ids;
  double rc_max = 0.0;
  for (const XmlNode* st : states->children_named("state")) {
    s.l_j.push_back(static_cast<int>(st->num("l")));
    s.n_j.push_back(static_cast<int>(st->num("n", -1.0)));
    s.f_j.push_back(st->num("f", 0.0));
    s.e_j.push_back(st->num("e", 0.0));
    rc_max = std::max(rc_max, st->num("rc", 0.0));
    ids.push_back(st->str("id"));
  }
  s.nj = static_cast<int>(ids.size());
  s.phi.resize(s.nj);
  s.phit.resize(s.nj);
  s.proj.resize(s.nj);
  auto find_state = [&](const std::vector<const XmlNode*>& nodes, const std::string& id, const char* what) {
    for (const XmlNode* nd : nodes)
      if (nd->str("state") == id) return radial(nd, what);
    fail(std::string("PAW-XML: missing <") + what + "> for state " + id);
  };
  const auto ae = root->children_named("ae_partial_wave"), ps = root->children_named("pseudo_partial_wave"),
             pr = root->children_named("projector_function");
  for (int j = 0; j < s.nj; ++j) {
    s.phi[j] = find_state(ae, ids[j], "ae_partial_wave");
    s.phit[j] = find_state(ps, ids[j], "pseudo_partial_wave");
    s.proj[j] = find_state(pr, ids[j], "projector_function");
  }
  s.nc = radial(root->child("ae_core_density"), "ae_core_density");
  s.nct = radial(root->child("pseudo_core_density"), "pseudo_core_density");
  s.vbar = radial(root->child("zero_potential"), "zero_potential");
  if (root->child("pseudo_valence_density"))
    s.nvt = radial(root->child("pseudo_valence_density"), "pseudo_valence_density");

  s.r_paw = rc_max;
  if (const XmlNode* x = root->child("paw_radius")) s.r_paw = std::max(s.r_paw, x->num("rc"));
  LSDA_CHECK(s.r_paw > 0.0, "PAW-XML: no augmentation radius");

  // kinetic energy differences
  const XmlNode* kin = root->child("kinetic_energy_differences");
  LSDA_CHECK(kin, "PAW-XML: missing <kinetic_energy_differences>");
  const std::vector<double> dk = kin->values();
  LSDA_CHECK(static_cast<int>(dk.size()) == s.nj * s.nj, "PAW-XML: wrong size of kinetic energy matrix");
  // exact-exchange data (optional): valence-core matrix per radial pair and core-core energy
  std::vector<double> xvc;
  if (const XmlNode* x = root->child("exact_exchange_X_matrix")) xvc = x->values();
  if (const XmlNode* x = root->child("exact_exchange")) s.exx_core = x->num("core-core");
  s.has_exx_data = static_cast<int>(xvc.size()) == s.nj * s.nj;

  // integration range: a few points beyond r_paw, even number of intervals
  s.ncut = s.grid.index_of(s.r_paw) + 3;
  if (s.ncut % 2 == 0) ++s.ncut;
  LSDA_CHECK(s.ncut < static_cast<int>(n), "PAW-XML: radial grid ends inside the augmentation sphere");

  // projector function indices
  for (int j = 0; j < s.nj; ++j) {
    s.lmax = std::max(s.lmax, s.l_j[j]);
    for (int m = -s.l_j[j]; m <= s.l_j[j]; ++m) {
      s.j_i.push_back(j);
      s.l_i.push_back(s.l_j[j]);
      s.L_i.push_back(lm_index(s.l_j[j], m));
    }
  }
  s.ni = static_cast<int>(s.j_i.size());
  s.nL = (2 * s.lmax + 1) * (2 * s.lmax + 1);

  // shape functions g_l(r) ~ r^l k(r), normalized to unit multipole moment
  const XmlNode* shp = root->child("shape_function");
  LSDA_CHECK(shp, "PAW-XML: missing <shape_function>");
  const std::string type = shp->str("type");
  s.rc_shape = shp->num("rc");
  std::vector<double> k(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    const double x = s.grid.r[i] / s.rc_shape;
    if (type == "gauss") {
      k[i] = std::exp(-x * x);
    } else if (type == "sinc") {
      if (x < 1.0) k[i] = x > 1e-12 ? std::pow(std::sin(pi * x) / (pi * x), 2) : 1.0;
    } else if (type == "exp") {
      k[i] = std::exp(-std::pow(x, shp->num("lamb")));
    } else {
      fail("PAW-XML: shape function type '" + type + "' is not supported");
    }
  }
  const double shape_extent = type == "sinc" ? s.rc_shape : 6.0 * s.rc_shape;
  LSDA_CHECK(shape_extent <= s.grid.r[s.ncut - 1] || type != "sinc",
             "PAW-XML: shape function extends beyond the augmentation sphere");
  const std::vector<double> w = s.grid.weights(s.ncut);
  s.wr2_.resize(s.ncut);
  for (int i = 0; i < s.ncut; ++i) s.wr2_[i] = w[i] * s.grid.r[i] * s.grid.r[i];
  s.shape_.resize(2 * s.lmax + 1);
  for (int l = 0; l <= 2 * s.lmax; ++l) {
    std::vector<double>& g = s.shape_[l];
    g.resize(s.ncut);
    double nrm = 0.0;
    for (int i = 0; i < s.ncut; ++i) {
      g[i] = std::pow(s.grid.r[i], l) * k[i];
      nrm += s.wr2_[i] * std::pow(s.grid.r[i], l) * g[i];
    }
    for (double& v : g) v /= nrm;
  }

  // diagnostics: core charge and projector/partial-wave duality
  {
    const std::vector<double> wfull = s.grid.weights(static_cast<int>(n));
    double q = 0.0;
    for (size_t i = 0; i < n; ++i) q += wfull[i] * s.grid.r[i] * s.grid.r[i] * s.nc[i];
    s.check_core_charge = std::sqrt(4 * pi) * q;
    double dev = 0.0;
    for (int j1 = 0; j1 < s.nj; ++j1)
      for (int j2 = 0; j2 < s.nj; ++j2) {
        if (s.l_j[j1] != s.l_j[j2]) continue;
        double o = 0.0;
        for (int i = 0; i < s.ncut; ++i) o += s.wr2_[i] * s.proj[j1][i] * s.phit[j2][i];
        dev = std::max(dev, std::abs(o - (j1 == j2 ? 1.0 : 0.0)));
      }
    s.check_duality = dev;
  }

  // kinetic part of B; the Coulomb and zero-potential parts are added in compute_onsite_tensors
  s.B.resize(s.ni, s.ni);
  for (int i = 0; i < s.ni; ++i)
    for (int j = 0; j < s.ni; ++j)
      if (s.L_i[i] == s.L_i[j]) s.B(i, j) = dk[s.j_i[i] * s.nj + s.j_i[j]];
  s.X.resize(s.ni, s.ni);
  if (s.has_exx_data)
    for (int i = 0; i < s.ni; ++i)
      for (int j = 0; j < s.ni; ++j)
        if (s.L_i[i] == s.L_i[j]) s.X(i, j) = 0.5 * (xvc[s.j_i[i] * s.nj + s.j_i[j]] + xvc[s.j_i[j] * s.nj + s.j_i[i]]);
  {
    // Sign convention: X is kept as the positive matrix X_ij = sum_c (i c|c j) over the core orbitals of one spin
    // direction (PAW-XML specification), E_vc = -sum_s sum_ij D^s_ij X_ij. The JTH/atompaw datasets store the
    // matrix with the sign of the energy contribution (negative diagonal).
    double tr = 0.0;
    for (int i = 0; i < s.ni; ++i) tr += s.X(i, i);
    if (tr < 0.0)
      for (size_t i = 0; i < s.X.size(); ++i) s.X.data()[i] = -s.X.data()[i];
  }

  s.quad_ = make_sphere_quadrature(ntheta);
  s.ylm_quad_.resize(s.nL, static_cast<int>(s.quad_.points.size()));
  for (size_t p = 0; p < s.quad_.points.size(); ++p)
    real_ylm(2 * s.lmax, s.quad_.points[p].data(), s.ylm_quad_.col(static_cast<int>(p)));

  s.ylm_grad_quad_.resize(3 * static_cast<size_t>(s.nL) * s.quad_.points.size());
  {
    std::vector<double> y(s.nL);
    for (size_t p = 0; p < s.quad_.points.size(); ++p)
      real_ylm_grad(2 * s.lmax, s.quad_.points[p].data(), y.data(), &s.ylm_grad_quad_[3 * s.nL * p]);
  }

  s.compute_onsite_tensors();
  return s;
}

void PawSetup::compute_onsite_tensors() {
  const int nfull = grid.size();
  const GauntTable gaunt(lmax);
  const double s4pi = std::sqrt(4 * pi);
  nq_ = nj * (nj + 1) / 2;

  // pair densities inside the sphere
  nq_ae_.assign(nq_, std::vector<double>(ncut));
  nq_ps_.assign(nq_, std::vector<double>(ncut));
  std::vector<int> q_l1(nq_), q_l2(nq_);
  for (int j1 = 0; j1 < nj; ++j1)
    for (int j2 = j1; j2 < nj; ++j2) {
      const int q = pair(j1, j2);
      q_l1[q] = l_j[j1];
      q_l2[q] = l_j[j2];
      for (int i = 0; i < ncut; ++i) {
        nq_ae_[q][i] = phi[j1][i] * phi[j2][i];
        nq_ps_[q][i] = phit[j1][i] * phit[j2][i];
      }
    }
  // radial derivatives for gradient-corrected functionals
  dnq_ae_.assign(nq_, {});
  dnq_ps_.assign(nq_, {});
  for (int j1 = 0; j1 < nj; ++j1)
    for (int j2 = j1; j2 < nj; ++j2) {
      std::vector<double> fa(ncut + 2), fp(ncut + 2);
      for (int i = 0; i < ncut + 2; ++i) {
        fa[i] = phi[j1][i] * phi[j2][i];
        fp[i] = phit[j1][i] * phit[j2][i];
      }
      dnq_ae_[pair(j1, j2)] = grid.derivative(fa, ncut);
      dnq_ps_[pair(j1, j2)] = grid.derivative(fp, ncut);
    }
  dnc_ = grid.derivative(nc, ncut);
  dnct_ = grid.derivative(nct, ncut);

  auto allowed = [&](int q, int l) {
    return l >= std::abs(q_l1[q] - q_l2[q]) && l <= q_l1[q] + q_l2[q] && (l + q_l1[q] + q_l2[q]) % 2 == 0;
  };
  auto inner = [&](const std::vector<double>& a, const std::vector<double>& b) {  // int r^2 a b dr over the sphere
    double sum = 0.0;
    for (int i = 0; i < ncut; ++i) sum += wr2_[i] * a[i] * b[i];
    return sum;
  };

  // ---- spherical (core) quantities on the full grid -----------------------------
  const std::vector<double> wfull = grid.weights(nfull);
  std::vector<double> wfull_r2(nfull);
  for (int i = 0; i < nfull; ++i) wfull_r2[i] = wfull[i] * grid.r[i] * grid.r[i];
  const std::vector<double> v_nc = grid.hartree(0, nc, nfull), v_nct = grid.hartree(0, nct, nfull);
  std::vector<double> g0_full(nfull, 0.0);
  for (int i = 0; i < ncut; ++i) g0_full[i] = shape_[0][i];
  const std::vector<double> v_g0_full = grid.hartree(0, g0_full, nfull);

  double ncore = 0.0, ncore_t = 0.0, e_nc = 0.0, e_nct = 0.0, e_nuc = 0.0, e_nct_g0 = 0.0, e_vbar = 0.0;
  for (int i = 0; i < nfull; ++i) {
    ncore += wfull_r2[i] * nc[i];
    ncore_t += wfull_r2[i] * nct[i];
    e_nc += 0.5 * wfull_r2[i] * nc[i] * v_nc[i];
    e_nct += 0.5 * wfull_r2[i] * nct[i] * v_nct[i];
    e_nuc += wfull[i] * grid.r[i] * nc[i];
    e_nct_g0 += wfull_r2[i] * nct[i] * v_g0_full[i];
    e_vbar += wfull_r2[i] * nct[i] * vbar[i];
  }
  Delta0 = ncore - ncore_t - Z / s4pi;
  const double e_g0g0 = inner(shape_[0], std::vector<double>(v_g0_full.begin(), v_g0_full.begin() + ncut));
  E0 = e_kin_core + e_nc - e_nct - Z * s4pi * e_nuc - Delta0 * e_nct_g0 - 0.5 * Delta0 * Delta0 * e_g0g0 - e_vbar;

  // ---- multipole moments and radial Coulomb integrals ---------------------------
  const int nl = 2 * lmax + 1;
  std::vector<std::vector<double>> dl(nl, std::vector<double>(nq_, 0.0));  // Delta^l_q
  std::vector<RMatrix> R(nl);
  for (int l = 0; l < nl; ++l) {
    R[l].resize(nq_, nq_);
    const std::vector<double> vg = grid.hartree(l, shape_[l], ncut);
    const double gg = inner(shape_[l], vg);
    std::vector<std::vector<double>> v_ae(nq_), v_ps(nq_);
    std::vector<double> ps_g(nq_, 0.0);
    for (int q = 0; q < nq_; ++q) {
      if (!allowed(q, l)) continue;
      for (int i = 0; i < ncut; ++i) dl[l][q] += wr2_[i] * std::pow(grid.r[i], l) * (nq_ae_[q][i] - nq_ps_[q][i]);
      v_ae[q] = grid.hartree(l, nq_ae_[q], ncut);
      v_ps[q] = grid.hartree(l, nq_ps_[q], ncut);
      ps_g[q] = inner(nq_ps_[q], vg);
    }
    for (int q = 0; q < nq_; ++q)
      for (int p = 0; p < nq_; ++p) {
        if (!allowed(q, l) || !allowed(p, l)) continue;
        R[l](q, p) = 0.5 * (inner(nq_ae_[q], v_ae[p]) - inner(nq_ps_[q], v_ps[p])) - 0.5 * dl[l][q] * ps_g[p] -
                     0.5 * dl[l][p] * ps_g[q] - 0.5 * dl[l][q] * dl[l][p] * gg;
      }
    for (int q = 0; q < nq_; ++q)  // enforce exact symmetry
      for (int p = 0; p < q; ++p) R[l](q, p) = R[l](p, q) = 0.5 * (R[l](q, p) + R[l](p, q));
  }

  // ---- linear term B (kinetic part is already stored) ---------------------------
  std::vector<double> bq(nq_, 0.0), sq(nq_, 0.0), nrm(nq_, 0.0);
  const std::vector<double> v_g0(v_g0_full.begin(), v_g0_full.begin() + ncut);
  for (int q = 0; q < nq_; ++q) {
    if (q_l1[q] != q_l2[q]) continue;
    double t_nc = 0.0, t_nct = 0.0, t_nuc = 0.0, t_vbar = 0.0;
    for (int i = 0; i < ncut; ++i) {
      t_nc += wr2_[i] * nq_ae_[q][i] * v_nc[i];
      t_nct += wr2_[i] * nq_ps_[q][i] * v_nct[i];
      t_nuc += (grid.r[i] > 0.0 ? wr2_[i] / grid.r[i] : 0.0) * nq_ae_[q][i];
      t_vbar += wr2_[i] * nq_ps_[q][i] * vbar[i];
      nrm[q] += wr2_[i] * nq_ae_[q][i];
    }
    bq[q] = t_nc - t_nct - Z * s4pi * t_nuc - Delta0 * inner(nq_ps_[q], v_g0) -
            dl[0][q] * (e_nct_g0 + Delta0 * e_g0g0) - t_vbar;
    sq[q] = dl[0][q];
  }

  ae_norm_q_ = nrm;

  // ---- expand to the (i,j) representation ---------------------------------------
  dS.resize(ni, ni);
  dN.resize(ni, ni);
  Delta_Lij.assign(static_cast<size_t>(nL) * ni * ni, 0.0);
  gaunt_ij_.assign(static_cast<size_t>(ni) * ni, {});
  const double g00 = 1.0 / s4pi;
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < ni; ++j) {
      const int q = pair(j_i[i], j_i[j]);
      if (L_i[i] == L_i[j]) {
        B(i, j) += g00 * bq[q];
        dS(i, j) = s4pi * g00 * sq[q];
        dN(i, j) = nrm[q];
      }
      for (int L = 0; L < nL; ++L) {
        const double g = gaunt(L, L_i[i], L_i[j]);
        if (g == 0.0) continue;
        const int l = static_cast<int>(std::floor(std::sqrt(static_cast<double>(L)) + 1e-9));
        Delta_Lij[(static_cast<size_t>(L) * ni + i) * ni + j] = g * dl[l][q];
        gaunt_ij_[static_cast<size_t>(i) * ni + j].push_back({L, q, g});
      }
    }
  radial_coulomb_ = R;
  delta_lq_ = dl;
  C = assemble_coulomb(R);
}

// C_ij,kl = sum_L G_L,ij G_L,kl R_l(q_ij, q_kl) from the radial integrals
RMatrix PawSetup::assemble_coulomb(const std::vector<RMatrix>& R) const {
  RMatrix c(ni * ni, ni * ni);
  for (int ij = 0; ij < ni * ni; ++ij)
    for (int kl = 0; kl < ni * ni; ++kl) {
      double sum = 0.0;
      for (const GauntEntry& a : gaunt_ij_[ij])
        for (const GauntEntry& b : gaunt_ij_[kl])
          if (a.L == b.L) {
            const int l = static_cast<int>(std::floor(std::sqrt(static_cast<double>(a.L)) + 1e-9));
            sum += a.g * b.g * R[l](a.q, b.q);
          }
      c(ij, kl) = sum;
    }
  return c;
}

// Coulomb tensor of the on-site exchange. For the screened kernel erfc(omega r)/r the long-range part
// erf(omega r)/r is subtracted; its multipole components
//   K_l(r, r') = (2l+1)/2 int_-1^1 P_l(x) erf(omega d)/d dx,  d^2 = r^2 + r'^2 - 2 r r' x
// are integrated numerically (the kernel is smooth), the pseudo pair densities include their compensation charges.
void PawSetup::prepare_exchange(double omega) {
  if (exx_omega == omega) return;
  exx_omega = omega;
  if (omega <= 0.0) {
    Cx = C;
    return;
  }
  const int nl = 2 * lmax + 1, nx = 24;
  std::vector<double> x, w;
  gauss_legendre(nx, x, w);
  std::vector<double> pl(static_cast<size_t>(nl) * nx);  // (2l+1)/2 w P_l(x)
  for (int k = 0; k < nx; ++k) {
    double p0 = 1.0, p1 = x[k];
    for (int l = 0; l < nl; ++l) {
      const double p = l == 0 ? p0 : (l == 1 ? p1 : ((2 * l - 1) * x[k] * p1 - (l - 1) * p0) / l);
      if (l >= 2) {
        p0 = p1;
        p1 = p;
      }
      pl[static_cast<size_t>(l) * nx + k] = 0.5 * (2 * l + 1) * w[k] * p;
    }
  }
  const size_t n = ncut;
  std::vector<std::vector<double>> K(nl, std::vector<double>(n * n));
#pragma omp parallel for schedule(dynamic, 8)
  for (int i = 0; i < ncut; ++i)
    for (int j = 0; j <= i; ++j) {
      const double ri = grid.r[i], rj = grid.r[j];
      double acc[16] = {0};
      for (int k = 0; k < nx; ++k) {
        const double d = std::sqrt(std::max(ri * ri + rj * rj - 2.0 * ri * rj * x[k], 0.0)), t = omega * d;
        const double kern = t < 1e-4 ? omega * (2.0 / std::sqrt(pi)) * (1.0 - t * t / 3.0) : std::erf(t) / d;
        for (int l = 0; l < nl; ++l) acc[l] += pl[static_cast<size_t>(l) * nx + k] * kern;
      }
      for (int l = 0; l < nl; ++l) K[l][i * n + j] = K[l][j * n + i] = acc[l];
    }
  auto allowed = [&](int q, int l) {
    int j1 = 0, j2 = 0;
    for (int a = 0; a < nj; ++a)
      for (int b = a; b < nj; ++b)
        if (pair(a, b) == q) j1 = a, j2 = b;
    return l >= std::abs(l_j[j1] - l_j[j2]) && l <= l_j[j1] + l_j[j2] && (l + l_j[j1] + l_j[j2]) % 2 == 0;
  };
  std::vector<RMatrix> R = radial_coulomb_;
  for (int l = 0; l < nl; ++l) {
    // weighted densities and their long-range potentials v(r) = 4 pi/(2l+1) int K_l(r, r') f(r') r'^2 dr'
    std::vector<int> qs;
    for (int q = 0; q < nq_; ++q)
      if (allowed(q, l)) qs.push_back(q);
    const int m = static_cast<int>(qs.size());
    std::vector<std::vector<double>> f(2 * m, std::vector<double>(n)), v(2 * m, std::vector<double>(n));
    for (int a = 0; a < m; ++a)
      for (size_t i = 0; i < n; ++i) {
        f[a][i] = wr2_[i] * nq_ae_[qs[a]][i];
        f[m + a][i] = wr2_[i] * (nq_ps_[qs[a]][i] + delta_lq_[l][qs[a]] * shape_[l][i]);
      }
    const double pref = 4 * pi / (2 * l + 1);
#pragma omp parallel for schedule(static)
    for (int a = 0; a < 2 * m; ++a)
      for (size_t i = 0; i < n; ++i) {
        const double* k = &K[l][i * n];
        double sum = 0.0;
        for (size_t j = 0; j < n; ++j) sum += k[j] * f[a][j];
        v[a][i] = pref * sum;
      }
    for (int a = 0; a < m; ++a)
      for (int b = 0; b < m; ++b) {
        double ae = 0.0, ps = 0.0;
        for (size_t i = 0; i < n; ++i) {
          ae += f[a][i] * v[b][i];
          ps += f[m + a][i] * v[m + b][i];
        }
        R[l](qs[a], qs[b]) -= 0.5 * (ae - ps);
      }
    for (int q = 0; q < nq_; ++q)
      for (int p = 0; p < q; ++p) R[l](q, p) = R[l](p, q) = 0.5 * (R[l](q, p) + R[l](p, q));
  }
  Cx = assemble_coulomb(R);
}

void PawSetup::multipoles(const RMatrix& Dtot, double* Q) const {
  for (int L = 0; L < nL; ++L) {
    double sum = L == 0 ? Delta0 : 0.0;
    const double* d = &Delta_Lij[static_cast<size_t>(L) * ni * ni];
    for (int ij = 0; ij < ni * ni; ++ij) sum += d[ij] * Dtot.data()[ij];
    Q[L] = sum;
  }
}

RMatrix PawSetup::occupation_matrix(const RMatrix& D) const {
  LSDA_CHECK(hubbard_l >= 0, "occupation_matrix: no Hubbard shell defined");
  const int nm = 2 * hubbard_l + 1, l2 = hubbard_l * hubbard_l;
  RMatrix n(nm, nm);
  for (int i = 0; i < ni; ++i) {
    if (l_i[i] != hubbard_l) continue;
    for (int j = 0; j < ni; ++j)
      if (l_i[j] == hubbard_l) n(L_i[i] - l2, L_i[j] - l2) += D(i, j) * ae_norm_q_[pair(j_i[i], j_i[j])];
  }
  return n;
}

RMatrix PawSetup::atomic_density_matrix() const {
  RMatrix D(ni, ni);
  for (int i = 0; i < ni; ++i) D(i, i) = f_j[j_i[i]] / (2 * l_i[i] + 1);
  return D;
}

double PawSetup::onsite(int nspin, const RMatrix* D, RMatrix* dH, const XCFunctional& xc) const {
  const int n2 = ni * ni;
  // ---- kinetic + Coulomb + zero-potential part (depends on the spin-summed D) ----
  std::vector<double> dt(n2), cd(n2, 0.0);
  // nspin = 4: non-collinear, D[0] is the charge component and D[1..3] the magnetization components
  LSDA_CHECK(nspin == 1 || nspin == 2 || nspin == 4, "onsite: nspin must be 1, 2 or 4");
  for (int ij = 0; ij < n2; ++ij) dt[ij] = D[0].data()[ij] + (nspin == 2 ? D[1].data()[ij] : 0.0);
  for (int kl = 0; kl < n2; ++kl) {
    const double d = dt[kl];
    if (d == 0.0) continue;
    const double* c = C.col(kl);
    for (int ij = 0; ij < n2; ++ij) cd[ij] += c[ij] * d;
  }
  double energy = E0;
  for (int ij = 0; ij < n2; ++ij) energy += dt[ij] * (B.data()[ij] + cd[ij]);
  if (dH)
    for (int s = 0; s < nspin; ++s) {
      dH[s].resize(ni, ni);
      for (int ij = 0; ij < n2; ++ij) dH[s].data()[ij] = (nspin == 4 && s > 0) ? 0.0 : B.data()[ij] + 2.0 * cd[ij];
    }

  // ---- exact exchange (hybrid functionals): valence-valence, valence-core and core-core ------------
  if (const double ax = xc.exx_fraction(); ax != 0.0) {
    LSDA_CHECK(nspin != 4, "hybrid functionals are not implemented for non-collinear magnetism");
    LSDA_CHECK(has_exx_data, "PAW dataset of " + symbol + " contains no exact-exchange data");
    LSDA_CHECK(exx_omega == xc.exx_screening(), "onsite: prepare_exchange was not called for this functional");
    // per spin channel: D^s (nspin = 1: both channels hold D/2)
    const double per_spin = nspin == 2 ? 1.0 : 0.5, channels = nspin == 2 ? 1.0 : 2.0;
    energy += ax * exx_core;
    std::vector<double> kx(n2);
    for (int s = 0; s < nspin; ++s) {
      const double* d = D[s].data();
      // kx_ik = sum_jl Cx_(ij),(kl) D_jl
      std::fill(kx.begin(), kx.end(), 0.0);
      for (int k = 0; k < ni; ++k)
        for (int l = 0; l < ni; ++l) {
          const double* c = Cx.col(k * ni + l);
          for (int j = 0; j < ni; ++j) {
            const double djl = per_spin * d[j * ni + l];
            if (djl == 0.0) continue;
            for (int i = 0; i < ni; ++i) kx[i * ni + k] += c[i * ni + j] * djl;
          }
        }
      double e = 0.0;
      for (int ij = 0; ij < n2; ++ij) e += per_spin * d[ij] * (kx[ij] + X.data()[ij]);
      energy -= ax * channels * e;
      if (dH)
        for (int i = 0; i < ni; ++i)
          for (int k = 0; k < ni; ++k)
            dH[s](i, k) -= ax * (kx[i * ni + k] + kx[k * ni + i] + X(i, k));
    }
  }

  // ---- Hubbard correction ------------------------------------------------------------
  if (hubbard_l >= 0 && hubbard_u != 0.0 && nspin == 4) {
    // non-collinear: occupation matrix n = (n0 + sum_c n_c sigma_c)/2 in spin space,
    // E_U = U/2 Tr[n - n n] = U/2 [Tr n0 - (Tr n0^2 + sum_c Tr n_c^2)/2]
    const int nm = 2 * hubbard_l + 1, l2 = hubbard_l * hubbard_l;
    for (int c = 0; c < 4; ++c) {
      const RMatrix n = occupation_matrix(D[c]);
      double tr = 0.0, tr2 = 0.0;
      for (int a = 0; a < nm; ++a) {
        tr += n(a, a);
        for (int b = 0; b < nm; ++b) tr2 += n(a, b) * n(b, a);
      }
      energy += 0.5 * hubbard_u * ((c == 0 ? tr : 0.0) - 0.5 * tr2);
      if (dH)
        for (int i = 0; i < ni; ++i) {
          if (l_i[i] != hubbard_l) continue;
          for (int j = 0; j < ni; ++j) {
            if (l_i[j] != hubbard_l) continue;
            const int a = L_i[i] - l2, b = L_i[j] - l2;
            dH[c](i, j) += hubbard_u * ((c == 0 && a == b ? 0.5 : 0.0) - 0.5 * n(a, b)) * ae_norm_q_[pair(j_i[i], j_i[j])];
          }
        }
    }
  } else if (hubbard_l >= 0 && hubbard_u != 0.0) {
    const int nm = 2 * hubbard_l + 1, l2 = hubbard_l * hubbard_l;
    // occupations per spin channel: for nspin = 1 both channels hold half of the total
    const double per_spin = nspin == 2 ? 1.0 : 0.5, channels = nspin == 2 ? 1.0 : 2.0;
    for (int s = 0; s < nspin; ++s) {
      RMatrix n = occupation_matrix(D[s]);
      double tr = 0.0, tr2 = 0.0;
      for (int a = 0; a < nm; ++a) {
        tr += per_spin * n(a, a);
        for (int b = 0; b < nm; ++b) tr2 += per_spin * per_spin * n(a, b) * n(b, a);
      }
      energy += channels * 0.5 * hubbard_u * (tr - tr2);
      if (dH)  // dE/dD_ij = U (1/2 delta_mm' - n^s_mm') <phi_i|phi_j>
        for (int i = 0; i < ni; ++i) {
          if (l_i[i] != hubbard_l) continue;
          for (int j = 0; j < ni; ++j) {
            if (l_i[j] != hubbard_l) continue;
            const int a = L_i[i] - l2, b = L_i[j] - l2;
            dH[s](i, j) += hubbard_u * ((a == b ? 0.5 : 0.0) - per_spin * n(a, b)) * ae_norm_q_[pair(j_i[i], j_i[j])];
          }
        }
    }
  }

  // ---- exchange-correlation correction: angular quadrature x radial integration ---
  const int npts = static_cast<int>(quad_.points.size());
  const double inv_s4pi = 1.0 / std::sqrt(4 * pi);
  // D_Lq^s = sum_ij G(L,Li,Lj) D^s_ij
  std::vector<RMatrix> dlq(nspin);
  for (int s = 0; s < nspin; ++s) {
    dlq[s].resize(nL, nq_);
    for (int ij = 0; ij < n2; ++ij) {
      const double d = D[s].data()[ij];
      if (d == 0.0) continue;
      for (const GauntEntry& e : gaunt_ij_[ij]) dlq[s](e.L, e.q) += e.g * d;
    }
  }
  std::vector<RMatrix> vlq(nspin);  // sum_k w_k Y_L(k) int r^2 [v_ae n_q - v_ps n~_q] dr
  for (int s = 0; s < nspin; ++s) vlq[s].resize(nL, nq_);
  double exc_total = 0.0;
  if (nspin == 4) {
    // non-collinear LSDA: at every point the functional is evaluated in the local spin frame,
    // n_up/down = (n +- |m|)/2;  dE/dn = (v_up + v_dn)/2,  dE/dm_c = (v_up - v_dn)/2 m_c/|m|
    LSDA_CHECK(!xc.is_gga(), "non-collinear magnetism is implemented for the LSDA only");
    const int npts = static_cast<int>(quad_.points.size());
    const double inv_s4pi = 1.0 / std::sqrt(4 * pi);
    std::vector<double> dens[4], na(ncut), nb(ncut), e(ncut), va(ncut), vb(ncut), pot(ncut), cq(4 * nq_);
    for (auto& d : dens) d.resize(ncut);
    for (int k = 0; k < npts; ++k) {
      const double* y = ylm_quad_.col(k);
      for (int c = 0; c < 4; ++c)
        for (int q = 0; q < nq_; ++q) {
          double sum = 0.0;
          for (int L = 0; L < nL; ++L) sum += y[L] * dlq[c](L, q);
          cq[c * nq_ + q] = sum;
        }
      for (int pass = 0; pass < 2; ++pass) {  // 0: all-electron, 1: pseudo
        const auto& nq = pass == 0 ? nq_ae_ : nq_ps_;
        const std::vector<double>& core = pass == 0 ? nc : nct;
        const double sign = pass == 0 ? 1.0 : -1.0;
        for (int c = 0; c < 4; ++c) {
          for (int i = 0; i < ncut; ++i) dens[c][i] = c == 0 ? inv_s4pi * core[i] : 0.0;
          for (int q = 0; q < nq_; ++q) {
            const double f = cq[c * nq_ + q];
            if (f == 0.0) continue;
            const double* g = nq[q].data();
            for (int i = 0; i < ncut; ++i) dens[c][i] += f * g[i];
          }
        }
        for (int i = 0; i < ncut; ++i) {
          const double m = std::sqrt(dens[1][i] * dens[1][i] + dens[2][i] * dens[2][i] + dens[3][i] * dens[3][i]);
          na[i] = 0.5 * (dens[0][i] + m);
          nb[i] = 0.5 * (dens[0][i] - m);
        }
        xc.evaluate(ncut, na.data(), nb.data(), e.data(), va.data(), vb.data());
        double esum = 0.0;
        for (int i = 0; i < ncut; ++i) esum += wr2_[i] * e[i];
        exc_total += sign * quad_.weights[k] * esum;
        if (!dH) continue;
        for (int c = 0; c < 4; ++c) {
          for (int i = 0; i < ncut; ++i) {
            if (c == 0) {
              pot[i] = wr2_[i] * 0.5 * (va[i] + vb[i]);
            } else {
              const double m = na[i] - nb[i];
              pot[i] = m > 1e-14 ? wr2_[i] * 0.5 * (va[i] - vb[i]) * dens[c][i] / m : 0.0;
            }
          }
          for (int q = 0; q < nq_; ++q) {
            const double* g = nq[q].data();
            double sum = 0.0;
            for (int i = 0; i < ncut; ++i) sum += pot[i] * g[i];
            const double w = sign * quad_.weights[k] * sum;
            for (int L = 0; L < nL; ++L) vlq[c](L, q) += y[L] * w;
          }
        }
      }
    }
    energy += exc_total;
    if (dH)
      for (int c = 0; c < 4; ++c)
        for (int ij = 0; ij < n2; ++ij) {
          double sum = 0.0;
          for (const GauntEntry& ge : gaunt_ij_[ij]) sum += ge.g * vlq[c](ge.L, ge.q);
          dH[c].data()[ij] += sum;
        }
    return energy;
  }
  if (xc.is_gga()) {
    energy += onsite_xc_gga(nspin, dlq, dH ? &vlq : nullptr, xc);
    if (dH)
      for (int s = 0; s < nspin; ++s)
        for (int ij = 0; ij < n2; ++ij) {
          double sum = 0.0;
          for (const GauntEntry& ge : gaunt_ij_[ij]) sum += ge.g * vlq[s](ge.L, ge.q);
          dH[s].data()[ij] += sum;
        }
    return energy;
  }
  std::vector<double> na(ncut), nb(ncut), e(ncut), va(ncut), vb(ncut), cq(nq_ * 2);
  for (int k = 0; k < npts; ++k) {
    const double* y = ylm_quad_.col(k);
    for (int s = 0; s < nspin; ++s)
      for (int q = 0; q < nq_; ++q) {
        double sum = 0.0;
        for (int L = 0; L < nL; ++L) sum += y[L] * dlq[s](L, q);
        cq[s * nq_ + q] = sum;
      }
    for (int pass = 0; pass < 2; ++pass) {  // 0: all-electron, 1: pseudo
      const auto& nq = pass == 0 ? nq_ae_ : nq_ps_;
      const std::vector<double>& core = pass == 0 ? nc : nct;
      const double sign = pass == 0 ? 1.0 : -1.0;
      const double spin_fac = nspin == 2 ? 1.0 : 0.5;
      for (int i = 0; i < ncut; ++i) na[i] = 0.5 * inv_s4pi * core[i];
      nb = na;
      for (int q = 0; q < nq_; ++q) {
        const double ca = spin_fac * cq[q], cb = nspin == 2 ? cq[nq_ + q] : ca;
        const double* f = nq[q].data();
        if (ca != 0.0)
          for (int i = 0; i < ncut; ++i) na[i] += ca * f[i];
        if (cb != 0.0)
          for (int i = 0; i < ncut; ++i) nb[i] += cb * f[i];
      }
      xc.evaluate(ncut, na.data(), nb.data(), e.data(), va.data(), vb.data());
      double esum = 0.0;
      for (int i = 0; i < ncut; ++i) esum += wr2_[i] * e[i];
      exc_total += sign * quad_.weights[k] * esum;
      if (!dH) continue;
      for (int i = 0; i < ncut; ++i) {
        va[i] *= wr2_[i];
        vb[i] *= wr2_[i];
      }
      for (int q = 0; q < nq_; ++q) {
        const double* f = nq[q].data();
        double ia = 0.0, ib = 0.0;
        for (int i = 0; i < ncut; ++i) {
          ia += va[i] * f[i];
          ib += vb[i] * f[i];
        }
        const double wa = sign * quad_.weights[k] * ia, wb = sign * quad_.weights[k] * ib;
        for (int L = 0; L < nL; ++L) {
          vlq[0](L, q) += y[L] * wa;
          if (nspin == 2) vlq[1](L, q) += y[L] * wb;
        }
      }
    }
  }
  energy += exc_total;
  if (dH)
    for (int s = 0; s < nspin; ++s)
      for (int ij = 0; ij < n2; ++ij) {
        double sum = 0.0;
        for (const GauntEntry& ge : gaunt_ij_[ij]) sum += ge.g * vlq[s](ge.L, ge.q);
        dH[s].data()[ij] += sum;
      }
  return energy;
}

// Gradient-corrected on-site xc energy E_xc[n_a] - E_xc[n~_a] and, if vlq is given, its derivative with
// respect to the multipole density-matrix elements D^s_Lq. With n_s(r, Omega) = sum_L n_sL(r) Y_L(Omega):
//   grad n_s = r^ dn_s/dr + (1/r) sum_L n_sL grad_Omega Y_L ,
//   dE/dD^s_Lq = sum_k w_k int r^2 dr { v_s Y_L n_q + h_s.r^ Y_L dn_q/dr + (1/r) h_s . grad_Omega Y_L n_q },
// h_s = de/d(grad n_s) = 2 e_ss grad n_s + e_ud grad n_s'.
double PawSetup::onsite_xc_gga(int nspin, const std::vector<RMatrix>& dlq, std::vector<RMatrix>* vlq,
                               const XCFunctional& xc) const {
  const int npts = static_cast<int>(quad_.points.size());
  const double inv_s4pi = 1.0 / std::sqrt(4 * pi), spin_fac = nspin == 2 ? 1.0 : 0.5;
  std::vector<double> inv_r2(ncut, 0.0);
  for (int i = 0; i < ncut; ++i)
    if (grid.r[i] > 0.0) inv_r2[i] = 1.0 / (grid.r[i] * grid.r[i]);
  double exc_total = 0.0;
  // per spin channel: density, radial derivative, tangential gradient (three components, without the 1/r)
  std::vector<double> n[2], dn[2], tg[2][3];
  for (int s = 0; s < 2; ++s) {
    n[s].resize(ncut);
    dn[s].resize(ncut);
    for (auto& t : tg[s]) t.resize(ncut);
  }
  std::vector<double> suu(ncut), sud(ncut), sdd(ncut), e(ncut), v[2], vss[2], vud(ncut), hr(ncut), ht(ncut);
  for (int s = 0; s < 2; ++s) {
    v[s].resize(ncut);
    vss[s].resize(ncut);
  }
  std::vector<double> c(2 * nq_), tc(2 * 3 * nq_);
  // meta-GGA: kinetic energy densities tau_s(r) = core/2 + sum_{l,q} ct(s, l, q) tq[l][q](r)
  const bool mgga = xc.is_mgga();
  const int nl = 2 * lmax + 1;
  LSDA_CHECK(!mgga || has_core_tau, "onsite: prepare_mgga was not called");
  std::vector<double> tau[2], vt[2], ct(mgga ? 2 * nl * nq_ : 0), i2(mgga ? nl * nq_ : 0);
  std::vector<int> l_of(nL);
  for (int L = 0; L < nL; ++L) l_of[L] = static_cast<int>(std::floor(std::sqrt(static_cast<double>(L)) + 1e-9));
  if (mgga)
    for (int s = 0; s < 2; ++s) {
      tau[s].resize(ncut);
      vt[s].resize(ncut);
    }
  for (int k = 0; k < npts; ++k) {
    const double* y = ylm_quad_.col(k);
    const double* gy = &ylm_grad_quad_[3 * static_cast<size_t>(nL) * k];
    for (int s = 0; s < nspin; ++s)
      for (int q = 0; q < nq_; ++q) {
        double sum = 0.0, t[3] = {0, 0, 0};
        for (int L = 0; L < nL; ++L) {
          const double d = dlq[s](L, q);
          sum += y[L] * d;
          for (int a = 0; a < 3; ++a) t[a] += gy[3 * L + a] * d;
        }
        c[s * nq_ + q] = spin_fac * sum;
        for (int a = 0; a < 3; ++a) tc[(s * 3 + a) * nq_ + q] = spin_fac * t[a];
        if (mgga) {
          for (int l = 0; l < nl; ++l) ct[(s * nl + l) * nq_ + q] = 0.0;
          for (int L = 0; L < nL; ++L) ct[(s * nl + l_of[L]) * nq_ + q] += spin_fac * y[L] * dlq[s](L, q);
        }
      }
    for (int pass = 0; pass < 2; ++pass) {  // 0: all-electron, 1: pseudo
      const auto& nq = pass == 0 ? nq_ae_ : nq_ps_;
      const auto& dnq = pass == 0 ? dnq_ae_ : dnq_ps_;
      const std::vector<double>& core = pass == 0 ? nc : nct;
      const std::vector<double>& dcore = pass == 0 ? dnc_ : dnct_;
      const double sign = pass == 0 ? 1.0 : -1.0;
      for (int s = 0; s < nspin; ++s) {
        for (int i = 0; i < ncut; ++i) {
          n[s][i] = 0.5 * inv_s4pi * core[i];
          dn[s][i] = 0.5 * inv_s4pi * dcore[i];
        }
        for (auto& t : tg[s]) std::fill(t.begin(), t.end(), 0.0);
        for (int q = 0; q < nq_; ++q) {
          const double cq = c[s * nq_ + q];
          const double *f = nq[q].data(), *df = dnq[q].data();
          if (cq != 0.0)
            for (int i = 0; i < ncut; ++i) {
              n[s][i] += cq * f[i];
              dn[s][i] += cq * df[i];
            }
          for (int a = 0; a < 3; ++a) {
            const double ta = tc[(s * 3 + a) * nq_ + q];
            if (ta == 0.0) continue;
            double* t = tg[s][a].data();
            for (int i = 0; i < ncut; ++i) t[i] += ta * f[i];
          }
        }
      }
      const int o = nspin == 2 ? 1 : 0;  // index of the second spin channel (identical to the first for nspin = 1)
      if (mgga) {
        const auto& tq = pass == 0 ? tq_ae_ : tq_ps_;
        const std::vector<double>& tcore = pass == 0 ? tauc : tauct;
        for (int s = 0; s < nspin; ++s) {
          for (int i = 0; i < ncut; ++i) tau[s][i] = 0.5 * inv_s4pi * tcore[i];
          for (int l = 0; l < nl; ++l)
            for (int q = 0; q < nq_; ++q) {
              const double cq = ct[(s * nl + l) * nq_ + q];
              if (cq == 0.0) continue;
              const double* t = tq[l][q].data();
              for (int i = 0; i < ncut; ++i) tau[s][i] += cq * t[i];
            }
        }
      }
      for (int i = 0; i < ncut; ++i) {
        double tuu = 0.0, tud = 0.0, tdd = 0.0;
        for (int a = 0; a < 3; ++a) {
          tuu += tg[0][a][i] * tg[0][a][i];
          tud += tg[0][a][i] * tg[o][a][i];
          tdd += tg[o][a][i] * tg[o][a][i];
        }
        suu[i] = dn[0][i] * dn[0][i] + tuu * inv_r2[i];
        sud[i] = dn[0][i] * dn[o][i] + tud * inv_r2[i];
        sdd[i] = dn[o][i] * dn[o][i] + tdd * inv_r2[i];
      }
      if (mgga)
        xc.evaluate_mgga(ncut, n[0].data(), n[o].data(), suu.data(), sud.data(), sdd.data(), tau[0].data(), tau[o].data(), e.data(),
                         v[0].data(), v[1].data(), vss[0].data(), vud.data(), vss[1].data(), vt[0].data(), vt[1].data());
      else
        xc.evaluate_gga(ncut, n[0].data(), n[o].data(), suu.data(), sud.data(), sdd.data(), e.data(), v[0].data(), v[1].data(),
                        vss[0].data(), vud.data(), vss[1].data());
      double esum = 0.0;
      for (int i = 0; i < ncut; ++i) esum += wr2_[i] * e[i];
      exc_total += sign * quad_.weights[k] * esum;
      if (!vlq) continue;
      const double wk = sign * quad_.weights[k];
      for (int s = 0; s < nspin; ++s) {
        const int s2 = nspin == 2 ? 1 - s : 0;
        // radial part of h_s and the weighted local potential
        for (int i = 0; i < ncut; ++i) hr[i] = wr2_[i] * (2.0 * vss[s][i] * dn[s][i] + vud[i] * dn[s2][i]);
        std::vector<double> i1(nq_, 0.0);
        for (int q = 0; q < nq_; ++q) {
          const double *f = nq[q].data(), *df = dnq[q].data();
          double sum = 0.0;
          for (int i = 0; i < ncut; ++i) sum += wr2_[i] * v[s][i] * f[i] + hr[i] * df[i];
          i1[q] = sum;
        }
        if (mgga) {  // int r^2 v_tau tq[l][q] dr
          const auto& tq = pass == 0 ? tq_ae_ : tq_ps_;
          for (int l = 0; l < nl; ++l)
            for (int q = 0; q < nq_; ++q) {
              const double* t = tq[l][q].data();
              double sum = 0.0;
              for (int i = 0; i < ncut; ++i) sum += wr2_[i] * vt[s][i] * t[i];
              i2[l * nq_ + q] = sum;
            }
        }
        for (int L = 0; L < nL; ++L) {
          const double f = wk * y[L];
          if (f == 0.0) continue;
          for (int q = 0; q < nq_; ++q) (*vlq)[s](L, q) += f * (i1[q] + (mgga ? i2[l_of[L] * nq_ + q] : 0.0));
        }
        // tangential part: int r^2 (1/r^2) (2 e_ss T_s + e_ud T_s') . grad_Omega Y_L n_q dr
        for (int a = 0; a < 3; ++a) {
          for (int i = 0; i < ncut; ++i)
            ht[i] = wr2_[i] * inv_r2[i] * (2.0 * vss[s][i] * tg[s][a][i] + vud[i] * tg[s2][a][i]);
          for (int q = 0; q < nq_; ++q) {
            const double* f = nq[q].data();
            double sum = 0.0;
            for (int i = 0; i < ncut; ++i) sum += ht[i] * f[i];
            if (sum == 0.0) continue;
            for (int L = 0; L < nL; ++L) (*vlq)[s](L, q) += wk * gy[3 * L + a] * sum;
          }
        }
      }
    }
  }
  return exc_total;
}

void PawSetup::build_tables(double qmax_wfc, double qmax_dense) {
  const double dq = 0.01;
  const int nfull = grid.size();
  const double s4pi = std::sqrt(4 * pi);
  auto extent = [&](const std::vector<double>& f) {  // last significant point (+ margin)
    double fmax = 0.0;
    for (double v : f) fmax = std::max(fmax, std::abs(v));
    int last = nfull - 1;
    while (last > 8 && std::abs(f[last]) < 1e-13 * fmax) --last;
    return std::min(nfull, last + 2);
  };
  proj_q.resize(nj);
  phit_q.assign(nj, RadialTable{});
  for (int j = 0; j < nj; ++j) {
    proj_q[j] = bessel_transform(grid, proj[j], extent(proj[j]), l_j[j], qmax_wfc, dq);
    if (is_bound(j)) phit_q[j] = bessel_transform(grid, phit[j], extent(phit[j]), l_j[j], qmax_wfc, dq, 2);
  }
  g_q.resize(2 * lmax + 1);
  for (int l = 0; l <= 2 * lmax; ++l) g_q[l] = bessel_transform(grid, shape_[l], ncut, l, qmax_dense, dq);
  auto spherical = [&](const std::vector<double>& f) {
    RadialTable t = bessel_transform(grid, f, extent(f), 0, qmax_dense, dq);
    for (double& v : t.f) v *= s4pi;  // 4 pi Y_00
    return t;
  };
  nct_q = spherical(nct);
  vbar_q = spherical(vbar);
  if (!nvt.empty()) nvt_q = spherical(nvt);
  if (has_core_tau) tauct_q = spherical(tauct);
}

}  // namespace lsda
