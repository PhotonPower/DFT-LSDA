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

  s.quad_ = make_sphere_quadrature(ntheta);
  s.ylm_quad_.resize(s.nL, static_cast<int>(s.quad_.points.size()));
  for (size_t p = 0; p < s.quad_.points.size(); ++p)
    real_ylm(2 * s.lmax, s.quad_.points[p].data(), s.ylm_quad_.col(static_cast<int>(p)));

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
  C.resize(ni * ni, ni * ni);
  for (int ij = 0; ij < ni * ni; ++ij)
    for (int kl = 0; kl < ni * ni; ++kl) {
      double sum = 0.0;
      for (const GauntEntry& a : gaunt_ij_[ij])
        for (const GauntEntry& b : gaunt_ij_[kl])
          if (a.L == b.L) {
            const int l = static_cast<int>(std::floor(std::sqrt(static_cast<double>(a.L)) + 1e-9));
            sum += a.g * b.g * R[l](a.q, b.q);
          }
      C(ij, kl) = sum;
    }
}

void PawSetup::multipoles(const RMatrix& Dtot, double* Q) const {
  for (int L = 0; L < nL; ++L) {
    double sum = L == 0 ? Delta0 : 0.0;
    const double* d = &Delta_Lij[static_cast<size_t>(L) * ni * ni];
    for (int ij = 0; ij < ni * ni; ++ij) sum += d[ij] * Dtot.data()[ij];
    Q[L] = sum;
  }
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
      for (int ij = 0; ij < n2; ++ij) dH[s].data()[ij] = B.data()[ij] + 2.0 * cd[ij];
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
}

}  // namespace lsda
