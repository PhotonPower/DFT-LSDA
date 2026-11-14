#include "dft/phonons.hpp"

#include "base/parallel.hpp"

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>

namespace lsda {

double atomic_mass(int Z) {
  static const double m[] = {0,       1.008,  4.0026, 6.94,   9.0122, 10.81,  12.011, 14.007, 15.999, 18.998, 20.180, 22.990, 24.305,
                             26.982,  28.085, 30.974, 32.06,  35.45,  39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938,
                             55.845,  58.933, 58.693, 63.546, 65.38,  69.723, 72.630, 74.922, 78.971, 79.904, 83.798, 85.468, 87.62,
                             88.906,  91.224, 92.906, 95.95,  98.0,   101.07, 102.91, 106.42, 107.87, 112.41, 114.82, 118.71, 121.76,
                             127.60,  126.90, 131.29, 132.91, 137.33, 138.91, 140.12, 140.91, 144.24, 145.0,  150.36, 151.96, 157.25,
                             158.93,  162.50, 164.93, 167.26, 168.93, 173.05, 174.97, 178.49, 180.95, 183.84, 186.21, 190.23, 192.22,
                             195.08,  196.97, 200.59, 204.38, 207.2,  208.98};
  LSDA_CHECK(Z >= 1 && Z < static_cast<int>(sizeof m / sizeof m[0]), "no atomic mass tabulated for this element (give it in the species block)");
  return m[Z];
}

PhononResult run_phonons(const Crystal& crystal, const Params& params, const PhononParams& pp, const std::vector<double>& masses) {
  const int nat = crystal.natoms(), n3 = 3 * nat;
  LSDA_CHECK(static_cast<int>(masses.size()) == nat, "phonons: one mass per atom is required");
  LSDA_CHECK(pp.displacement > 0.0, "phonons: displacement must be positive");
  PhononResult res;
  const std::string chk = pp.prefix + ".chk", store = pp.prefix + ".phonons.forces";

  // ---- reference structure: density for the start of the displaced runs, symmetry, residual forces ----
  Calculation ref(crystal, params);
  if (pp.restart) ref.load_checkpoint(chk);
  ref.scf();
  ref.save_checkpoint(chk);
  if (ref.stopped_by_time()) {
    res.stopped = true;
    return res;
  }
  res.energy = ref.energies().free;
  for (const Vec3& f : ref.forces()) res.max_force_reference = std::max(res.max_force_reference, norm(f));
  const std::vector<SymOp> ops = ref.symmetry_ops();
  logf("\n  Phonons: reference structure has max |F| = %.6f Ha/bohr%s\n", res.max_force_reference,
       res.max_force_reference > 1e-3 ? "  (WARNING: not relaxed)" : "");

  // orbits of atoms: representative (smallest index) and an operation that maps it onto each atom
  std::vector<int> rep(nat, -1), op_from_rep(nat, -1);
  for (int a = 0; a < nat; ++a) {
    if (rep[a] >= 0) continue;
    for (size_t io = 0; io < ops.size(); ++io) {
      const int b = ops[io].atom_map[a];
      if (rep[b] < 0) {
        rep[b] = a;
        op_from_rep[b] = static_cast<int>(io);
      }
    }
  }
  int nrep = 0;
  for (int a = 0; a < nat; ++a) nrep += rep[a] == a;
  logf("  Phonons: %d atoms, %d symmetry-inequivalent -> %d displaced calculations (h = %.4f bohr)\n", nat, nrep, 6 * nrep,
       pp.displacement);

  // ---- forces of displaced structures (stored on disk as they are completed) ---------------------------
  std::map<std::string, std::vector<double>> done;  // key "atom dir sign" -> 3N forces
  if (pp.restart) {
    std::ifstream in(store);
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream ls(line);
      int a, d, sg;
      double h;
      if (!(ls >> a >> d >> sg >> h) || std::abs(h - pp.displacement) > 1e-12) continue;
      std::vector<double> f(n3);
      bool ok = true;
      for (double& v : f) ok = ok && static_cast<bool>(ls >> v);
      if (ok) done[strf("%d %d %d", a, d, sg)] = f;
    }
    if (!done.empty()) logf("  Phonons: %zu displacements read from %s\n", done.size(), store.c_str());
  } else if (par::root()) {
    std::ofstream(store, std::ios::trunc);
  }
  Params quiet = params;
  quiet.verbosity = 0;
  RMatrix phi(n3, n3);  // phi(3b+beta, 3a+alpha) = d^2 E / d u_b,beta d u_a,alpha
  for (int a = 0; a < nat; ++a) {
    if (rep[a] != a) continue;
    for (int d = 0; d < 3; ++d) {
      std::vector<double> fpm[2];
      for (int sg = 0; sg < 2; ++sg) {
        const std::string key = strf("%d %d %d", a, d, sg);
        auto it = done.find(key);
        if (it == done.end()) {
          Crystal c = crystal;
          Vec3 r = c.atom_cart(a);
          r[d] += (sg == 0 ? 1.0 : -1.0) * pp.displacement;
          c.atoms[a].frac = c.frac_from_cart(r);
          Calculation calc(std::move(c), quiet);
          calc.adopt_density(ref);
          const bool ok = calc.scf();
          if (calc.stopped_by_time()) {
            res.stopped = true;
            return res;
          }
          if (!ok) logf("  WARNING: SCF not converged for the displacement of atom %d\n", a + 1);
          const std::vector<Vec3> f = calc.forces();
          std::vector<double> flat(n3);
          for (int b = 0; b < nat; ++b)
            for (int k = 0; k < 3; ++k) flat[3 * b + k] = f[b][k];
          if (par::root()) {
            std::ofstream out(store, std::ios::app);
            out.precision(17);
            out << a << " " << d << " " << sg << " " << pp.displacement;
            for (double v : flat) out << " " << v;
            out << "\n";
          }
          it = done.emplace(key, flat).first;
          ++res.n_displacements;
          logf("  displacement atom %3d %c%c: %d SCF iterations\n", a + 1, sg == 0 ? '+' : '-', "xyz"[d], calc.n_scf_iterations());
        }
        fpm[sg] = it->second;
      }
      for (int i = 0; i < n3; ++i) phi(i, 3 * a + d) = -(fpm[0][i] - fpm[1][i]) / (2.0 * pp.displacement);
    }
  }
  // ---- remaining columns by symmetry: Phi(g(b), g(a)) = R Phi(b, a) R^T ---------------------------------
  for (int a = 0; a < nat; ++a) {
    if (rep[a] == a) continue;
    const SymOp& op = ops[op_from_rep[a]];
    const int a0 = rep[a];
    for (int b0 = 0; b0 < nat; ++b0) {
      const int b = op.atom_map[b0];
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          double sum = 0.0;
          for (int k = 0; k < 3; ++k)
            for (int l = 0; l < 3; ++l) sum += op.R[i][k] * phi(3 * b0 + k, 3 * a0 + l) * op.R[j][l];
          phi(3 * b + i, 3 * a + j) = sum;
        }
    }
  }
  // ---- diagnostics, symmetrization, acoustic sum rule --------------------------------------------------
  for (int i = 0; i < n3; ++i)
    for (int j = 0; j < i; ++j) res.asymmetry = std::max(res.asymmetry, std::abs(phi(i, j) - phi(j, i)));
  for (int a = 0; a < nat; ++a)
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        double sum = 0.0;
        for (int b = 0; b < nat; ++b) sum += phi(3 * b + i, 3 * a + j);
        res.asr_violation = std::max(res.asr_violation, std::abs(sum));
      }
  auto symmetrize = [&]() {
    for (int i = 0; i < n3; ++i)
      for (int j = 0; j < i; ++j) phi(i, j) = phi(j, i) = 0.5 * (phi(i, j) + phi(j, i));
  };
  symmetrize();
  for (int a = 0; a < nat; ++a)  // translation invariance: Phi(a, a) = - sum_{b != a} Phi(b, a)
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        double sum = 0.0;
        for (int b = 0; b < nat; ++b)
          if (b != a) sum += phi(3 * b + i, 3 * a + j);
        phi(3 * a + i, 3 * a + j) = -sum;
      }
  symmetrize();
  res.force_constants = phi;

  // ---- dynamical matrix and frequencies -------------------------------------------------------------------
  const double amu = 1822.888486209;        // electron masses
  const double ha_to_thz = 6579.683920502;  // nu = E/h for E = 1 Ha
  RMatrix dyn(n3, n3);
  for (int i = 0; i < n3; ++i)
    for (int j = 0; j < n3; ++j) dyn(i, j) = phi(i, j) / (amu * std::sqrt(masses[i / 3] * masses[j / 3]));
  std::vector<double> lam;
  eigh_real(dyn, lam);
  res.frequency_thz.resize(n3);
  for (int i = 0; i < n3; ++i) res.frequency_thz[i] = (lam[i] < 0 ? -1.0 : 1.0) * std::sqrt(std::abs(lam[i])) * ha_to_thz;
  return res;
}

Crystal make_supercell(const Crystal& prim, const int n[3]) {
  LSDA_CHECK(n[0] >= 1 && n[1] >= 1 && n[2] >= 1, "supercell dimensions must be positive");
  Crystal c = prim;
  Mat3 a{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) a[i][j] = n[i] * prim.A[i][j];
  c.set_lattice(a);
  c.atoms.clear();
  c.sym_constraints.clear();
  for (int i = 0; i < n[0]; ++i)
    for (int j = 0; j < n[1]; ++j)
      for (int k = 0; k < n[2]; ++k)
        for (const Atom& at : prim.atoms) {
          Atom b = at;
          b.frac = {(at.frac[0] + i) / n[0], (at.frac[1] + j) / n[1], (at.frac[2] + k) / n[2]};
          c.atoms.push_back(b);
        }
  return c;
}

std::vector<double> phonon_frequencies(const Crystal& prim, const Crystal& super, const RMatrix& phi,
                                       const std::vector<double>& masses_prim, const Vec3& q) {
  const int np = prim.natoms(), ns = super.natoms(), n3 = 3 * np;
  LSDA_CHECK(ns % np == 0 && phi.rows() == 3 * ns, "phonon_frequencies: supercell does not match the primitive cell");
  Vec3 qc{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) qc[j] += q[i] * prim.B[i][j];
  CMatrix dyn(n3, n3);
  const double amu = 1822.888486209;
  for (int a = 0; a < np; ++a) {
    const Vec3 ra = super.atom_cart(a);
    for (int bs = 0; bs < ns; ++bs) {
      const int b = bs % np;
      const Vec3 rb = super.atom_cart(bs);
      // nearest periodic images of the supercell atom
      double dmin = 1e300;
      std::vector<Vec3> images;
      for (int i = -1; i <= 1; ++i)
        for (int j = -1; j <= 1; ++j)
          for (int k = -1; k <= 1; ++k) {
            Vec3 d;
            for (int x = 0; x < 3; ++x) d[x] = rb[x] - ra[x] + i * super.A[0][x] + j * super.A[1][x] + k * super.A[2][x];
            const double len = norm(d);
            if (len < dmin - 1e-6) {
              dmin = len;
              images.clear();
            }
            if (len < dmin + 1e-6) images.push_back(d);
          }
      cplx phase = 0.0;
      for (const Vec3& d : images) phase += std::exp(cplx(0.0, dot(qc, d)));
      phase /= static_cast<double>(images.size());
      const double f = 1.0 / (amu * std::sqrt(masses_prim[a] * masses_prim[b]));
      for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y) dyn(3 * a + x, 3 * b + y) += f * phi(3 * bs + y, 3 * a + x) * phase;
    }
  }
  for (int i = 0; i < n3; ++i)  // Hermitian part (removes rounding asymmetry)
    for (int j = 0; j <= i; ++j) {
      const cplx v = 0.5 * (dyn(i, j) + std::conj(dyn(j, i)));
      dyn(i, j) = v;
      dyn(j, i) = std::conj(v);
    }
  std::vector<double> lam;
  eigh(dyn, lam);
  for (double& v : lam) v = (v < 0 ? -1.0 : 1.0) * std::sqrt(std::abs(v)) * 6579.683920502;
  return lam;
}

void write_phonon_results(const std::string& path, const Crystal& c, const std::vector<double>& masses, const PhononResult& r) {
  if (!par::root()) return;
  std::FILE* f = std::fopen(path.c_str(), "w");
  LSDA_CHECK(f != nullptr, "cannot write '" + path + "'");
  std::fprintf(f, "{\n  \"n_atoms\": %d,\n  \"volume\": %.10f,\n  \"asr_violation\": %.6e,\n  \"asymmetry\": %.6e,\n", c.natoms(), c.volume,
               r.asr_violation, r.asymmetry);
  std::fprintf(f, "  \"max_force_reference\": %.6e,\n  \"masses_amu\": [", r.max_force_reference);
  for (size_t i = 0; i < masses.size(); ++i) std::fprintf(f, "%.5f%s", masses[i], i + 1 < masses.size() ? ", " : "],\n");
  std::fprintf(f, "  \"comment\": \"Gamma-point modes of the cell; negative values are imaginary frequencies\",\n");
  std::fprintf(f, "  \"frequencies_THz\": [");
  for (size_t i = 0; i < r.frequency_thz.size(); ++i)
    std::fprintf(f, "%.8f%s", r.frequency_thz[i], i + 1 < r.frequency_thz.size() ? ", " : "]\n}\n");
  std::fclose(f);
}

}  // namespace lsda
