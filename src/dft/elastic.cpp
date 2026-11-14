#include "dft/elastic.hpp"

#include <cmath>

#include "base/parallel.hpp"

namespace lsda {

namespace {

const int va[6] = {0, 1, 2, 1, 0, 0}, vb[6] = {0, 1, 2, 2, 2, 1};  // Voigt index -> tensor indices

Mat3 strain_pattern(int j) {  // unit Voigt strain j as a symmetric tensor (engineering shear: eps_ab = 1/2)
  Mat3 e{};
  if (j < 3)
    e[va[j]][vb[j]] = 1.0;
  else
    e[va[j]][vb[j]] = e[vb[j]][va[j]] = 0.5;
  return e;
}

Mat3 rotate(const Mat3& R, const Mat3& t) {  // R t R^T
  return matmul3(matmul3(R, t), transpose3(R));
}

}  // namespace

ElasticResult run_elastic(const Crystal& crystal, const Params& params, const RelaxParams& rp, double delta) {
  LSDA_CHECK(delta > 0.0 && delta < 0.05, "elastic: strain amplitude must be between 0 and 0.05");
  ElasticResult res;
  const int nat = crystal.natoms();
  auto run = [&](Calculation& calc) {  // returns false if interrupted by the wall-time limit
    if (nat > 1)
      relax(calc, rp);
    else
      calc.scf();
    return !calc.stopped_by_time();
  };

  // ---- reference cell ---------------------------------------------------------------------------
  Calculation ref(crystal, params);
  if (!run(ref)) {
    res.stopped = true;
    return res;
  }
  res.stress0 = ref.stress();
  const std::vector<SymOp> ops = ref.symmetry_ops();
  const Crystal base = ref.crystal();  // with relaxed positions
  const double gpa = 29421.02648438959;
  logf("\n  Elastic constants: residual stress of the reference cell (GPa): %.4f %.4f %.4f %.4f %.4f %.4f\n", res.stress0[0][0] * gpa,
       res.stress0[1][1] * gpa, res.stress0[2][2] * gpa, res.stress0[1][2] * gpa, res.stress0[0][2] * gpa, res.stress0[0][1] * gpa);

  Params quiet = params;
  quiet.verbosity = 0;
  std::vector<Mat3> dsig(6);          // d sigma / d eps_j as tensors
  std::vector<char> have(6, 0);
  for (int j = 0; j < 6; ++j) {
    const Mat3 ej = strain_pattern(j);
    // related to an already computed pattern by symmetry?  E_j = R E_k R^T  ->  dsigma_j = R dsigma_k R^T
    for (int k = 0; k < j && !have[j]; ++k)
      for (const SymOp& op : ops) {
        const Mat3 ek = rotate(op.R, strain_pattern(k));
        double dev = 0.0;
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) dev = std::max(dev, std::abs(ek[a][b] - ej[a][b]));
        if (dev < 1e-8) {
          dsig[j] = rotate(op.R, dsig[k]);
          have[j] = 1;
          logf("  strain %d: obtained from strain %d by symmetry\n", j + 1, k + 1);
          break;
        }
      }
    if (have[j]) continue;
    Mat3 sig[2];
    for (int sg = 0; sg < 2; ++sg) {
      const double f = (sg == 0 ? 1.0 : -1.0) * delta;
      Crystal c = base;
      Mat3 a_new{};
      for (int i = 0; i < 3; ++i)
        for (int a = 0; a < 3; ++a) {
          a_new[i][a] = base.A[i][a];
          for (int b = 0; b < 3; ++b) a_new[i][a] += f * ej[a][b] * base.A[i][b];
        }
      c.set_lattice(a_new);
      Calculation calc(std::move(c), quiet);
      if (!run(calc)) {
        res.stopped = true;
        return res;
      }
      sig[sg] = calc.stress();
      ++res.n_strained;
    }
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) dsig[j][a][b] = (sig[0][a][b] - sig[1][a][b]) / (2.0 * delta);
    have[j] = 1;
    logf("  strain %d: two strained cells computed\n", j + 1);
  }
  res.C.resize(6, 6);
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j) res.C(i, j) = dsig[j][va[i]][vb[i]];
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < i; ++j) {
      res.asymmetry = std::max(res.asymmetry, std::abs(res.C(i, j) - res.C(j, i)));
      res.C(i, j) = res.C(j, i) = 0.5 * (res.C(i, j) + res.C(j, i));
    }
  return res;
}

void write_elastic_results(const std::string& path, const Crystal& c, const ElasticResult& r) {
  if (!par::root()) return;
  const double gpa = 29421.02648438959;
  std::FILE* f = std::fopen(path.c_str(), "w");
  LSDA_CHECK(f != nullptr, "cannot write '" + path + "'");
  const RMatrix& C = r.C;
  const double bulk = (C(0, 0) + C(1, 1) + C(2, 2) + 2.0 * (C(0, 1) + C(0, 2) + C(1, 2))) / 9.0;
  std::fprintf(f, "{\n  \"n_atoms\": %d,\n  \"volume\": %.10f,\n  \"bulk_modulus_GPa\": %.6f,\n  \"asymmetry_GPa\": %.6f,\n", c.natoms(), c.volume,
               bulk * gpa, r.asymmetry * gpa);
  std::fprintf(f, "  \"comment\": \"Voigt order xx yy zz yz xz xy, engineering shear strains; bulk modulus = Voigt average\",\n  \"C_GPa\": [\n");
  for (int i = 0; i < 6; ++i) {
    std::fprintf(f, "    [");
    for (int j = 0; j < 6; ++j) std::fprintf(f, "%.6f%s", C(i, j) * gpa, j < 5 ? ", " : "]");
    std::fprintf(f, "%s\n", i < 5 ? "," : "");
  }
  std::fprintf(f, "  ]\n}\n");
  std::fclose(f);
}

}  // namespace lsda
