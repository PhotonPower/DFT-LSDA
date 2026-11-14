#include <cmath>
#include <map>

#include "crystal/crystal.hpp"

namespace lsda {

namespace {

double wrap_diff(double x) { return x - std::round(x); }

void inverse_transpose_int(const int W[3][3], int out[3][3]) {
  // W is unimodular: inverse = adjugate / det
  const int det = W[0][0] * (W[1][1] * W[2][2] - W[1][2] * W[2][1]) - W[0][1] * (W[1][0] * W[2][2] - W[1][2] * W[2][0]) +
                  W[0][2] * (W[1][0] * W[2][1] - W[1][1] * W[2][0]);
  int inv[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      const int a = (j + 1) % 3, b = (j + 2) % 3, c = (i + 1) % 3, d = (i + 2) % 3;
      inv[i][j] = (W[a][c] * W[b][d] - W[a][d] * W[b][c]) * det;  // det = +-1
    }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) out[i][j] = inv[j][i];
}

}  // namespace

std::vector<SymOp> find_symmetry(const Crystal& c, bool enabled, double tol) {
  std::vector<SymOp> ops;
  const int nat = c.natoms();
  auto make_identity = [&]() {
    SymOp op;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        op.W[i][j] = op.Wk[i][j] = (i == j);
        op.R[i][j] = (i == j);
      }
    op.atom_map.resize(nat);
    for (int a = 0; a < nat; ++a) op.atom_map[a] = a;
    return op;
  };
  if (!enabled || nat == 0) {
    ops.push_back(make_identity());
    return ops;
  }

  // metric tensor M_ij = a_i . a_j
  double M[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) M[i][j] = dot(c.A[i], c.A[j]);
  const double mscale = std::max({M[0][0], M[1][1], M[2][2]});

  // atoms are equivalent if species and initial moments agree
  auto same_kind = [&](int a, int b) {
    return c.atoms[a].species == c.atoms[b].species && std::abs(c.atoms[a].magmom - c.atoms[b].magmom) < 1e-6;
  };
  // reference atom: member of the smallest class
  int ref = 0, best = nat + 1;
  for (int a = 0; a < nat; ++a) {
    int cnt = 0;
    for (int b = 0; b < nat; ++b) cnt += same_kind(a, b);
    if (cnt < best) {
      best = cnt;
      ref = a;
    }
  }

  const Mat3 At = transpose3(c.A), Atinv = inverse3(At);
  int W[3][3];
  for (int code = 0; code < 19683; ++code) {
    int k = code;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        W[i][j] = k % 3 - 1;
        k /= 3;
      }
    // W^T M W == M ?
    bool ok = true;
    for (int i = 0; i < 3 && ok; ++i)
      for (int j = 0; j < 3 && ok; ++j) {
        double s = 0.0;
        for (int p = 0; p < 3; ++p)
          for (int q = 0; q < 3; ++q) s += W[p][i] * M[p][q] * W[q][j];
        if (std::abs(s - M[i][j]) > 1e-6 * mscale) ok = false;
      }
    if (!ok) continue;

    auto apply = [&](const Vec3& x) {
      Vec3 y{};
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) y[i] += W[i][j] * x[j];
      return y;
    };
    const Vec3 wref = apply(c.atoms[ref].frac);
    for (int b = 0; b < nat; ++b) {
      if (!same_kind(ref, b)) continue;
      Vec3 t;
      for (int i = 0; i < 3; ++i) t[i] = c.atoms[b].frac[i] - wref[i], t[i] -= std::floor(t[i] + 1e-9);
      std::vector<int> map(nat, -1);
      bool all = true;
      for (int a = 0; a < nat && all; ++a) {
        const Vec3 y = apply(c.atoms[a].frac);
        for (int d = 0; d < nat; ++d) {
          if (!same_kind(a, d)) continue;
          bool match = true;
          for (int i = 0; i < 3; ++i)
            if (std::abs(wrap_diff(y[i] + t[i] - c.atoms[d].frac[i])) > tol) match = false;
          if (match) {
            map[a] = d;
            break;
          }
        }
        if (map[a] < 0) all = false;
      }
      if (!all) continue;
      SymOp op;
      Mat3 Wd;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          op.W[i][j] = W[i][j];
          Wd[i][j] = W[i][j];
        }
      inverse_transpose_int(W, op.Wk);
      op.t = t;
      op.R = matmul3(matmul3(At, Wd), Atinv);
      op.atom_map = map;
      ops.push_back(op);
    }
  }
  LSDA_CHECK(!ops.empty(), "symmetry search failed (identity not found)");
  return ops;
}

std::vector<KPoint> make_kpoints(const int n[3], const Vec3& shift, std::vector<SymOp>& ops) {
  const int ntot = n[0] * n[1] * n[2];
  LSDA_CHECK(ntot > 0, "k-point grid must have positive dimensions");
  auto grid_index = [&](const Vec3& k, int& idx) {  // true if k is a grid point
    int m[3];
    for (int d = 0; d < 3; ++d) {
      const double u = k[d] * n[d] - shift[d];
      const double ur = std::round(u);
      if (std::abs(u - ur) > 1e-6) return false;
      m[d] = ((static_cast<int>(ur) % n[d]) + n[d]) % n[d];
    }
    idx = (m[0] * n[1] + m[1]) * n[2] + m[2];
    return true;
  };
  auto grid_point = [&](int idx) {
    const int m[3] = {idx / (n[1] * n[2]), (idx / n[2]) % n[1], idx % n[2]};
    Vec3 k;
    for (int d = 0; d < 3; ++d) {
      k[d] = (m[d] + shift[d]) / n[d];
      k[d] -= std::round(k[d]);  // fold into (-1/2, 1/2]
      if (k[d] < -0.5 + 1e-9) k[d] += 1.0;
    }
    return k;
  };
  auto rotate = [&](const SymOp& op, const Vec3& k) {
    Vec3 y{};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) y[i] += op.Wk[i][j] * k[j];
    return y;
  };

  // keep only operations that map the grid onto itself
  std::vector<SymOp> kept;
  for (const SymOp& op : ops) {
    bool ok = true;
    for (int idx = 0; idx < ntot && ok; ++idx) {
      int img;
      if (!grid_index(rotate(op, grid_point(idx)), img)) ok = false;
    }
    if (ok) kept.push_back(op);
  }
  LSDA_CHECK(!kept.empty(), "no symmetry operation is compatible with the k-point grid");
  ops.swap(kept);

  std::vector<int> owner(ntot, -1);
  std::vector<KPoint> kpts;
  for (int idx = 0; idx < ntot; ++idx) {
    if (owner[idx] >= 0) continue;
    const int id = static_cast<int>(kpts.size());
    const Vec3 k = grid_point(idx);
    int count = 0;
    for (const SymOp& op : ops) {
      Vec3 kr = rotate(op, k);
      for (int sign = 0; sign < 2; ++sign) {
        int img;
        if (grid_index(kr, img) && owner[img] < 0) {
          owner[img] = id;
          ++count;
        }
        for (double& v : kr) v = -v;  // time reversal
      }
    }
    kpts.push_back({k, static_cast<double>(count) / ntot});
  }
  return kpts;
}

}  // namespace lsda
