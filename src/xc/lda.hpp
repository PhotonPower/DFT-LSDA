// Exchange-correlation functionals. The interface works on spin densities so that
// further functionals (GGA, ...) can be added by deriving from XCFunctional.
#pragma once
#include <memory>

#include "base/common.hpp"

namespace lsda {

class XCFunctional {
 public:
  virtual ~XCFunctional() = default;
  virtual std::string name() const = 0;
  virtual bool is_gga() const { return false; }
  // Meta-GGA: depends in addition on the kinetic energy densities tau_s = 1/2 sum_i f_i |grad psi_i,s|^2;
  // returns in addition v_tau,s = d e_xc / d tau_s (is_gga() is true as well: gradients are needed).
  virtual bool is_mgga() const { return false; }
  virtual void evaluate_mgga(size_t n, const double* nup, const double* ndn, const double* suu, const double* sud,
                             const double* sdd, const double* tup, const double* tdn, double* exc, double* vup,
                             double* vdn, double* vsuu, double* vsud, double* vsdd, double* vtup, double* vtdn) const;
  // Hybrid functionals: fraction of exact (Fock) exchange and its screening parameter omega (1/bohr);
  // omega = 0: full Coulomb kernel, omega > 0: short-range kernel erfc(omega r)/r.
  virtual double exx_fraction() const { return 0.0; }
  virtual double exx_screening() const { return 0.0; }
  // Local functionals. For n points: energy density per volume e_xc(n_up, n_dn) and
  // potentials v_sigma = d e_xc / d n_sigma.
  virtual void evaluate(size_t n, const double* nup, const double* ndn, double* exc, double* vup,
                        double* vdn) const = 0;
  // Gradient-corrected functionals: e_xc(n_up, n_dn, s_uu, s_ud, s_dd) with s_xy = grad n_x . grad n_y;
  // returns in addition the partial derivatives with respect to the three gradient invariants.
  virtual void evaluate_gga(size_t n, const double* nup, const double* ndn, const double* suu, const double* sud,
                            const double* sdd, double* exc, double* vup, double* vdn, double* vsuu, double* vsud,
                            double* vsdd) const;
};

// Local spin density approximation: Slater exchange + Perdew-Wang 1992 correlation.
class LSDA_PW92 final : public XCFunctional {
 public:
  std::string name() const override { return "LSDA (Slater exchange + Perdew-Wang 92 correlation)"; }
  void evaluate(size_t n, const double* nup, const double* ndn, double* exc, double* vup,
                double* vdn) const override;
};

// Generalized gradient approximation of Perdew, Burke and Ernzerhof (PRL 77, 3865 (1996)), spin-polarized.
// With exx_fraction a > 0 the semilocal part of a hybrid functional:
//   omega = 0 (PBE0):  E_xc = a E_x^Fock + (1 - a) E_x^PBE + E_c^PBE
//   omega > 0 (HSE):   E_xc = a E_x^Fock,SR(omega) + E_x^PBE - a E_x^PBE,SR(omega) + E_c^PBE
// The short-range PBE exchange uses the exchange-hole model of Henderson, Janesko and Scuseria,
// J. Chem. Phys. 128, 194105 (2008).
class GGA_PBE final : public XCFunctional {
 public:
  explicit GGA_PBE(double exx_fraction = 0.0, double omega = 0.0) : exx_(exx_fraction), omega_(omega) {}
  std::string name() const override;
  bool is_gga() const override { return true; }
  double exx_fraction() const override { return exx_; }
  double exx_screening() const override { return omega_; }
  void evaluate(size_t n, const double* nup, const double* ndn, double* exc, double* vup,
                double* vdn) const override;  // zero-gradient limit (= LSDA)
  void evaluate_gga(size_t n, const double* nup, const double* ndn, const double* suu, const double* sud,
                    const double* sdd, double* exc, double* vup, double* vdn, double* vsuu, double* vsud,
                    double* vsdd) const override;

 private:
  double exx_, omega_;
};

// r2SCAN meta-GGA (Furness et al., J. Phys. Chem. Lett. 11, 8208 (2020))
class MGGA_R2SCAN final : public XCFunctional {
 public:
  std::string name() const override { return "meta-GGA r2SCAN"; }
  bool is_gga() const override { return true; }
  bool is_mgga() const override { return true; }
  void evaluate(size_t, const double*, const double*, double*, double*, double*) const override;
  void evaluate_mgga(size_t n, const double* nup, const double* ndn, const double* suu, const double* sud,
                     const double* sdd, const double* tup, const double* tdn, double* exc, double* vup, double* vdn,
                     double* vsuu, double* vsud, double* vsdd, double* vtup, double* vtdn) const override;
};

// Hartree-Fock: exact exchange only, no semilocal exchange-correlation (for tests and reference calculations)
class HartreeFock final : public XCFunctional {
 public:
  std::string name() const override { return "Hartree-Fock (exact exchange, no correlation)"; }
  double exx_fraction() const override { return 1.0; }
  void evaluate(size_t n, const double*, const double*, double* exc, double* vup, double* vdn) const override {
    for (size_t i = 0; i < n; ++i) exc[i] = vup[i] = vdn[i] = 0.0;
  }
};

// Enhancement factor of the short-range PBE exchange, F(s, nu) with nu = omega / k_F (HJS model);
// F(s, 0) reproduces the PBE enhancement factor closely, F(0, nu) is the exact short-range LDA factor.
double sr_pbe_enhancement(double s, double nu, double* dF_ds = nullptr, double* dF_dnu = nullptr);

// name: "pw92"/"lda", "pbe"/"gga" (also the PAW-XML notation "LDA/PW", "GGA/PBE"), or the hybrids "pbe0",
// "hse06", "hf". exx_fraction/omega < 0: defaults of the functional (PBE0: 0.25; HSE06: 0.25, 0.11/bohr).
std::unique_ptr<XCFunctional> make_xc(const std::string& name, double exx_fraction = -1.0, double omega = -1.0);

}  // namespace lsda
