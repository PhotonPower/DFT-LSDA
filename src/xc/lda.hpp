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
class GGA_PBE final : public XCFunctional {
 public:
  std::string name() const override { return "GGA (Perdew-Burke-Ernzerhof)"; }
  bool is_gga() const override { return true; }
  void evaluate(size_t n, const double* nup, const double* ndn, double* exc, double* vup,
                double* vdn) const override;  // zero-gradient limit (= LSDA)
  void evaluate_gga(size_t n, const double* nup, const double* ndn, const double* suu, const double* sud,
                    const double* sdd, double* exc, double* vup, double* vdn, double* vsuu, double* vsud,
                    double* vsdd) const override;
};

// name: "pw92"/"lda" or "pbe"/"gga"; also accepts the PAW-XML notation "LDA/PW", "GGA/PBE"
std::unique_ptr<XCFunctional> make_xc(const std::string& name);

}  // namespace lsda
