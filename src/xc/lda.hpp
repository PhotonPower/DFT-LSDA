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
  // For n points: energy density per volume e_xc(n_up, n_dn) and potentials v_sigma = d e_xc / d n_sigma.
  virtual void evaluate(size_t n, const double* nup, const double* ndn, double* exc, double* vup,
                        double* vdn) const = 0;
};

// Local spin density approximation: Slater exchange + Perdew-Wang 1992 correlation.
class LSDA_PW92 final : public XCFunctional {
 public:
  std::string name() const override { return "LSDA (Slater exchange + Perdew-Wang 92 correlation)"; }
  void evaluate(size_t n, const double* nup, const double* ndn, double* exc, double* vup,
                double* vdn) const override;
};

std::unique_ptr<XCFunctional> make_xc(const std::string& name);

}  // namespace lsda
