// Density mixing: Pulay/Anderson (multisecant Broyden) acceleration with a diagonal preconditioner
// (Kerker for the charge density) and a weighted inner product.
#pragma once
#include "base/common.hpp"

namespace lsda {

class Mixer {
 public:
  Mixer() = default;
  // weight: metric of the inner product, precond: linear-mixing factor per component
  Mixer(std::vector<double> weight, std::vector<double> precond, int history)
      : weight_(std::move(weight)), precond_(std::move(precond)), history_(history) {}

  // Given input x and output F(x) of the self-consistency map, return the next input.
  std::vector<double> mix(const std::vector<double>& x_in, const std::vector<double>& x_out);
  void reset() { dx_.clear(); dr_.clear(); have_prev_ = false; }
  double dot(const std::vector<double>& a, const std::vector<double>& b) const;

 private:
  std::vector<double> weight_, precond_;
  int history_ = 8;
  bool have_prev_ = false;
  std::vector<double> x_prev_, r_prev_;
  std::vector<std::vector<double>> dx_, dr_;
};

}  // namespace lsda
