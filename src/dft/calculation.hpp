// Self-consistent PAW/LSDA calculation for a periodic crystal.
#pragma once
#include <memory>
#include <optional>

#include "base/common.hpp"
#include "base/fft.hpp"
#include "base/linalg.hpp"
#include "crystal/crystal.hpp"
#include "dft/basis.hpp"
#include "dft/mixer.hpp"
#include "dft/occupations.hpp"
#include "xc/lda.hpp"

namespace lsda {

class Hamiltonian;
struct Projectors;

struct Params {
  // basis
  double ecut = 0.0;          // plane-wave cutoff for wavefunctions (Ha); 0: largest recommendation of the datasets
  double ecut_dense = 0.0;    // cutoff of the dense G-sphere for densities (Ha); 0: dense_factor * ecut
  double dense_factor = 8.0;  // >= 4
  double ecut_smooth = 0.0;   // width (Ha) of the smoothed kinetic-energy cutoff; needed for stresses at constant ecut
  // Brillouin zone
  int kgrid[3] = {1, 1, 1};
  Vec3 kshift{0, 0, 0};
  bool symmetry = true;
  // electrons
  int nspin = 1;
  bool noncollinear = false;  // two-component spinors, magnetization as a vector field (LSDA only, no symmetry)
  int nbands = 0;  // 0: automatic
  std::string smearing = "cold";  // fermi | gauss | mp1 | cold | tetrahedron
  double width = 0.005;  // Ha
  std::optional<double> fixed_moment;
  std::string xc;             // pw92 | pbe; empty: taken from the PAW datasets
  // self-consistency
  int max_scf = 100;
  double scf_tol = 1e-9;       // threshold for the squared norm of the density residual (Hartree metric)
  double mix_alpha = 0.3;      // linear mixing factor (charge density and density matrices)
  double mix_alpha_mag = 0.3;  // same for the magnetization
  double kerker_q0 = 0.8;      // bohr^-1; 0 disables Kerker preconditioning
  int mix_history = 8;
  // eigensolver
  std::string eigensolver = "ppcg";  // ppcg | lobpcg
  int ppcg_block = 8;          // sub-block size of the PPCG Rayleigh-Ritz steps
  int eig_max_iter = 2;        // eigensolver iterations per SCF step
  int eig_max_iter_first = 6;  // in the first SCF step (5x more if no atomic-orbital start is possible)
  double eig_tol = 1e-7;       // tightest residual norm requested from the eigensolver
  double eig_tol_factor = 0.1; // eigensolver tolerance = factor * sqrt(|density residual|^2 / N_el)
  double eig_tol_max = 1e-3;   // loosest residual norm
  // PAW
  int paw_ntheta = 7;          // angular quadrature for the on-site xc correction (2 n^2 points)
  // run control
  // hybrid functionals (xc = pbe0, hse06, hf)
  double exx_fraction = -1.0;   // fraction of exact exchange (< 0: default of the functional)
  double exx_screening = -1.0;  // screening parameter omega in 1/bohr (< 0: default of the functional)
  double exx_tol = 1e-6;        // convergence of the exchange energy between outer iterations (Ha); the error of the
                                // forces is of first order, of the energy of second order (input default for
                                // tasks with forces: 1e-9)
  int exx_max_iter = 30;        // outer iterations (updates of the exchange operator)
  int exx_history = 4;          // Anderson mixing of the exchange operators over this many outer iterations (1: none)
  std::string exx_singularity = "auto";  // Coulomb singularity: truncated, auxiliary or auto
  double max_seconds = 0.0;    // stop cleanly when the wall time of the run would exceed this (0: no limit)
  bool checkpoint_wfc = true;  // store the wavefunctions in the checkpoint (large, but restarts are immediate)
  // misc
  unsigned seed = 20240;
  int verbosity = 1;
};

struct Energies {
  double band = 0, hartree = 0, xc = 0, zero = 0, onsite = 0, entropy_term = 0;
  double total = 0;   // Kohn-Sham energy E
  double free = 0;    // E - TS (variational; consistent with the forces)
  double sigma0 = 0;  // extrapolation to zero smearing
  double harris = 0;  // Harris-Weinert-Foulkes estimate (equals `total` at self-consistency)
  double exx = 0;     // plane-wave part of the exact-exchange energy (hybrid functionals; contained in `total`)
};

class Calculation {
 public:
  Calculation(Crystal crystal, const Params& params);

  bool scf();                           // returns true if converged
  std::vector<Vec3> forces();           // Cartesian forces (Ha/bohr), symmetrized
  // Stress tensor sigma_ab = (1/Omega) dF/d eps_ab (Ha/bohr^3) at fixed plane-wave basis set;
  // the pressure is -trace/3. Requires a converged scf().
  Mat3 stress();
  void move_atoms(const std::vector<Vec3>& frac);

  // Take over wavefunctions, density and density matrices of a calculation with the same basis-set
  // layout (same reference lattice, k-points and bands), e.g. after a small change of the cell.
  // Returns false (and leaves this object untouched) if the layouts differ.
  bool adopt_state(Calculation&& other);
  // Start density and density matrices from another calculation of the same cell (any k-points/symmetry).
  bool adopt_density(const Calculation& other);

  // Checkpoint (atomic positions, pseudo density, PAW density matrices and optionally wavefunctions)
  // for restarting SCF runs and relaxations. load_checkpoint returns false if the file is missing or incompatible.
  void save_checkpoint(const std::string& path) const;
  bool stopped_by_time() const { return stopped_by_time_; }   // scf() was interrupted by max_seconds
  bool scf_converged() const { return scf_converged_; }
  bool load_checkpoint(const std::string& path);
  static bool read_checkpoint_structure(const std::string& path, int natoms, std::vector<Vec3>& frac, Mat3& lattice);

  const Crystal& crystal() const { return c_; }
  const Energies& energies() const { return en_; }
  const Params& params() const { return p_; }
  void set_verbosity(int v) { p_.verbosity = v; }  // 0: no SCF tables (used inside long loops such as MD)
  double fermi_level(int s = 0) const { return occ_.efermi[s]; }
  double total_magnetization() const;          // collinear: N_up - N_down; non-collinear: |M|
  Vec3 magnetization_vector() const;           // non-collinear: total moment (mu_B); collinear: (0, 0, M)
  std::vector<Vec3> local_moment_vectors() const;
  bool noncollinear() const { return noncol_; }
  // Pseudo electron density (valence + pseudo core; without the augmentation inside the PAW spheres) on the
  // fine grid in Gaussian cube format; magnetization = true writes n_up - n_down instead of the total.
  void write_cube(const std::string& path, bool magnetization) const;
  // Kohn-Sham eigenvalues [spin][band] at an arbitrary k-point (fractional reciprocal coordinates) in the
  // self-consistent potential (non-self-consistent band-structure step).
  std::vector<std::vector<double>> eigenvalues_at(const Vec3& kfrac, double tol = 1e-7, int max_iter = 80);
  // eigenvalues of the self-consistent calculation at k-point ik (for consistency checks)
  const std::vector<double>& scf_eigenvalues(int ik, int spin) const { return eig_[static_cast<size_t>(ik) * nspin_ + spin]; }
  const std::vector<KPoint>& kpoints() const { return kpts_; }
  int n_bands() const { return nbands_; }
  int n_spin() const { return nspin_; }

  // Density of states (Gaussian broadening, width in Ha) and its projection on the PAW spheres, resolved
  // by atom, angular momentum and spin. Text file: energy relative to the Fermi level in eV, states/eV.
  void write_dos(const std::string& path, double width, int npoints) const;
  std::vector<double> local_moments() const;   // all-electron moments inside the PAW spheres
  std::vector<double> local_charges() const;   // valence electrons inside the PAW spheres
  int n_symmetry_ops() const { return static_cast<int>(ops_.size()); }
  int n_kpoints() const { return static_cast<int>(kpts_.size()); }
  int n_scf_iterations() const { return n_scf_; }
  const std::vector<SymOp>& symmetry_ops() const { return ops_; }

 private:
  using RhoG = std::vector<std::vector<cplx>>;      // [spin][G]
  using DMat = std::vector<std::vector<RMatrix>>;   // [atom][spin]
  struct Functional {
    double hartree = 0, xc = 0, zero = 0, onsite = 0;
    double total() const { return hartree + xc + zero + onsite; }
  };

  void setup_structure();
  void init_density();
  void init_wavefunctions();
  CMatrix atomic_orbitals(const PWBasis& b) const;
  Functional evaluate(const RhoG& rho, const DMat& D, bool build_potential);
  void add_compensation(const std::vector<std::vector<double>>& Q, std::vector<cplx>& out) const;
  void project_on_shape(const std::vector<cplx>& v, std::vector<std::vector<double>>& W) const;
  void solve_bands(double tol, int max_iter, double& max_residual, int& not_converged);
  Hamiltonian hamiltonian(int ik, int s, const Projectors& proj) const;
  void compute_density(RhoG& rho, DMat& D, RhoG* tau = nullptr) const;
  void symmetrize_density(RhoG& rho) const;
  void symmetrize_D(DMat& D) const;
  double double_counting(const RhoG& rho, const DMat& D) const;
  std::vector<double> pack(const RhoG& rho, const DMat& D) const;
  void unpack(const std::vector<double>& x, RhoG& rho, DMat& D) const;
  void setup_mixer();

  Crystal c_;
  Params p_;
  std::unique_ptr<XCFunctional> xc_;
  // ---- hybrid functionals (exchange.cpp) ----
  std::unique_ptr<XCFunctional> xc_other_;  // the functional that is currently not active (semilocal start / hybrid, meta-GGA)
  bool start_xc_active_ = false;            // the semilocal start functional is in use
  void use_start_xc(bool on);
  // ---- meta-GGA ----
  // tau_: kinetic energy density of the current wavefunctions (not mixed); v_tau = d e_xc / d tau enters the
  // Hamiltonian as -1/2 div(v_tau grad). Until wavefunctions exist the start functional is used.
  bool mgga_ = false, have_tau_ = false;
  RhoG tau_;
  std::vector<std::vector<cplx>> vtau_G_;
  std::vector<std::vector<double>> vtau_coarse_;
  std::vector<std::vector<double>> ff_tauct_;
  std::vector<cplx> tauct_G_;
  bool hybrid_ = false, exx_active_ = false, exx_ready_ = false, exx_truncated_ = true;
  double exx_fraction_ = 0.0, exx_omega_ = 0.0, exx_rc_ = 0.0, exx_v0_ = 0.0;
  double exx_shift_ = 0.0;      // constant of the linearized exchange energy of the inner loop, -1/2 Tr[gamma_in V_in]
  double exx_energy_ = 0.0;
  std::vector<CMatrix> ace_;    // compressed exchange operator of the current orbitals, per k-point and spin
  // history of the outer loop: operators of the output orbitals of earlier iterations with the packed densities
  // (proxy for the Anderson mixing) and the residuals; the operator in use is sum_i exx_coef_[i] * operator i
  struct ExxEntry {
    std::vector<CMatrix> ace;
    std::vector<double> x, r;
    bool has_residual = false;
  };
  std::vector<ExxEntry> exx_hist_;
  std::vector<double> exx_coef_, exx_xin_;
  RMatrix exx_e2_;              // Tr[gamma_i V_j] between the stored iterations
  void setup_exchange();
  void set_exchange_active(bool on);
  double exchange_kernel(double p2) const;
  double exchange_pass(bool build, std::vector<double>* forces);
  double exchange_cross_energy(const std::vector<CMatrix>& ace) const;
  bool scf_cycle();
  Smearing smearing_;
  std::vector<SymOp> ops_;
  std::vector<std::vector<RMatrix>> rot_;  // [op][species] rotation of projector indices
  std::vector<KPoint> kpts_;
  std::vector<double> kweight_;
  TetraGrid tetra_;                       // k-grid connectivity for the tetrahedron method
  FFT3D fft_c_, fft_f_;
  GSphere gs_;
  std::vector<PWBasis> basis_;
  int nspin_ = 1, nbands_ = 0, lmax_comp_ = 0;  // nspin_: separate spin channels of the wavefunctions
  int ncomp_ = 1;        // components of density, potential and density matrices: 1, 2 (up/down) or 4 (n, m_x, m_y, m_z)
  bool noncol_ = false;
  double nelec_ = 0;

  // species form factors on the dense sphere
  std::vector<std::vector<double>> ff_g_;            // [species][l * ng + ig]
  std::vector<std::vector<double>> ff_nct_, ff_vbar_, ff_nvt_;  // [species][ig]
  std::vector<cplx> nct_G_, vbar_G_;                  // structure-dependent sums

  // electronic state
  RhoG rho_;                              // pseudo valence density (input of the current potential)
  DMat D_;
  std::vector<CMatrix> psi_;              // [ik * nspin + s]
  std::vector<CMatrix> proj_psi_;         // <p_i|psi_n>
  std::vector<std::vector<double>> eig_;
  Occupations occ_;
  bool have_occ_ = false, have_density_ = false;
  bool proj_valid_ = false;   // proj_psi_ belongs to the present wavefunctions and atomic positions

  // potentials generated by (rho_, D_)
  std::vector<std::vector<double>> vcoarse_;  // [spin] local potential on the coarse grid
  RhoG veff_G_;                               // [spin] local potential on the dense sphere
  std::vector<cplx> vH_G_;
  DMat dH_;
  std::vector<RMatrix> dS_;                   // per atom (copy of the species matrices)

  Functional f_pot_;        // energy terms of the density that generated the current potential
  double vxc_n_ = 0;        // int v_xc n~ d^3r for that density (including the pseudo core)
  double gga_stress_[6] = {0, 0, 0, 0, 0, 0};  // int sum_s (de/d grad n_s)_a (grad n_s)_b d^3r  (xx yy zz yz xz xy)

  Mixer mixer_;
  Energies en_;
  int n_scf_ = 0;
  bool first_solve_ = true;
  bool lcao_start_ok_ = true;
  bool stopped_by_time_ = false, scf_converged_ = false;
  double last_dr2_ = 1.0;      // density residual of the last SCF iteration
  double resume_dr2_ = -1.0;   // > 0: an interrupted SCF cycle is continued (from a checkpoint)
  bool restarted_ = false;   // density read from a checkpoint: converge the bands well in the first step
};

}  // namespace lsda
