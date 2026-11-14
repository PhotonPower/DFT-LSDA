#include "io/input.hpp"

#include "base/parallel.hpp"

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>

namespace lsda {

namespace {

std::string trim(const std::string& s) {
  const size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  const size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

std::string lower(std::string s) {
  for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return s;
}

std::vector<std::string> split(const std::string& s) {
  std::istringstream in(s);
  std::vector<std::string> out;
  for (std::string t; in >> t;) out.push_back(t);
  return out;
}

double to_double(const std::string& s, const std::string& key) {
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  LSDA_CHECK(end != s.c_str() && *end == '\0', "input: bad number '" + s + "' for " + key);
  return v;
}

bool to_bool(const std::string& s, const std::string& key) {
  const std::string v = lower(s);
  if (v == "true" || v == "yes" || v == "1" || v == "on") return true;
  if (v == "false" || v == "no" || v == "0" || v == "off") return false;
  fail("input: bad logical value '" + s + "' for " + key);
}

std::string dirname(const std::string& path) {
  const size_t p = path.find_last_of('/');
  return p == std::string::npos ? "." : path.substr(0, p);
}

}  // namespace

Input read_input(const std::string& path) {
  std::ifstream in(path);
  LSDA_CHECK(in.good(), "cannot open input file '" + path + "'");
  std::map<std::string, std::string> kv;
  std::map<std::string, std::vector<std::string>> blocks;
  std::string line, block;
  int lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    const size_t hash = line.find('#');
    if (hash != std::string::npos) line.erase(hash);
    line = trim(line);
    if (line.empty()) continue;
    const std::vector<std::string> tok = split(line);
    if (!block.empty()) {
      if (lower(tok[0]) == "end") {
        block.clear();
      } else {
        blocks[block].push_back(line);
      }
      continue;
    }
    if (lower(tok[0]) == "begin") {
      LSDA_CHECK(tok.size() == 2, strf("input line %d: expected 'begin <block>'", lineno));
      block = lower(tok[1]);
      blocks[block];
      continue;
    }
    const size_t eq = line.find('=');
    LSDA_CHECK(eq != std::string::npos, strf("input line %d: expected 'key = value'", lineno));
    kv[lower(trim(line.substr(0, eq)))] = trim(line.substr(eq + 1));
  }
  LSDA_CHECK(block.empty(), "input: block '" + block + "' is not closed with 'end'");

  Input inp;
  Params& p = inp.params;
  std::string pseudo_dir = dirname(path), units = "bohr", coordinates = "fractional";
  double scale = 1.0;
  auto take = [&](const std::string& key, auto&& setter) {
    auto it = kv.find(key);
    if (it == kv.end()) return;
    setter(it->second);
    kv.erase(it);
  };
  auto ints3 = [&](const std::string& v, const std::string& key, int* out) {
    const auto t = split(v);
    LSDA_CHECK(t.size() == 3, "input: " + key + " needs three integers");
    for (int i = 0; i < 3; ++i) out[i] = static_cast<int>(to_double(t[i], key));
  };
  take("task", [&](const std::string& v) { inp.task = lower(v); });
  take("prefix", [&](const std::string& v) { inp.prefix = v; });
  take("stress", [&](const std::string& v) { inp.stress = to_bool(v, "stress"); });
  take("dos", [&](const std::string& v) { inp.dos = to_bool(v, "dos"); });
  take("dos_width", [&](const std::string& v) { inp.dos_width = to_double(v, "dos_width"); });
  take("dos_points", [&](const std::string& v) { inp.dos_points = static_cast<int>(to_double(v, "dos_points")); });
  take("restart", [&](const std::string& v) { inp.restart = to_bool(v, "restart"); });
  take("pseudo_dir", [&](const std::string& v) { pseudo_dir = v[0] == '/' ? v : dirname(path) + "/" + v; });
  take("units", [&](const std::string& v) { units = lower(v); });
  take("coordinates", [&](const std::string& v) { coordinates = lower(v); });
  take("scale", [&](const std::string& v) { scale = to_double(v, "scale"); });
  take("ecut", [&](const std::string& v) { p.ecut = to_double(v, "ecut"); });
  take("ecut_dense", [&](const std::string& v) { p.ecut_dense = to_double(v, "ecut_dense"); });
  bool smooth_given = false;
  take("ecut_smooth", [&](const std::string& v) { p.ecut_smooth = to_double(v, "ecut_smooth"); smooth_given = true; });
  take("dense_factor", [&](const std::string& v) { p.dense_factor = to_double(v, "dense_factor"); });
  take("kpoints", [&](const std::string& v) { ints3(v, "kpoints", p.kgrid); });
  take("kshift", [&](const std::string& v) {
    const auto t = split(v);
    LSDA_CHECK(t.size() == 3, "input: kshift needs three numbers (0 or 0.5)");
    for (int i = 0; i < 3; ++i) p.kshift[i] = to_double(t[i], "kshift");
  });
  take("symmetry", [&](const std::string& v) { p.symmetry = to_bool(v, "symmetry"); });
  take("nspin", [&](const std::string& v) { p.nspin = static_cast<int>(to_double(v, "nspin")); });
  take("nbands", [&](const std::string& v) { p.nbands = static_cast<int>(to_double(v, "nbands")); });
  take("smearing", [&](const std::string& v) { p.smearing = lower(v); });
  take("width", [&](const std::string& v) { p.width = to_double(v, "width"); });
  take("fixed_moment", [&](const std::string& v) { p.fixed_moment = to_double(v, "fixed_moment"); });
  take("xc", [&](const std::string& v) { p.xc = lower(v); });
  take("max_scf", [&](const std::string& v) { p.max_scf = static_cast<int>(to_double(v, "max_scf")); });
  take("scf_tol", [&](const std::string& v) { p.scf_tol = to_double(v, "scf_tol"); });
  take("mix_alpha", [&](const std::string& v) { p.mix_alpha = to_double(v, "mix_alpha"); });
  take("mix_alpha_mag", [&](const std::string& v) { p.mix_alpha_mag = to_double(v, "mix_alpha_mag"); });
  take("kerker_q0", [&](const std::string& v) { p.kerker_q0 = to_double(v, "kerker_q0"); });
  take("mix_history", [&](const std::string& v) { p.mix_history = static_cast<int>(to_double(v, "mix_history")); });
  take("eigensolver", [&](const std::string& v) { p.eigensolver = lower(v); });
  take("ppcg_block", [&](const std::string& v) { p.ppcg_block = static_cast<int>(to_double(v, "ppcg_block")); });
  take("eig_max_iter", [&](const std::string& v) { p.eig_max_iter = static_cast<int>(to_double(v, "eig_max_iter")); });
  take("eig_max_iter_first", [&](const std::string& v) { p.eig_max_iter_first = static_cast<int>(to_double(v, "eig_max_iter_first")); });
  take("eig_tol", [&](const std::string& v) { p.eig_tol = to_double(v, "eig_tol"); });
  take("eig_tol_factor", [&](const std::string& v) { p.eig_tol_factor = to_double(v, "eig_tol_factor"); });
  take("eig_tol_max", [&](const std::string& v) { p.eig_tol_max = to_double(v, "eig_tol_max"); });
  take("paw_ntheta", [&](const std::string& v) { p.paw_ntheta = static_cast<int>(to_double(v, "paw_ntheta")); });
  take("max_seconds", [&](const std::string& v) { p.max_seconds = to_double(v, "max_seconds"); });
  take("checkpoint_wfc", [&](const std::string& v) { p.checkpoint_wfc = to_bool(v, "checkpoint_wfc"); });
  take("seed", [&](const std::string& v) { p.seed = static_cast<unsigned>(to_double(v, "seed")); });
  take("relax_steps", [&](const std::string& v) { inp.relax.max_steps = static_cast<int>(to_double(v, "relax_steps")); });
  take("force_tol", [&](const std::string& v) { inp.relax.force_tol = to_double(v, "force_tol"); });
  take("max_step", [&](const std::string& v) { inp.relax.max_step = to_double(v, "max_step"); });
  const double gpa = 29421.02648438959;
  take("pressure", [&](const std::string& v) { inp.cell.pressure = to_double(v, "pressure") / gpa; });
  take("stress_tol", [&](const std::string& v) { inp.cell.stress_tol = to_double(v, "stress_tol") / gpa; });
  take("cell_modulus", [&](const std::string& v) { inp.cell.modulus = to_double(v, "cell_modulus") / gpa; });
  take("cell_steps", [&](const std::string& v) { inp.cell.max_steps = static_cast<int>(to_double(v, "cell_steps")); });
  take("max_strain", [&](const std::string& v) { inp.cell.max_strain = to_double(v, "max_strain"); });
  take("neb_images", [&](const std::string& v) { inp.neb.n_images = static_cast<int>(to_double(v, "neb_images")); });
  take("neb_spring", [&](const std::string& v) { inp.neb.spring = to_double(v, "neb_spring"); });
  take("neb_climb", [&](const std::string& v) { inp.neb.climb = to_bool(v, "neb_climb"); });
  take("neb_steps", [&](const std::string& v) { inp.neb.max_steps = static_cast<int>(to_double(v, "neb_steps")); });
  take("displacement", [&](const std::string& v) { inp.phonons.displacement = to_double(v, "displacement"); });
  LSDA_CHECK(inp.task == "scf" || inp.task == "forces" || inp.task == "stress" || inp.task == "relax" ||
                 inp.task == "cellrelax" || inp.task == "neb" || inp.task == "phonons",
             "input: task must be scf, forces, stress, relax, cellrelax, neb or phonons");
  if (inp.task == "stress" || inp.task == "cellrelax") inp.stress = true;
  // a cell optimization needs an energy that is a smooth function of the cell at constant cutoff
  if (inp.task == "cellrelax" && !smooth_given) p.ecut_smooth = 0.5;
  if (!kv.empty()) fail("input: unknown keyword '" + kv.begin()->first + "'");
  LSDA_CHECK(units == "bohr" || units == "angstrom", "input: units must be bohr or angstrom");
  LSDA_CHECK(coordinates == "fractional" || coordinates == "cartesian", "input: coordinates must be fractional or cartesian");
  const double length = scale * (units == "angstrom" ? 1.0 / bohr_in_angstrom : 1.0);

  // lattice
  LSDA_CHECK(blocks.count("lattice") && blocks["lattice"].size() == 3, "input: block 'lattice' needs three vectors");
  Mat3 A;
  for (int i = 0; i < 3; ++i) {
    const auto t = split(blocks["lattice"][i]);
    LSDA_CHECK(t.size() == 3, "input: lattice vectors need three components");
    for (int j = 0; j < 3; ++j) A[i][j] = length * to_double(t[j], "lattice");
  }
  inp.crystal.set_lattice(A);
  // optional reference lattice that defines the plane-wave basis (constant-basis strain calculations)
  if (blocks.count("basis_lattice")) {
    LSDA_CHECK(blocks["basis_lattice"].size() == 3, "input: block 'basis_lattice' needs three vectors");
    Mat3 Ar;
    for (int i = 0; i < 3; ++i) {
      const auto t = split(blocks["basis_lattice"][i]);
      LSDA_CHECK(t.size() == 3, "input: basis_lattice vectors need three components");
      for (int j = 0; j < 3; ++j) Ar[i][j] = length * to_double(t[j], "basis_lattice");
    }
    inp.crystal.set_basis_lattice(Ar);
  }

  // species:  <label> <dataset file>
  LSDA_CHECK(blocks.count("species") && !blocks["species"].empty(), "input: block 'species' is missing");
  for (const std::string& l : blocks["species"]) {
    const auto t = split(l);
    LSDA_CHECK(t.size() == 2 || t.size() == 3, "input: species line must be '<label> <PAW dataset file> [mass in amu]'");
    Species sp;
    sp.name = t[0];
    if (t.size() == 3) sp.mass = to_double(t[2], "species mass");
    sp.file = t[1][0] == '/' ? t[1] : pseudo_dir + "/" + t[1];
    sp.setup = PawSetup::load(sp.file, p.paw_ntheta);
    inp.crystal.species.push_back(std::move(sp));
  }

  // atoms:  <label> x y z [magnetic moment]
  LSDA_CHECK(blocks.count("atoms") && !blocks["atoms"].empty(), "input: block 'atoms' is missing");
  for (const std::string& l : blocks["atoms"]) {
    const auto t = split(l);
    LSDA_CHECK(t.size() == 4 || t.size() == 5, "input: atom line must be '<label> x y z [moment]'");
    Atom at;
    at.species = -1;
    for (size_t is = 0; is < inp.crystal.species.size(); ++is)
      if (inp.crystal.species[is].name == t[0]) at.species = static_cast<int>(is);
    LSDA_CHECK(at.species >= 0, "input: atom of unknown species '" + t[0] + "'");
    Vec3 v;
    for (int j = 0; j < 3; ++j) v[j] = to_double(t[1 + j], "atoms");
    if (coordinates == "cartesian") {
      for (double& c : v) c *= length;
      v = inp.crystal.frac_from_cart(v);
    }
    at.frac = v;
    if (t.size() == 5) at.magmom = to_double(t[4], "atoms");
    inp.crystal.atoms.push_back(at);
  }
  // second end point of a NEB path: same atoms in the same order
  if (blocks.count("atoms_final")) {
    LSDA_CHECK(blocks["atoms_final"].size() == inp.crystal.atoms.size(), "input: atoms_final needs the same number of atoms as atoms");
    for (size_t ia = 0; ia < blocks["atoms_final"].size(); ++ia) {
      const auto t = split(blocks["atoms_final"][ia]);
      LSDA_CHECK(t.size() >= 4, "input: atoms_final line must be '<label> x y z'");
      LSDA_CHECK(t[0] == inp.crystal.species[inp.crystal.atoms[ia].species].name, "input: atoms_final must list the atoms in the same order");
      Vec3 v;
      for (int j = 0; j < 3; ++j) v[j] = to_double(t[1 + j], "atoms_final");
      if (coordinates == "cartesian") {
        for (double& c : v) c *= length;
        v = inp.crystal.frac_from_cart(v);
      }
      inp.frac_final.push_back(v);
    }
  }
  LSDA_CHECK(inp.task != "neb" || !inp.frac_final.empty(), "input: task = neb needs the block 'atoms_final'");
  inp.neb.force_tol = inp.relax.force_tol;
  inp.neb.max_step = inp.relax.max_step;
  inp.phonons.prefix = inp.prefix;
  inp.phonons.restart = inp.restart;
  inp.neb.prefix = inp.prefix;
  inp.neb.restart = inp.restart;
  for (const auto& [name, lines] : blocks)
    LSDA_CHECK(name == "lattice" || name == "basis_lattice" || name == "species" || name == "atoms" || name == "atoms_final",
               "input: unknown block '" + name + "'");
  return inp;
}

void write_results(const std::string& path, const Calculation& calc, bool converged, const std::vector<Vec3>* forces,
                   const Mat3* stress) {
  if (!par::root()) return;
  std::FILE* f = std::fopen(path.c_str(), "w");
  LSDA_CHECK(f != nullptr, "cannot write '" + path + "'");
  const Crystal& c = calc.crystal();
  const Energies& e = calc.energies();
  std::fprintf(f, "{\n  \"converged\": %s,\n", converged ? "true" : "false");
  std::fprintf(f, "  \"units\": {\"energy\": \"Hartree\", \"length\": \"bohr\", \"force\": \"Hartree/bohr\", \"stress\": \"Hartree/bohr^3\"},\n");
  std::fprintf(f, "  \"energy\": %.12f,\n  \"free_energy\": %.12f,\n  \"energy_sigma0\": %.12f,\n", e.total, e.free, e.sigma0);
  std::fprintf(f, "  \"fermi_level\": %.10f,\n  \"magnetization\": %.8f,\n", calc.fermi_level(), calc.total_magnetization());
  std::fprintf(f, "  \"n_atoms\": %d,\n  \"n_electrons\": %.6f,\n  \"volume\": %.10f,\n", c.natoms(), c.n_electrons(), c.volume);
  std::fprintf(f, "  \"scf_iterations\": %d,\n  \"n_kpoints\": %d,\n  \"n_symmetry\": %d,\n", calc.n_scf_iterations(),
               calc.n_kpoints(), calc.n_symmetry_ops());
  if (stress) {
    const Mat3& t = *stress;
    std::fprintf(f, "  \"stress\": [[%.12e, %.12e, %.12e], [%.12e, %.12e, %.12e], [%.12e, %.12e, %.12e]],\n", t[0][0], t[0][1],
                 t[0][2], t[1][0], t[1][1], t[1][2], t[2][0], t[2][1], t[2][2]);
    std::fprintf(f, "  \"pressure_GPa\": %.8f,\n", -(t[0][0] + t[1][1] + t[2][2]) / 3.0 * 29421.02648438959);
  }
  std::fprintf(f, "  \"lattice\": [");
  for (int i = 0; i < 3; ++i) std::fprintf(f, "[%.12f, %.12f, %.12f]%s", c.A[i][0], c.A[i][1], c.A[i][2], i < 2 ? ", " : "],\n");
  const std::vector<double> mom = calc.local_moments();
  std::fprintf(f, "  \"atoms\": [\n");
  for (int a = 0; a < c.natoms(); ++a) {
    const Atom& at = c.atoms[a];
    std::fprintf(f, "    {\"species\": \"%s\", \"frac\": [%.12f, %.12f, %.12f], \"moment\": %.6f", c.species[at.species].name.c_str(),
                 at.frac[0], at.frac[1], at.frac[2], mom[a]);
    if (forces) std::fprintf(f, ", \"force\": [%.10f, %.10f, %.10f]", (*forces)[a][0], (*forces)[a][1], (*forces)[a][2]);
    std::fprintf(f, "}%s\n", a + 1 < c.natoms() ? "," : "");
  }
  std::fprintf(f, "  ]\n}\n");
  std::fclose(f);
}

}  // namespace lsda
