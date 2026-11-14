#!/bin/sh
# Point-defect workflow for a B2 compound (NiAl or FeAl):
#   1. perfect supercell and the four point-defect supercells (atomic positions relaxed)
#   2. statistical model -> effective formation energies and defect concentrations
#
# Usage: ./run_defects.sh Ni 2.834 2 "4 4 4"        (TM, lattice constant in A, supercell size n, k-point grid)
#        ./run_defects.sh Fe 2.813 3 "4 4 4" 2 1.0  (optional: nspin and initial Fe moment)
# Environment: ECUT (Ha, default 25), WIDTH (smearing width in Ha, default 0.01), LSDA (path to the executable),
#              DEFECT_TASK (relax = fixed volume [default], cellrelax = zero pressure),
#              ECUT_SMOOTH (Ha, default 0.5; must be identical in all calculations that are compared)
set -e
TM=${1:-Ni}; A=${2:-2.834}; N=${3:-2}; KP=${4:-"4 4 4"}; NSPIN=${5:-1}; MOM=${6:-0.0}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LSDA=${LSDA:-$ROOT/build/lsda}
DIR=${TM}Al_n${N}
mkdir -p "$DIR"; cd "$DIR"
for D in none vac_tm vac_al al_tm tm_al; do
  TASK=${DEFECT_TASK:-relax}; [ "$D" = none ] && [ "$TASK" = relax ] && TASK=scf
  python3 "$ROOT/tools/b2_supercell.py" --tm "$TM" --a "$A" --n "$N" --defect "$D" --task "$TASK" \
      --kpoints $KP --nspin "$NSPIN" --moment "$MOM" --moment-antisite 3.0 --ecut "${ECUT:-25}" \
      --width "${WIDTH:-0.01}" --set restart=true --set ecut_smooth="${ECUT_SMOOTH:-0.5}" --pseudo-dir "$ROOT/pseudos" -o "$D.in"
  # finished runs are skipped; interrupted ones continue from their checkpoint (restart = true)
  if ! grep -q '"converged": true' "$D.json" 2>/dev/null; then
    "$LSDA" "$D.in" >> "$D.out"
  fi
  grep -E "free energy  E-TS|Relaxation converged|NOT" "$D.out" | tail -2
done
python3 "$ROOT/tools/defect_thermo.py" --perfect none.json --vac-a vac_tm.json --vac-b vac_al.json \
    --b-on-a al_tm.json --a-on-b tm_al.json --names "$TM" Al | tee thermo.txt
