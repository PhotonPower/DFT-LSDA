#!/bin/sh
# Point-defect workflow for a B2 compound (NiAl or FeAl):
#   1. perfect supercell and the four point-defect supercells (atomic positions relaxed)
#   2. statistical model -> effective formation energies and defect concentrations
#
# Usage: ./run_defects.sh Ni 2.834 2 "4 4 4"        (TM, lattice constant in A, supercell size n, k-point grid)
#        ./run_defects.sh Fe 2.813 3 "4 4 4" 2 1.0  (optional: nspin and initial Fe moment)
# Environment: XC (lda [default] or pbe; selects the PAW datasets and thereby the functional),
#              ECUT (Ha, default 25), WIDTH (smearing width in Ha, default 0.01), LSDA (path to the executable),
#              DEFECT_TASK (relax = fixed volume [default], cellrelax = zero pressure),
#              ECUT_SMOOTH (Ha, default 0.5; must be identical in all calculations that are compared),
#              PHONONS=1 (additionally Gamma-point phonons of every relaxed supercell -> vibrational
#              formation entropies in the thermodynamic model; many force calculations),
#              MAX_SECONDS (wall-time limit per lsda run; the script stops with exit code 4 and
#              continues from the checkpoint when it is started again)
set -e
TM=${1:-Ni}; A=${2:-2.834}; N=${3:-2}; KP=${4:-"4 4 4"}; NSPIN=${5:-1}; MOM=${6:-0.0}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LSDA=${LSDA:-$ROOT/build/lsda}
DIR=${TM}Al_n${N}; [ "${XC:-lda}" = pbe ] && DIR=${DIR}_pbe
mkdir -p "$DIR"; cd "$DIR"
for D in none vac_tm vac_al al_tm tm_al; do
  TASK=${DEFECT_TASK:-relax}; [ "$D" = none ] && [ "$TASK" = relax ] && TASK=scf
  python3 "$ROOT/tools/b2_supercell.py" --tm "$TM" --a "$A" --n "$N" --defect "$D" --task "$TASK" \
      --xc "${XC:-lda}" --kpoints $KP --nspin "$NSPIN" --moment "$MOM" --moment-antisite 3.0 --ecut "${ECUT:-25}" \
      --width "${WIDTH:-0.01}" --set restart=true --set ecut_smooth="${ECUT_SMOOTH:-0.5}" --set max_seconds="${MAX_SECONDS:-0}" --pseudo-dir "$ROOT/pseudos" -o "$D.in"
  # finished runs are skipped; interrupted ones continue from their checkpoint (restart = true)
  if ! grep -q '"converged": true' "$D.json" 2>/dev/null; then
    rc=0
    "$LSDA" "$D.in" >> "$D.out" || rc=$?
    if [ "$rc" -eq 4 ]; then
      echo "$D: wall-time limit reached, state saved - run this script again to continue"
      exit 4
    elif [ "$rc" -ne 0 ]; then
      echo "$D: lsda failed with exit code $rc (see $DIR/$D.out)"
      exit "$rc"
    fi
  fi
  grep -E "free energy  E-TS|Relaxation converged|NOT" "$D.out" | tail -2
  if [ "${PHONONS:-0}" = 1 ] && [ ! -f "$D.phonons.json" ]; then
    # phonons of the relaxed structure: positions (and cell) come from the checkpoint of the relaxation
    python3 "$ROOT/tools/b2_supercell.py" --tm "$TM" --a "$A" --n "$N" --defect "$D" --task phonons --prefix "$D" \
        --xc "${XC:-lda}" --kpoints $KP --nspin "$NSPIN" --moment "$MOM" --moment-antisite 3.0 --ecut "${ECUT:-25}" \
        --width "${WIDTH:-0.01}" --set restart=true --set ecut_smooth="${ECUT_SMOOTH:-0.5}" --set max_seconds="${MAX_SECONDS:-0}" \
        --set scf_tol=1e-11 --pseudo-dir "$ROOT/pseudos" -o "$D.ph.in" > /dev/null
    rc=0
    "$LSDA" "$D.ph.in" >> "$D.ph.out" || rc=$?
    [ "$rc" -eq 4 ] && { echo "$D: wall-time limit reached during the phonon run - start the script again"; exit 4; }
    [ "$rc" -ne 0 ] && { echo "$D: phonon run failed with exit code $rc"; exit "$rc"; }
  fi
done
VIB=""; [ "${PHONONS:-0}" = 1 ] && VIB="--vibrations"
python3 "$ROOT/tools/defect_thermo.py" --perfect none.json --vac-a vac_tm.json --vac-b vac_al.json \
    --b-on-a al_tm.json --a-on-b tm_al.json --names "$TM" Al $VIB | tee thermo.txt
