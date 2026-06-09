#!/usr/bin/env bash
# =============================================================================
#  setup_gmx.sh — GROMACS pre-production pipeline (nanobody + target complex)
# =============================================================================
#  Usage : ./setup_gmx.sh <complex.pdb> [output_dir]
#          output_dir defaults to ./gmx_prep
#
#  Targets  : Ace / EbpC / Esp (Enterococcus faecalis virulence proteins)
#  Nanobody : VHH scaffold designed by RFAntibody (RFdiffusion + ProteinMPNN)
#  Pipeline : pdb2gmx → editconf → solvate → genion → EM → NVT → NPT
#  FF       : AMBER ff99SB-ILDN + TIP3P
#  Box      : Dodecahedron, 1.2 nm margin
#  Ions     : 0.15 M NaCl + neutralise
#  Temp     : 310 K (physiological)
#  Chains   : merged into one molecule (-merge all)
#  GPU      : CUDA offload for A100 (nb + PME + bonded + update)
#
#  ⚠️  BEFORE RUNNING — verify your PDB has:
#    • Signal peptide removed  (Ace ~aa1-26 / EbpC ~aa1-30 / Esp ~aa1-26)
#    • LPXTG anchor removed    (last ~35-60 aa of each target)
#    • No HETATM ligands unless separately parametrised
# =============================================================================
set -euo pipefail

# ── GROMACS executable ────────────────────────────────────────────────────────
GMX="gmx_mpi"       # MPI build on HPC; change to "gmx" for thread-MPI builds

# ── Configurable parameters ───────────────────────────────────────────────────
TEMP=310            # Kelvin (physiological)
FF="amber99sb-ildn"
WATER="tip3p"
BOX_TYPE="dodecahedron"
BOX_D="1.2"         # nm — protein-to-box-edge distance
ION_CONC="0.15"     # M NaCl
EM_STEPS=50000      # steepest descent steps (max)
NVT_STEPS=50000     # 100 ps at dt=0.002 ps
NPT_STEPS=50000     # 100 ps at dt=0.002 ps

# ── GPU / CPU settings (tuned for A100 80GB) ──────────────────────────────────
GPU_ID=0            # GPU device index (0 = first GPU)
CPU_THREADS=8       # OpenMP threads per mdrun call; raise if your node has >8 cores/GPU

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*" >&2; }
ok()   { echo -e "${GREEN}[ OK ]${NC}  $*"; }
die()  { echo -e "${RED}[FAIL]${NC}  $*" >&2; exit 1; }
step() { echo -e "\n${BOLD}${GREEN}━━━  $*  ━━━${NC}"; }

# ── GPU-accelerated mdrun wrapper ─────────────────────────────────────────────
#    gmx_mpi uses MPI ranks (mpirun/srun) — no -ntmpi flag (thread-MPI only)
#    EM  : only non-bonded on GPU (steep integrator doesn't support full offload)
#    MD  : nb + PME + bonded + coordinate-update all on GPU (A100 optimal)
run_em() {
    ${GMX} mdrun -v -deffnm "$1" \
        -ntomp "${CPU_THREADS}" \
        -gpu_id "${GPU_ID}" \
        -nb gpu \
        -pin on "$@"
}
run_md() {
    ${GMX} mdrun -v -deffnm "$1" \
        -ntomp "${CPU_THREADS}" \
        -gpu_id "${GPU_ID}" \
        -nb gpu -pme gpu -bonded gpu -update gpu \
        -pin on "$@"
}

# ── Validate input ────────────────────────────────────────────────────────────
[[ $# -lt 1 ]] && die "Usage: $0 <complex.pdb> [output_dir]"
INPUT_PDB="$(realpath "$1")"
[[ -f "$INPUT_PDB" ]] || die "PDB not found: $INPUT_PDB"
OUTDIR="${2:-./gmx_prep}"

# ── Require GROMACS ───────────────────────────────────────────────────────────
command -v "${GMX}" &>/dev/null || {
    die "${GMX} not found. Activate your environment first:\n  micromamba activate <your-gromacs-env>"
}
GMX_VER=$(${GMX} --version 2>&1 | awk '/GROMACS version/{print $NF}')
log "GROMACS version : $GMX_VER"
log "Input PDB       : $INPUT_PDB"
log "Output dir      : $OUTDIR"

# ── Create and enter output directory ────────────────────────────────────────
mkdir -p "$OUTDIR"
cd "$OUTDIR"

# =============================================================================
# STEP 0 — Pre-process PDB
# =============================================================================
step "STEP 0: Pre-processing PDB"

# Count potentially problematic records
HETATM_N=$(grep -c "^HETATM" "$INPUT_PDB" || true)
HOH_N=$(grep -cE "^(ATOM|HETATM).{12}.{4}HOH" "$INPUT_PDB" || true)
SS_N=$(grep -c "^SSBOND" "$INPUT_PDB" || true)
CHAIN_IDS=$(grep "^ATOM" "$INPUT_PDB" | cut -c22 | sort -u | tr -d ' \n')

if [[ "$HETATM_N" -gt 0 ]]; then
    warn "$HETATM_N HETATM record(s) found — will be stripped."
    warn "If your complex has essential small-molecule ligands, parametrise them"
    warn "with acpype/antechamber first and merge their topology manually."
fi
[[ "$HOH_N" -gt 0 ]] && \
    warn "Stripping $HOH_N crystallographic water atom(s) — TIP3P added later."

[[ "$SS_N" -gt 0 ]] && \
    log "Found $SS_N SSBOND record(s) in PDB (will also scan by distance via -ss)."

log "Chain IDs detected: [${CHAIN_IDS}]"

# ── E. faecalis-specific domain warnings ─────────────────────────────────────
# Check approximate PDB size to catch un-truncated proteins
ATOM_TOTAL=$(grep -c "^ATOM" "$INPUT_PDB" || true)
HEAVY_APPROX=$(( ATOM_TOTAL / 5 ))   # rough residue count estimate
log "Total ATOM records: $ATOM_TOTAL (≈$HEAVY_APPROX residues)"

if [[ "$HEAVY_APPROX" -gt 400 ]]; then
    warn "Large complex detected. Verify that signal peptides and LPXTG anchors"
    warn "have been removed from each target chain before proceeding:"
    warn "  Ace   → remove ~aa1-26 (N-term) and ~last 60 aa (LPXTG anchor)"
    warn "  EbpC  → remove ~aa1-30 (N-term) and ~last 35 aa (LPXTG anchor)"
    warn "  Esp   → remove ~aa1-26 (N-term) and ~last 60 aa (LPXTG anchor)"
fi

# Strip HETATM and crystallographic waters; keep protein + structural headers
grep -E "^(ATOM|SSBOND|SEQRES|TER|END)" "$INPUT_PDB" \
    | grep -vE ".{17}HOH" \
    > clean_input.pdb

ATOM_CLEAN=$(grep -c "^ATOM" clean_input.pdb || true)
ok "clean_input.pdb written: $ATOM_CLEAN ATOM records retained"

# ── Snap close CYS SG pairs into disulfide detection range ────────────────────
# RFdiffusion does not model disulfide bonds.  ProteinMPNN places CYS22/96 by
# backbone geometry; the resulting SG–SG distance can be 2.0–2.8 Å — a range
# that is "close enough to clash but sometimes outside pdb2gmx's specbond.dat
# detection threshold" (1.25 × 0.2 nm = 2.5 Å).
#
# Strategy: for any CYS SG pair in (2.1, 3.0) Å, move both atoms symmetrically
# to exactly 2.04 Å (ideal S–S bond length).  pdb2gmx -ss then reliably detects
# all pairs ≤ 2.5 Å and prompts for confirmation; yes y answers every prompt.
# We do NOT inject SSBOND records: combining them with -ss causes pdb2gmx 2026
# to double-process the bond (once from the record, once from specbond.dat),
# corrupting the topology for designs whose SG–SG is already at bond distance.
python3 - << 'PYEOF'
import sys, math

SS_IDEAL   = 2.04   # Å — target SG–SG after snap
SS_SNAP_LO = 2.1    # Å — snap only if distance is above this
SS_DETECT  = 3.0    # Å — pairs closer than this are disulfide candidates

pdb = 'clean_input.pdb'
with open(pdb) as f:
    lines = f.readlines()

# Parse CYS SG atoms with their line index for in-place coordinate patching
cys_sg = []
for idx, line in enumerate(lines):
    if (line.startswith('ATOM')
            and line[12:16].strip() == 'SG'
            and line[17:20].strip() == 'CYS'):
        chain  = line[21]
        resnum = int(line[22:26].strip())
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        cys_sg.append((chain, resnum, x, y, z, idx))

snapped = False
for i in range(len(cys_sg)):
    for j in range(i + 1, len(cys_sg)):
        c1, r1, x1, y1, z1, li1 = cys_sg[i]
        c2, r2, x2, y2, z2, li2 = cys_sg[j]
        d = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        if SS_SNAP_LO < d < SS_DETECT:
            mx = (x1+x2)/2;  my = (y1+y2)/2;  mz = (z1+z2)/2
            dx = x1-x2;      dy = y1-y2;      dz = z1-z2
            half = (SS_IDEAL / 2) / d
            nx1 = mx + dx*half;  ny1 = my + dy*half;  nz1 = mz + dz*half
            nx2 = mx - dx*half;  ny2 = my - dy*half;  nz2 = mz - dz*half
            old1 = lines[li1]
            lines[li1] = old1[:30] + f'{nx1:8.3f}{ny1:8.3f}{nz1:8.3f}' + old1[54:]
            old2 = lines[li2]
            lines[li2] = old2[:30] + f'{nx2:8.3f}{ny2:8.3f}{nz2:8.3f}' + old2[54:]
            print(f'  [SS-snap] CYS-{c1}{r1}--CYS-{c2}{r2}: {d:.2f} Å → {SS_IDEAL:.2f} Å',
                  file=sys.stderr)
            snapped = True

if snapped:
    with open(pdb, 'w') as f:
        f.writelines(lines)
PYEOF

# =============================================================================
# STEP 1 — pdb2gmx: topology + force field
# =============================================================================
step "STEP 1: pdb2gmx  [$FF / $WATER / merge all chains]"

PDB2GMX_ARGS=(
    -f  clean_input.pdb
    -o  processed.gro
    -p  topol.top
    -i  posre.itp
    -ff "$FF"
    -water "$WATER"
    -merge all    # both chains → single molecule entry; no inter-chain bond added
    -ignh         # discard input H atoms; rebuild from FF definitions
)

[[ "$SS_N" -gt 0 ]] && log "SSBOND record(s) found in source PDB."
# -ss scans for CYS SG pairs within specbond.dat threshold (2.5 Å) and prompts.
# Coordinate snap above ensures RFdiffusion designs land within that threshold.
# set +o pipefail prevents SIGPIPE 141 when pdb2gmx closes stdin before yes ends.
set +o pipefail
yes y 2>/dev/null | ${GMX} pdb2gmx "${PDB2GMX_ARGS[@]}" -ss
PDB2GMX_RC="${PIPESTATUS[1]}"
set -o pipefail
[[ "$PDB2GMX_RC" -eq 0 ]] || die "pdb2gmx failed (exit ${PDB2GMX_RC})"
ok "pdb2gmx done → processed.gro, topol.top"

# =============================================================================
# STEP 2 — editconf: simulation box
# =============================================================================
step "STEP 2: editconf  [$BOX_TYPE, d = ${BOX_D} nm]"

${GMX} editconf \
    -f  processed.gro \
    -o  boxed.gro \
    -c \
    -d  "$BOX_D" \
    -bt "$BOX_TYPE"
ok "Box defined → boxed.gro"

# =============================================================================
# STEP 3 — solvate: TIP3P water
# =============================================================================
step "STEP 3: solvate  [TIP3P, spc216.gro template]"

${GMX} solvate \
    -cp boxed.gro \
    -cs spc216.gro \
    -o  solvated.gro \
    -p  topol.top
ok "System solvated → solvated.gro"

# =============================================================================
# STEP 4 — genion: 0.15 M NaCl + neutralise
# =============================================================================
step "STEP 4: genion  [${ION_CONC} M NaCl + neutralise]"

cat > ions.mdp << 'MDP'
; Minimal mdp — used only to generate ions.tpr for genion preprocessing
integrator    = steep
emtol         = 1000.0
nsteps        = 1
nstlist       = 1
cutoff-scheme = Verlet
coulombtype   = cutoff
rcoulomb      = 1.0
rvdw          = 1.0
pbc           = xyz
MDP

${GMX} grompp \
    -f ions.mdp \
    -c solvated.gro \
    -p topol.top \
    -o ions.tpr \
    -maxwarn 2

echo "SOL" | ${GMX} genion \
    -s  ions.tpr \
    -o  ionized.gro \
    -p  topol.top \
    -pname NA \
    -nname CL \
    -neutral \
    -conc "$ION_CONC"
ok "Ions added → ionized.gro"

# =============================================================================
# Write .mdp parameter files
# =============================================================================
step "Writing .mdp files  [minim / nvt / npt]"

# ── minim.mdp ─────────────────────────────────────────────────────────────────
cat > minim.mdp << MDP
; ── Energy Minimisation (steepest descent) ────────────────────────────────────
integrator              = steep
emtol                   = 1000.0        ; kJ/mol/nm — convergence threshold
emstep                  = 0.01
nsteps                  = ${EM_STEPS}

cutoff-scheme           = Verlet
ns_type                 = grid
nstlist                 = 1
rcoulomb                = 1.0
rvdw                    = 1.0
coulombtype             = PME
pme_order               = 4
fourierspacing          = 0.16

nstxout                 = 0
nstvout                 = 0
nstenergy               = 500
nstlog                  = 500

pbc                     = xyz
MDP

# ── nvt.mdp ───────────────────────────────────────────────────────────────────
cat > nvt.mdp << MDP
; ── NVT Equilibration — 100 ps @ ${TEMP} K, heavy atoms position-restrained ──
define                  = -DPOSRES

integrator              = md
nsteps                  = ${NVT_STEPS}  ; 100 ps  (dt = 0.002 ps)
dt                      = 0.002

nstxout                 = 500
nstvout                 = 500
nstenergy               = 500
nstlog                  = 500

cutoff-scheme           = Verlet
ns_type                 = grid
nstlist                 = 10
rcoulomb                = 1.0
rvdw                    = 1.0
coulombtype             = PME
pme_order               = 4
fourierspacing          = 0.16

; Temperature — V-rescale (canonical)
tcoupl                  = V-rescale
tc-grps                 = Protein Non-Protein
tau_t                   = 0.1   0.1
ref_t                   = ${TEMP}  ${TEMP}

pcoupl                  = no         ; NVT — no pressure coupling

constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4

gen_vel                 = yes        ; generate velocities from Maxwell-Boltzmann
gen_temp                = ${TEMP}
gen_seed                = -1
continuation            = no

pbc                     = xyz
MDP

# ── npt.mdp ───────────────────────────────────────────────────────────────────
cat > npt.mdp << MDP
; ── NPT Equilibration — 100 ps @ ${TEMP} K / 1 bar, heavy atoms restrained ───
define                  = -DPOSRES

integrator              = md
nsteps                  = ${NPT_STEPS}  ; 100 ps  (dt = 0.002 ps)
dt                      = 0.002

nstxout                 = 500
nstvout                 = 500
nstenergy               = 500
nstlog                  = 500

cutoff-scheme           = Verlet
ns_type                 = grid
nstlist                 = 10
rcoulomb                = 1.0
rvdw                    = 1.0
coulombtype             = PME
pme_order               = 4
fourierspacing          = 0.16

; Temperature — V-rescale
tcoupl                  = V-rescale
tc-grps                 = Protein Non-Protein
tau_t                   = 0.1   0.1
ref_t                   = ${TEMP}  ${TEMP}

; Pressure — Parrinello-Rahman (correct NPT ensemble)
pcoupl                  = Parrinello-Rahman
pcoupltype              = isotropic
tau_p                   = 2.0
ref_p                   = 1.0
compressibility         = 4.5e-5
refcoords_scaling       = com

constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4

gen_vel                 = no
continuation            = yes        ; read velocities from NVT checkpoint

pbc                     = xyz
MDP

ok ".mdp files written: minim.mdp  nvt.mdp  npt.mdp"

# =============================================================================
# STEP 5 — Energy Minimisation
# =============================================================================
step "STEP 5: Energy Minimisation  [max ${EM_STEPS} steps]"

${GMX} grompp \
    -f minim.mdp \
    -c ionized.gro \
    -p topol.top \
    -o em.tpr \
    -maxwarn 2

# EM: only non-bonded on GPU (steep integrator doesn't support update/pme offload)
${GMX} mdrun -v -deffnm em \
    -ntomp "${CPU_THREADS}" \
    -gpu_id "${GPU_ID}" \
    -nb gpu -pin on

# Verify EM outcome — GROMACS splits the infinite-force error across two lines:
#   "...the force on at least one atom is not\nfinite."
# so we match on the single-line phrase that always appears just before it.
if grep -q "Energy minimization has stopped" em.log 2>/dev/null; then
    die "EM crashed: infinite force (Fmax = inf) — atom overlap in initial structure.\n\
  Check em.log for the 'Special Atom Distance Matrix' — look for SG–SG < 3 Å.\n\
  If a disulfide exists but no SSBOND record was injected, verify that\n\
  clean_input.pdb has valid ATOM records with correct residue names (CYS)."
fi
# "converged to machine precision" means EM hit the step-size floor WITHOUT
# reaching Fmax < emtol — the structure may still have bad geometry.
if grep -q "converged to machine precision" em.log 2>/dev/null \
        && grep -q "did not reach the requested Fmax" em.log 2>/dev/null; then
    warn "EM converged to machine precision but Fmax > emtol — structure may still be strained."
    warn "Check em.log and consider increasing EM_STEPS or reducing emstep in minim.mdp."
fi
ok "Energy minimisation done → em.gro"

# =============================================================================
# STEP 6 — NVT Equilibration
# =============================================================================
step "STEP 6: NVT Equilibration  [${NVT_STEPS} steps, ${TEMP} K]"

${GMX} grompp \
    -f nvt.mdp \
    -c em.gro \
    -r em.gro \
    -p topol.top \
    -o nvt.tpr \
    -maxwarn 2

${GMX} mdrun -v -deffnm nvt \
    -ntomp "${CPU_THREADS}" \
    -gpu_id "${GPU_ID}" \
    -nb gpu -pme gpu -bonded gpu -update gpu \
    -pin on

ok "NVT equilibration done → nvt.gro"

# =============================================================================
# STEP 7 — NPT Equilibration
# =============================================================================
step "STEP 7: NPT Equilibration  [${NPT_STEPS} steps, ${TEMP} K, 1 bar]"

${GMX} grompp \
    -f npt.mdp \
    -c nvt.gro \
    -r nvt.gro \
    -p topol.top \
    -t nvt.cpt \
    -o npt.tpr \
    -maxwarn 2

${GMX} mdrun -v -deffnm npt \
    -ntomp "${CPU_THREADS}" \
    -gpu_id "${GPU_ID}" \
    -nb gpu -pme gpu -bonded gpu -update gpu \
    -pin on

ok "NPT equilibration done → npt.gro"

# =============================================================================
# Summary
# =============================================================================
OUTABS="$(realpath .)"
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║  Pre-production pipeline COMPLETE                            ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
log "Output dir    : $OUTABS/"
log "Final struct  : $OUTABS/npt.gro"
log "Checkpoint    : $OUTABS/npt.cpt"
log "Topology      : $OUTABS/topol.top"
echo ""
echo -e "${BOLD}Next — production MD (create md.mdp, remove -DPOSRES, set nsteps):${NC}"
cat << 'NEXT'

  gmx_mpi grompp -f md.mdp -c npt.gro -r npt.gro -p topol.top -t npt.cpt -o md.tpr
  gmx_mpi mdrun  -v -deffnm md \
      -ntomp 8 -gpu_id 0 \
      -nb gpu -pme gpu -bonded gpu -update gpu -pin on

NEXT
