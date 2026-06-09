#!/usr/bin/env bash
# =============================================================================
#  run_production.sh — Batch production MD (100 ns) for all 9 complexes
# =============================================================================
#  Usage : ./run_production.sh
#  Expects: ./runs/<stem>/npt.gro + npt.cpt + topol.top  (from run_all.sh)
#  Outputs: ./runs/<stem>/md.xtc + md_center.xtc + index.ndx
#
#  Settings: 100 ns, 310 K, 1 bar, AMBER ff99SB-ILDN, A100 GPU
# =============================================================================
set -euo pipefail

GMX="gmx_mpi"
GPU_ID=0
CPU_THREADS=8
PROD_NS=100
PROD_STEPS=$(( PROD_NS * 1000000 / 2 ))   # nsteps = ns × 1e6 / dt(fs) = 50M

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${CYAN}[PROD]${NC}   $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}   $*" >&2; }
ok()   { echo -e "${GREEN}[ OK ]${NC}   $*"; }
die()  { echo -e "${RED}[FAIL]${NC}   $*" >&2; exit 1; }
sep()  { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════${NC}"; }

command -v "${GMX}" &>/dev/null || die "${GMX} not found — activate your micromamba env first."

# ── Collect completed NPT runs ────────────────────────────────────────────────
mapfile -t SYSTEMS < <(find ./runs -maxdepth 2 -name "npt.gro" | sed 's|/npt.gro||' | sort)
[[ "${#SYSTEMS[@]}" -eq 0 ]] && die "No completed NPT runs found in ./runs/\nRun ./run_all.sh first."

log "Found ${#SYSTEMS[@]} system(s) ready for production MD:"
for S in "${SYSTEMS[@]}"; do log "  $S"; done
echo ""

read -r -p "  Start ${#SYSTEMS[@]} production run(s) × ${PROD_NS} ns each? [y/N] " CONFIRM
[[ "${CONFIRM,,}" == "y" ]] || { log "Aborted."; exit 0; }

declare -a PASSED=() FAILED=()
BATCH_START=$(date +%s)

# =============================================================================
for SDIR in "${SYSTEMS[@]}"; do
    STEM="$(basename "$SDIR")"
    sep
    log "[$STEM] Starting ${PROD_NS} ns production MD"

    cd "$SDIR"

    # ── Write md.mdp ───────────────────────────────────────────────────────────
    cat > md.mdp << MDP
; ── Production MD — ${PROD_NS} ns @ 310 K / 1 bar ─────────────────────────────
integrator              = md
nsteps                  = ${PROD_STEPS}
dt                      = 0.002

; Output — compressed trajectory, no full-precision .trr
nstxout                 = 0
nstxout-compressed      = 5000        ; save coords every 10 ps → 10,000 frames
nstvout                 = 0
nstfout                 = 0
nstenergy               = 5000
nstlog                  = 5000

; Neighbour list
cutoff-scheme           = Verlet
ns_type                 = grid
nstlist                 = 10
rcoulomb                = 1.0
rvdw                    = 1.0

; Electrostatics
coulombtype             = PME
pme_order               = 4
fourierspacing          = 0.16

; Temperature
tcoupl                  = V-rescale
tc-grps                 = Protein Non-Protein
tau_t                   = 0.1   0.1
ref_t                   = 310   310

; Pressure
pcoupl                  = Parrinello-Rahman
pcoupltype              = isotropic
tau_p                   = 2.0
ref_p                   = 1.0
compressibility         = 4.5e-5
refcoords_scaling       = com

; Constraints
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4

; Continuation from NPT — no new velocities, no position restraints
gen_vel                 = no
continuation            = yes

pbc                     = xyz
MDP

    # ── grompp ─────────────────────────────────────────────────────────────────
    ${GMX} grompp \
        -f md.mdp \
        -c npt.gro \
        -r npt.gro \
        -p topol.top \
        -t npt.cpt \
        -o md.tpr \
        -maxwarn 2

    # ── mdrun ──────────────────────────────────────────────────────────────────
    T_START=$(date +%s)
    ${GMX} mdrun -v -deffnm md \
        -ntomp "${CPU_THREADS}" \
        -gpu_id "${GPU_ID}" \
        -nb gpu -pme gpu -bonded gpu \
        -pin on

    T_END=$(date +%s)
    ELAPSED=$(( T_END - T_START ))
    ok "[$STEM] mdrun done in $(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m"

    # ── Post-process: centre protein, remove PBC ───────────────────────────────
    log "[$STEM] Centring trajectory..."
    printf "Protein\nSystem\n" | ${GMX} trjconv \
        -s md.tpr \
        -f md.xtc \
        -o md_center.xtc \
        -center \
        -pbc mol \
        -ur compact 2>/dev/null
    ok "[$STEM] md_center.xtc written"

    # ── Create chain index (Nanobody + Target) ─────────────────────────────────
    log "[$STEM] Building chain index..."
    CLEAN_PDB="clean_input.pdb"

    if [[ -f "$CLEAN_PDB" ]]; then
        # Detect chain IDs and residue counts from original PDB
        CHAIN_DATA=$(python3 - "$CLEAN_PDB" << 'PYEOF'
import sys, collections
chains = collections.OrderedDict()
seen = {}
with open(sys.argv[1]) as f:
    for line in f:
        if not line.startswith('ATOM'):
            continue
        chain = line[21]
        resnum = line[22:26].strip()
        key = (chain, resnum)
        if key not in seen:
            seen[key] = True
            chains.setdefault(chain, 0)
            chains[chain] += 1
for c, n in chains.items():
    print(f"{c}:{n}")
PYEOF
        )

        readarray -t CHAIN_LINES <<< "$CHAIN_DATA"
        if [[ "${#CHAIN_LINES[@]}" -ge 2 ]]; then
            N_A=$(echo "${CHAIN_LINES[0]}" | cut -d: -f2)
            N_B=$(echo "${CHAIN_LINES[1]}" | cut -d: -f2)
            N_TOTAL=$(( N_A + N_B ))
            N_B_START=$(( N_A + 1 ))

            # Shorter chain = nanobody
            if [[ $N_A -le $N_B ]]; then
                NB_RANGE="1-${N_A}"
                TGT_RANGE="${N_B_START}-${N_TOTAL}"
            else
                NB_RANGE="${N_B_START}-${N_TOTAL}"
                TGT_RANGE="1-${N_A}"
            fi

            printf "ri %s\nname 20 Nanobody\nri %s\nname 21 Target\nq\n" \
                "$NB_RANGE" "$TGT_RANGE" \
                | ${GMX} make_ndx -f md.tpr -o index.ndx 2>/dev/null \
                && ok "[$STEM] index.ndx: Nanobody=ri${NB_RANGE}, Target=ri${TGT_RANGE}" \
                || warn "[$STEM] make_ndx failed — analysis needing index.ndx will be skipped"
        else
            warn "[$STEM] Only one chain detected in PDB — skipping chain index creation"
        fi
    else
        warn "[$STEM] clean_input.pdb not found — skipping chain index creation"
    fi

    PASSED+=("$STEM")
    cd - > /dev/null
done

# =============================================================================
BATCH_END=$(date +%s)
TOTAL=$(( BATCH_END - BATCH_START ))
sep
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║  PRODUCTION MD COMPLETE                                      ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
log "Total wall time : $(( TOTAL/3600 ))h $(( (TOTAL%3600)/60 ))m $(( TOTAL%60 ))s"
log "Completed       : ${#PASSED[@]} / ${#SYSTEMS[@]}"
echo ""
echo -e "${BOLD}Next — run analysis:${NC}"
echo "  ./run_analysis.sh"
echo ""
