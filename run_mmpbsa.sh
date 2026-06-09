#!/usr/bin/env bash
# =============================================================================
#  run_mmpbsa.sh — MM-PBSA binding free energy for all 9 complexes
# =============================================================================
#  Uses gmx_MMPBSA (https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA)
#
#  Install:
#      pip install gmx_MMPBSA
#
#  Usage : ./run_mmpbsa.sh
#  Expects: ./runs/<stem>/md.tpr + md_center.xtc + index.ndx
#  Outputs: ./runs/<stem>/mmpbsa/FINAL_RESULTS_MMPBSA.dat
#
#  Strategy: MM-GBSA on last 50 ns (frames 5001–10000), every 10th frame
#            → 500 frames per complex (~1–4 h per complex on A100 CPU side)
# =============================================================================
set -euo pipefail

GMX="gmx_mpi"
START_FRAME=5001      # skip first 50 ns (equilibration check)
END_FRAME=10000       # last frame of 100 ns trajectory at 10 ps/frame
INTERVAL=10           # use every 10th frame → 500 frames total

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${CYAN}[MMPBSA]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]  ${NC} $*" >&2; }
ok()   { echo -e "${GREEN}[ OK ]  ${NC} $*"; }
die()  { echo -e "${RED}[FAIL]  ${NC} $*" >&2; exit 1; }
sep()  { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════${NC}"; }

# ── Dependency checks ─────────────────────────────────────────────────────────
command -v "${GMX}"       &>/dev/null || die "${GMX} not found — activate your micromamba env."
command -v gmx_MMPBSA     &>/dev/null || {
    die "gmx_MMPBSA not found.\n\
Install it with:  pip install gmx_MMPBSA\n\
Then re-activate your environment and retry."
}
command -v python3        &>/dev/null || die "python3 not found."

log "gmx_MMPBSA version: $(gmx_MMPBSA --version 2>&1 | head -1 || echo unknown)"

# ── Collect systems ───────────────────────────────────────────────────────────
mapfile -t SYSTEMS < <(find ./runs -maxdepth 2 -name "md.tpr" | sed 's|/md.tpr||' | sort)
[[ "${#SYSTEMS[@]}" -eq 0 ]] && die "No production MD found. Run ./run_production.sh first."

log "Found ${#SYSTEMS[@]} system(s):"
for S in "${SYSTEMS[@]}"; do
    HAS_ALL=""; [[ -f "$S/md_center.xtc" && -f "$S/index.ndx" ]] && HAS_ALL="OK"
    log "  $S  $HAS_ALL"
done
echo ""
read -r -p "  Run MM-GBSA for ${#SYSTEMS[@]} system(s)? (1–4 h/system) [y/N] " CONFIRM
[[ "${CONFIRM,,}" == "y" ]] || { log "Aborted."; exit 0; }

BATCH_START=$(date +%s)
declare -a PASSED=() FAILED=()

# =============================================================================
for SDIR in "${SYSTEMS[@]}"; do
    STEM="$(basename "$SDIR")"
    sep
    log "[$STEM] MM-GBSA calculation..."

    cd "$SDIR"

    if [[ ! -f "md.tpr" ]]; then
        warn "[$STEM] md.tpr missing — skipping"
        FAILED+=("$STEM"); cd - >/dev/null; continue
    fi
    if [[ ! -f "md_center.xtc" && ! -f "md.xtc" ]]; then
        warn "[$STEM] trajectory not found — skipping"
        FAILED+=("$STEM"); cd - >/dev/null; continue
    fi
    TRAJ="md_center.xtc"; [[ ! -f "$TRAJ" ]] && TRAJ="md.xtc"

    if [[ ! -f "index.ndx" ]]; then
        warn "[$STEM] index.ndx missing — run_production.sh must complete first"
        FAILED+=("$STEM"); cd - >/dev/null; continue
    fi

    # ── Detect Nanobody and Target group IDs in index.ndx ─────────────────────
    GROUP_IDS=$(python3 - << 'PYEOF'
import re, sys
with open('index.ndx') as f:
    content = f.read()
groups = [m.group(1) for m in re.finditer(r'\[ (\S+) \]', content)]
try:
    nb_id  = groups.index('Nanobody')
    tgt_id = groups.index('Target')
    print(f"{nb_id} {tgt_id}")
except ValueError:
    print("NOT_FOUND")
PYEOF
    )

    if [[ "$GROUP_IDS" == "NOT_FOUND" ]]; then
        warn "[$STEM] Nanobody/Target groups not in index.ndx — rebuild index with run_production.sh"
        FAILED+=("$STEM"); cd - >/dev/null; continue
    fi

    NB_ID=$(echo "$GROUP_IDS"  | awk '{print $1}')
    TGT_ID=$(echo "$GROUP_IDS" | awk '{print $2}')
    log "[$STEM] Groups: Nanobody=$NB_ID, Target=$TGT_ID"

    # Receptor = Target (larger protein), Ligand = Nanobody
    REC_ID="$TGT_ID"
    LIG_ID="$NB_ID"

    mkdir -p mmpbsa
    cd mmpbsa

    # ── Write mmpbsa.in ────────────────────────────────────────────────────────
    cat > mmpbsa.in << MMPBSA_IN
&general
sys_name="${STEM}",
startframe=${START_FRAME},
endframe=${END_FRAME},
interval=${INTERVAL},
/

&gb
igb=2,
saltcon=0.150,
/
MMPBSA_IN

    # ── Run gmx_MMPBSA ─────────────────────────────────────────────────────────
    T_START=$(date +%s)
    gmx_MMPBSA -O \
        -i  mmpbsa.in \
        -cs ../md.tpr \
        -ci ../index.ndx \
        -cg "${REC_ID}" "${LIG_ID}" \
        -ct "../${TRAJ}" \
        -o  FINAL_RESULTS_MMPBSA.dat \
        -eo FINAL_RESULTS_MMPBSA.csv \
        2>&1 | tee mmpbsa.log

    T_END=$(date +%s)
    ELAPSED=$(( T_END - T_START ))

    if [[ -f "FINAL_RESULTS_MMPBSA.dat" ]]; then
        DG=$(grep -E "TOTAL\s*=" FINAL_RESULTS_MMPBSA.dat | tail -1 || echo "see .dat file")
        ok "[$STEM] Done in $(( ELAPSED/60 ))m — ΔG: $DG"
        PASSED+=("$STEM")
    else
        warn "[$STEM] FINAL_RESULTS_MMPBSA.dat not generated — check mmpbsa/mmpbsa.log"
        FAILED+=("$STEM")
    fi

    cd ../..   # back to repo root
done

# =============================================================================
BATCH_END=$(date +%s)
TOTAL=$(( BATCH_END - BATCH_START ))
sep
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║  MM-PBSA COMPLETE                                            ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
log "Wall time : $(( TOTAL/3600 ))h $(( (TOTAL%3600)/60 ))m"
log "Completed : ${#PASSED[@]} / ${#SYSTEMS[@]}"
[[ "${#FAILED[@]}" -gt 0 ]] && warn "Failed    : ${FAILED[*]}"
echo ""
echo "Results:  ./runs/<stem>/mmpbsa/FINAL_RESULTS_MMPBSA.dat"
echo "Charts :  python3 analyze.py  (re-run to include MM-PBSA bars)"
echo ""
