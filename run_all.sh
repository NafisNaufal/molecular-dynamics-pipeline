#!/usr/bin/env bash
# =============================================================================
#  run_all.sh — Batch GROMACS pre-production runner
#               for nanobody + E. faecalis target complexes
# =============================================================================
#  Usage : ./run_all.sh [pdb_directory]
#          pdb_directory defaults to ./design/  (where your 9 PDBs live)
#
#  Expects: a folder containing 9 PDB files (3 designs × Ace / EbpC / Esp)
#  Actual file names in ./design/:
#      ace_1.pdb    ace_2.pdb    ace_3.pdb
#      ebpc_1.pdb   ebpc_2.pdb   ebpc_3.pdb
#      esp_1.pdb    esp_2.pdb    esp_3.pdb
#
#  Output structure:
#      ./runs/ace_1/   (full GROMACS tree per complex)
#      ./runs/ace_2/
#      ...
#
#  Environment: A100 80GB, micromamba-managed GROMACS 2023.4, JupyterHub
#
#  ⚠️  Run from a persistent terminal to avoid JupyterHub session timeouts:
#       nohup ./run_all.sh > batch.log 2>&1 &
#       tail -f batch.log           # watch progress
# =============================================================================
set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${CYAN}[BATCH]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN] ${NC}  $*" >&2; }
ok()   { echo -e "${GREEN}[ OK ] ${NC}  $*"; }
die()  { echo -e "${RED}[FAIL] ${NC}  $*" >&2; exit 1; }
sep()  { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════${NC}"; }

# ── Parse arguments (default to ./design/) ───────────────────────────────────
PDB_DIR="$(realpath "${1:-./design}")"
[[ -d "$PDB_DIR" ]] || die "PDB directory not found: $PDB_DIR\n  Put your PDB files in ./design/ or pass a path: $0 <pdb_directory>"

# ── Locate this script's own directory (to find setup_gmx.sh) ────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_SCRIPT="${SCRIPT_DIR}/setup_gmx.sh"
[[ -x "$SETUP_SCRIPT" ]] || die "setup_gmx.sh not found or not executable at: $SETUP_SCRIPT"

# ── Require GROMACS ───────────────────────────────────────────────────────────
GMX="gmx_mpi"       # must match setup_gmx.sh; change to "gmx" for thread-MPI builds
command -v "${GMX}" &>/dev/null || {
    die "${GMX} not found. Activate your micromamba environment first:\n\
  micromamba activate <your-gromacs-env>"
}
GMX_VER=$(${GMX} --version 2>&1 | awk '/GROMACS version/{print $NF}')
log "GROMACS version : $GMX_VER"

# ── Collect PDB files ─────────────────────────────────────────────────────────
mapfile -t PDB_FILES < <(find "$PDB_DIR" -maxdepth 1 -name "*.pdb" | sort)
N_PDBS="${#PDB_FILES[@]}"
[[ "$N_PDBS" -eq 0 ]] && die "No .pdb files found in: $PDB_DIR"

log "Found $N_PDBS PDB file(s) in: $PDB_DIR"
log "Output root     : ./runs/"
echo ""

# ── Print job list ────────────────────────────────────────────────────────────
printf "  %-4s  %-40s  %s\n" "No." "PDB file" "Output directory"
printf "  %-4s  %-40s  %s\n" "----" "----------------------------------------" "-------------------"
IDX=0
for PDB in "${PDB_FILES[@]}"; do
    IDX=$(( IDX + 1 ))
    STEM="$(basename "$PDB" .pdb)"
    printf "  %-4s  %-40s  ./runs/%s/\n" "$IDX" "$(basename "$PDB")" "$STEM"
done
echo ""

# ── Confirm before starting ───────────────────────────────────────────────────
read -r -p "  Start all $N_PDBS simulation(s)? [y/N] " CONFIRM
[[ "${CONFIRM,,}" == "y" ]] || { log "Aborted by user."; exit 0; }

# ── Tracking arrays ───────────────────────────────────────────────────────────
declare -a PASSED=()
declare -a FAILED=()
BATCH_START=$(date +%s)

mkdir -p ./runs

# ── Main loop ─────────────────────────────────────────────────────────────────
IDX=0
for PDB in "${PDB_FILES[@]}"; do
    IDX=$(( IDX + 1 ))
    STEM="$(basename "$PDB" .pdb)"
    OUTDIR="./runs/${STEM}"

    sep
    log "[$IDX/$N_PDBS] Starting: $STEM"
    log "  Input  : $PDB"
    log "  Output : $OUTDIR"
    echo ""

    T_START=$(date +%s)

    # Run setup_gmx.sh; capture exit status without triggering set -e
    if bash "$SETUP_SCRIPT" "$PDB" "$OUTDIR" 2>&1 | tee "${OUTDIR}.log"; then
        T_END=$(date +%s)
        ELAPSED=$(( T_END - T_START ))
        ok "[$IDX/$N_PDBS] $STEM completed in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
        ok "  Log: ${OUTDIR}.log"
        PASSED+=("$STEM")
    else
        T_END=$(date +%s)
        ELAPSED=$(( T_END - T_START ))
        warn "[$IDX/$N_PDBS] $STEM FAILED after $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
        warn "  See log: ${OUTDIR}.log"
        FAILED+=("$STEM")
        # Continue with next PDB instead of aborting the whole batch
    fi
done

# ── Final summary ─────────────────────────────────────────────────────────────
BATCH_END=$(date +%s)
TOTAL=$(( BATCH_END - BATCH_START ))

sep
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║  BATCH RUN COMPLETE                                          ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
log "Total wall time  : $(( TOTAL / 3600 ))h $(( (TOTAL % 3600) / 60 ))m $(( TOTAL % 60 ))s"
log "Passed : ${#PASSED[@]} / $N_PDBS"
log "Failed : ${#FAILED[@]} / $N_PDBS"
echo ""

if [[ "${#PASSED[@]}" -gt 0 ]]; then
    echo -e "${GREEN}  ✔ Completed:${NC}"
    for S in "${PASSED[@]}"; do echo "      ./runs/${S}/  (log: ./runs/${S}.log)"; done
fi

if [[ "${#FAILED[@]}" -gt 0 ]]; then
    echo ""
    echo -e "${RED}  ✘ Failed:${NC}"
    for S in "${FAILED[@]}"; do echo "      ${S}  →  check ./runs/${S}.log"; done
    echo ""
    warn "Common failure causes for E. faecalis targets:"
    warn "  • Signal peptide or LPXTG anchor still present in PDB"
    warn "  • Unrecognised residue name (pdb2gmx error) — check with:"
    warn "      grep '^ATOM' <your.pdb> | awk '{print \$4}' | sort -u"
    warn "  • pdb2gmx chain-break warning — add -missing flag to PDB2GMX_ARGS in setup_gmx.sh"
    warn "  • GPU offload incompatibility — set update gpu → no in setup_gmx.sh mdrun calls"
fi

echo ""
echo -e "${BOLD}Files ready for production MD:${NC}"
for S in "${PASSED[@]}"; do
    echo "  ./runs/${S}/npt.gro   (+ npt.cpt, topol.top)"
done
echo ""
