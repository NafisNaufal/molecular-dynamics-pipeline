# Nanobody MD Pipeline — E. faecalis Virulence Targets

End-to-end all-atom GROMACS pipeline: PDB preparation → energy minimisation → equilibration → **100 ns production MD** → trajectory analysis and charts.

**Targets:** Ace, EbpC, Esp (*Enterococcus faecalis* virulence proteins)  
**Nanobodies:** VHH scaffolds designed by RFdiffusion + ProteinMPNN (RFAntibody)  
**Hardware:** A100 80 GB GPU, HPC with `gmx_mpi` (GROMACS 2023.4 via micromamba)

---

## Repository layout

```
.
├── setup_gmx.sh       # Single-complex pre-production pipeline (one PDB → npt.gro)
├── run_all.sh         # Batch pre-production (all 9 PDBs)
├── run_production.sh  # Batch 100 ns production MD
├── run_analysis.sh    # GROMACS analysis tools + calls analyze.py
├── analyze.py         # Python charts (RMSD, RMSF, Rg, H-bonds, contacts, MM-PBSA)
├── run_mmpbsa.sh      # MM-GBSA binding free energies (gmx_MMPBSA)
└── design/            # Input PDB files (nanobody + target, docked)
    ├── ace_1.pdb   ace_2.pdb   ace_3.pdb
    ├── ebpc_1.pdb  ebpc_2.pdb  ebpc_3.pdb
    └── esp_1.pdb   esp_2.pdb   esp_3.pdb
```

Outputs:

```
runs/
└── ace_1/
    ├── npt.gro, npt.cpt, topol.top   ← pre-production outputs
    ├── md.tpr, md.xtc                ← 100 ns production
    ├── md_center.xtc                 ← centred trajectory
    ├── index.ndx                     ← Nanobody + Target groups
    └── analysis/
        ├── rmsd_backbone.xvg
        ├── rmsf_ca.xvg
        ├── gyrate.xvg
        ├── hbnum.xvg
        ├── mindist.xvg
        └── contacts.xvg

plots/
├── ace_1_rmsd_backbone.png           ← individual system charts
├── ace/rmsd_backbone_comparison.png  ← designs 1–3 overlaid per target
├── summary_bars.png
└── mmpbsa_binding_energy.png
```

---

## Simulation parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Force field | AMBER ff99SB-ILDN | Superior side-chain rotamer accuracy for protein–protein complexes [Lindorff-Larsen *et al.*, 2010] |
| Water model | TIP3P | Matched pairing for AMBER ff99SB-ILDN [Zhang *et al.*, 2018] |
| Box type | Rhombic dodecahedron | ~29% fewer water molecules than cubic box [Bekker, 1997] |
| Box margin | 1.2 nm | Sufficient buffering for the larger two-chain complex |
| Ions | 0.15 M NaCl + neutralise | Physiological ionic strength [Zhang *et al.*, 2010] |
| Temperature | 310 K | Human physiological temperature (37 °C) [Iyer *et al.*, 2025] |
| Thermostat | V-rescale, τ = 0.1 ps | Canonical ensemble [Bussi *et al.*, 2007] |
| Barostat (NPT) | Parrinello–Rahman, τ = 2.0 ps | Correct pressure fluctuations [Parrinello & Rahman, 1981] |
| Electrostatics | PME, spacing 0.16 nm | [Darden *et al.*, 1993] |
| Constraints | h-bonds, LINCS | Enables 2 fs timestep [Hess *et al.*, 1997] |

Full parameter justification with citations: [Notion page](https://app.notion.com/p/37a4798008008121a0d9feaa79fc74ae)

---

## Prerequisites

```bash
# GROMACS 2023.4 (MPI build) via micromamba
micromamba activate <your-gromacs-env>
gmx_mpi --version   # should print 2023.4
```

> The scripts use `gmx_mpi` throughout. To switch to a thread-MPI build, change  
> `GMX="gmx_mpi"` → `GMX="gmx"` at the top of both `setup_gmx.sh` and `run_all.sh`.

---

## Usage

### Install Python dependencies

```bash
# Analysis + plotting
micromamba install -y -c conda-forge numpy matplotlib pandas

# MM-PBSA (optional — needed only for binding free energies)
pip install gmx_MMPBSA
```

### Full workflow (4 steps)

```bash
# ── Step 1: Pre-production (EM + NVT + NPT) ──────────────────────────────────
nohup ./run_all.sh > batch_prep.log 2>&1 &
tail -f batch_prep.log

# ── Step 2: 100 ns production MD ─────────────────────────────────────────────
nohup ./run_production.sh > batch_prod.log 2>&1 &
tail -f batch_prod.log

# ── Step 3: Trajectory analysis + charts ─────────────────────────────────────
./run_analysis.sh               # generates ./plots/

# ── Step 4: MM-PBSA binding energies (optional, ~1–4 h/system) ───────────────
./run_mmpbsa.sh
python3 analyze.py              # re-run to include MM-PBSA bars
```

### Run a single complex

```bash
./setup_gmx.sh design/ace_1.pdb runs/ace_1
```

---

## Pipeline steps

```
Input PDB
    │
    ▼
[STEP 0]  Pre-process PDB
          • Strip HETATM / crystallographic waters
          • Warn on signal peptides / LPXTG anchors
          • Detect SSBOND records (auto-enables -ss in pdb2gmx)
    │
    ▼
[STEP 1]  gmx_mpi pdb2gmx
          AMBER ff99SB-ILDN, TIP3P, -merge all, -ignh
    │
    ▼
[STEP 2]  gmx_mpi editconf
          Dodecahedron box, 1.2 nm margin
    │
    ▼
[STEP 3]  gmx_mpi solvate
          TIP3P water (spc216.gro template)
    │
    ▼
[STEP 4]  gmx_mpi grompp + genion
          0.15 M NaCl + neutralise
    │
    ▼
[STEP 5]  Energy Minimisation (steepest descent, max 50,000 steps)
    │
    ▼
[STEP 6]  NVT Equilibration (100 ps, 310 K, position-restrained)
    │
    ▼
[STEP 7]  NPT Equilibration (100 ps, 310 K, 1 bar, position-restrained)
    │
    ▼
Output: npt.gro + npt.cpt + topol.top  ← ready for production MD
```

---

## Before running — PDB checklist

Because Ace, EbpC, and Esp are **cell-wall-anchored** surface proteins, make sure your input PDBs have these regions removed:

| Target | Signal peptide (N-term) | LPXTG anchor (C-term) |
|--------|------------------------|----------------------|
| Ace | ~aa 1–26 | ~last 60 aa |
| EbpC | ~aa 1–30 | ~last 35 aa |
| Esp | ~aa 1–26 | ~last 60 aa |

Quick check:
```bash
grep '^ATOM' design/ace_1.pdb | awk '{print $6}' | sort -n | head -3   # first residue
grep '^ATOM' design/ace_1.pdb | awk '{print $6}' | sort -n | tail -3   # last residue
```

---

## Charts produced by analyze.py

| Chart | Description |
|-------|-------------|
| `<stem>_rmsd_backbone.png` | Backbone RMSD vs time (ns) — raw + smoothed |
| `<stem>_rmsd_nanobody.png` | Nanobody-only RMSD vs time |
| `<stem>_rmsd_target.png` | Target-only RMSD vs time |
| `<stem>_rmsf.png` | Cα RMSF per residue |
| `<stem>_gyrate.png` | Radius of gyration vs time |
| `<stem>_hbonds.png` | Interface H-bond count vs time |
| `<stem>_mindist.png` | Interface minimum distance vs time |
| `<stem>_contacts.png` | Interface contact count (<0.35 nm) vs time |
| `<target>/rmsd_backbone_comparison.png` | 3 designs overlaid for one target |
| `<target>/rmsf_comparison.png` | RMSF overlay |
| `summary_bars.png` | All 9 systems: mean RMSD, Rg, H-bonds, contacts, min-dist |
| `mmpbsa_binding_energy.png` | ΔG binding bar chart (after run_mmpbsa.sh) |

All plots: 300 DPI PNG, dark/medium/light shading per design within each target colour family.

---

## GPU configuration

Edit these variables at the top of `setup_gmx.sh` to match your node:

```bash
GPU_ID=0        # GPU device index
CPU_THREADS=8   # OpenMP threads; increase if you have more cores per GPU (check: nproc)
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `gmx_mpi not found` | Run `micromamba activate <env>` first |
| `pdb2gmx` unrecognised residue | Check for non-standard residue names: `grep '^ATOM' file.pdb \| awk '{print $4}' \| sort -u` |
| `grompp` chain-break warning | Add `-missing` to `PDB2GMX_ARGS` in `setup_gmx.sh` |
| `genion` group not found | Replace `"SOL"` with the group number shown by `gmx_mpi make_ndx` |
| GPU offload error on mdrun | Remove `-update gpu` from the mdrun call in `setup_gmx.sh` (requires GROMACS ≥ 2021 with CUDA) |
| JupyterHub session killed | Always use `nohup ./run_all.sh > batch.log 2>&1 &` |
| `gmx_MMPBSA not found` | `pip install gmx_MMPBSA` then re-activate env |
| MM-PBSA group not found | `run_production.sh` must complete first to create `index.ndx` |
| RMSF shows only one chain | Expected — both chains are merged; chain boundary marked with dashed line |
| `matplotlib` import error | `micromamba install -y -c conda-forge numpy matplotlib pandas` |
