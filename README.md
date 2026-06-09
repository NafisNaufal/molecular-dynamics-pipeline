# Nanobody MD Pre-Production Pipeline

All-atom GROMACS pipeline to prepare nanobody–target complex PDB files (from **RFAntibody**) for molecular dynamics simulation — through energy minimisation, NVT equilibration, and NPT equilibration, stopping just before production MD.

**Targets:** Ace, EbpC, Esp (*Enterococcus faecalis* virulence proteins)  
**Nanobodies:** VHH scaffolds designed by RFdiffusion + ProteinMPNN (RFAntibody)  
**Hardware:** A100 80 GB GPU, HPC with `gmx_mpi` (GROMACS 2023.4 via micromamba)

---

## Repository layout

```
.
├── setup_gmx.sh   # Single-complex pipeline (one PDB → npt.gro)
├── run_all.sh     # Batch wrapper — loops all PDBs in ./design/
└── design/        # Input PDB files (nanobody + target, docked)
    ├── ace_1.pdb   ace_2.pdb   ace_3.pdb
    ├── ebpc_1.pdb  ebpc_2.pdb  ebpc_3.pdb
    └── esp_1.pdb   esp_2.pdb   esp_3.pdb
```

Outputs land in:

```
runs/
├── ace_1/    ace_2/    ace_3/      ← full GROMACS tree per complex
├── ebpc_1/   ebpc_2/   ebpc_3/
└── esp_1/    esp_2/    esp_3/
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

### Run all 9 complexes (recommended)

```bash
# JupyterHub: wrap in nohup so the job survives session disconnect
nohup ./run_all.sh > batch.log 2>&1 &
echo "PID: $!"

# Watch progress from another terminal / Jupyter cell
tail -f batch.log
```

`run_all.sh` defaults to `./design/` as the PDB directory. Pass an explicit path to override:

```bash
./run_all.sh /path/to/other/pdbs/
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

## Next step — production MD

After the pipeline completes, create `md.mdp` (based on `npt.mdp`, remove `-DPOSRES`, increase `nsteps`):

```bash
cd runs/ace_1/
gmx_mpi grompp -f md.mdp -c npt.gro -r npt.gro -p topol.top -t npt.cpt -o md.tpr
gmx_mpi mdrun  -v -deffnm md \
    -ntomp 8 -gpu_id 0 \
    -nb gpu -pme gpu -bonded gpu -update gpu -pin on
```

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
