#!/usr/bin/env python3
"""
analyze.py — MD trajectory analysis and visualization
======================================================
Reads GROMACS .xvg output from run_analysis.sh.
Generates 300 DPI PNG charts — both individual and overlaid per target.

Usage:
    python3 analyze.py [runs_dir] [plots_dir]

    runs_dir  : directory containing ./runs/<stem>/analysis/*.xvg  (default: ./runs)
    plots_dir : output directory for PNG charts                     (default: ./plots)

Install dependencies (micromamba):
    micromamba install -y -c conda-forge numpy matplotlib pandas

Chart output layout:
    plots/
    ├── <stem>_rmsd_backbone.png       ← individual per system
    ├── <stem>_rmsd_nanobody.png
    ├── <stem>_rmsf.png
    ├── <stem>_gyrate.png
    ├── <stem>_hbonds.png
    ├── <stem>_mindist.png
    ├── <stem>_contacts.png
    ├── ace/
    │   ├── rmsd_backbone_comparison.png   ← ace_1, ace_2, ace_3 overlaid
    │   ├── rmsf_comparison.png
    │   └── ...
    ├── ebpc/
    ├── esp/
    ├── summary_bars.png               ← all 9 systems, grouped bar chart
    └── mmpbsa_binding_energy.png      ← if run_mmpbsa.sh was run
"""

import sys
import os
import re
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
TARGETS = ['ace', 'ebpc', 'esp']
DESIGNS = ['1', '2', '3']

COLORS = {
    'ace_1':  '#0D47A1',  # deep blue
    'ace_2':  '#1E88E5',  # mid blue
    'ace_3':  '#90CAF9',  # light blue
    'ebpc_1': '#B71C1C',  # deep red
    'ebpc_2': '#E53935',  # mid red
    'ebpc_3': '#EF9A9A',  # light red
    'esp_1':  '#1B5E20',  # deep green
    'esp_2':  '#43A047',  # mid green
    'esp_3':  '#A5D6A7',  # light green
}

LINESTYLES = {'1': '-', '2': '--', '3': ':'}

MPLSTYLE = {
    'savefig.dpi': 300,
    'figure.dpi': 110,
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.4,
}


# ─────────────────────────────────────────────────────────────────────────────
#  XVG parser
# ─────────────────────────────────────────────────────────────────────────────
def read_xvg(path):
    """
    Parse a GROMACS .xvg file.
    Returns (x, y) as 1-D numpy arrays.  If multiple y-columns, y is 2-D;
    caller is responsible for slicing y[:, 0] when needed.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s[0] in '#@&':
                continue
            try:
                rows.append([float(v) for v in s.split()])
            except ValueError:
                continue
    if not rows:
        return np.array([]), np.array([])
    arr = np.asarray(rows, dtype=float)
    x = arr[:, 0]
    y = arr[:, 1] if arr.shape[1] == 2 else arr[:, 1:]
    return x, y


def first_col(y):
    """Return y as 1-D, taking column 0 if 2-D."""
    return y[:, 0] if y.ndim == 2 else y


def xvg_path(sdir, name):
    p = Path(sdir) / 'analysis' / name
    return str(p) if p.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
#  Smoothing
# ─────────────────────────────────────────────────────────────────────────────
def smooth(y, window=100):
    """Box-car (running mean) smoothing."""
    w = min(window, max(1, len(y) // 20))
    if w < 2:
        return y
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode='same')


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def plot_time_series(x, y, xlabel, ylabel, title, outpath, color,
                     raw_alpha=0.25, figsize=(8.5, 4)):
    """Single time-series plot with raw + smoothed overlay."""
    with plt.rc_context(MPLSTYLE):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, alpha=raw_alpha, color=color, linewidth=0.7, rasterized=True)
        ax.plot(x, smooth(y), color=color, linewidth=1.8, label='smoothed')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _save(fig, outpath)


def plot_time_series_overlay(series, xlabel, ylabel, title, outpath,
                              raw_alpha=0.15, figsize=(9.5, 4.5)):
    """Overlay time-series for multiple systems; series = [(label, x, y, color, ls)]."""
    with plt.rc_context(MPLSTYLE):
        fig, ax = plt.subplots(figsize=figsize)
        for label, x, y, color, ls in series:
            ax.plot(x, y, alpha=raw_alpha, color=color, linewidth=0.5, rasterized=True)
            ax.plot(x, smooth(y), color=color, linewidth=1.9,
                    linestyle=ls, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='upper right')
        _save(fig, outpath)


def plot_rmsf(x, y, title, outpath, color, nb_boundary=None, figsize=(10, 4)):
    """RMSF bar/line plot with optional chain-boundary marker."""
    with plt.rc_context(MPLSTYLE):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, color=color, linewidth=1.2)
        ax.fill_between(x, 0, y, alpha=0.15, color=color)
        if nb_boundary and nb_boundary < x.max():
            ax.axvline(nb_boundary, color='#616161', linestyle='--',
                       linewidth=1.0, label=f'chain boundary (res {int(nb_boundary)})')
            ax.legend()
        ax.set_xlabel('Residue number')
        ax.set_ylabel('RMSF (nm)')
        ax.set_title(title)
        _save(fig, outpath)


def plot_rmsf_overlay(series, title, outpath, nb_boundary=None, figsize=(11, 5)):
    """RMSF overlay for 3 designs on same axes."""
    with plt.rc_context(MPLSTYLE):
        fig, ax = plt.subplots(figsize=figsize)
        for label, x, y, color, ls in series:
            ax.plot(x, y, color=color, linewidth=1.5, linestyle=ls,
                    label=label, alpha=0.85)
        if nb_boundary:
            ax.axvline(nb_boundary, color='#9E9E9E', linestyle='--',
                       linewidth=1.0, label=f'chain boundary (res {int(nb_boundary)})')
        ax.set_xlabel('Residue number')
        ax.set_ylabel('RMSF (nm)')
        ax.set_title(title)
        ax.legend()
        _save(fig, outpath)


# ─────────────────────────────────────────────────────────────────────────────
#  Summary + MM-PBSA charts
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary(stats, outpath, figsize=(16, 5)):
    """Multi-panel bar chart — mean value ± std for each metric."""
    metric_defs = [
        ('rmsd_mean',     'Mean RMSD\n(nm)'),
        ('rg_mean',       'Mean Rg\n(nm)'),
        ('hbonds_mean',   'Mean H-bonds'),
        ('contacts_mean', 'Mean contacts\n(<0.35 nm)'),
        ('mindist_mean',  'Mean min-dist\n(nm)'),
    ]
    # Only keep metrics that have data
    metric_defs = [(k, lbl) for k, lbl in metric_defs
                   if any(k in s for s in stats.values())]
    if not metric_defs:
        return

    stems  = sorted(stats)
    colors = [COLORS.get(s, '#90A4AE') for s in stems]
    x      = np.arange(len(stems))
    xtick_labels = [s.replace('_', '\n') for s in stems]

    with plt.rc_context(MPLSTYLE):
        ncols = len(metric_defs)
        fig, axes = plt.subplots(1, ncols, figsize=figsize, sharey=False)
        if ncols == 1:
            axes = [axes]
        for ax, (key, ylabel) in zip(axes, metric_defs):
            vals = [stats[s].get(key, (0, 0))[0] for s in stems]
            errs = [stats[s].get(key, (0, 0))[1] for s in stems]
            ax.bar(x, vals, yerr=errs, color=colors,
                   capsize=3, edgecolor='white', linewidth=0.4,
                   error_kw={'elinewidth': 1, 'alpha': 0.6})
            ax.set_xticks(x)
            ax.set_xticklabels(xtick_labels, fontsize=7)
            ax.set_ylabel(ylabel)
        fig.suptitle('MD Analysis Summary — All Nanobody–Target Complexes',
                     fontsize=12, y=1.02)
        _save(fig, outpath)


def plot_mmpbsa(mmpbsa_data, outpath, figsize=(10, 5)):
    """Bar chart of MM-PBSA ΔG binding for all systems."""
    valid = {k: v for k, v in mmpbsa_data.items() if v[0] is not None}
    if not valid:
        return
    stems  = sorted(valid)
    vals   = [valid[s][0] for s in stems]
    errs   = [valid[s][1] or 0.0 for s in stems]
    colors = [COLORS.get(s, '#90A4AE') for s in stems]

    with plt.rc_context(MPLSTYLE):
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(range(len(stems)), vals, yerr=errs,
                      color=colors, capsize=4, edgecolor='white',
                      error_kw={'elinewidth': 1.2})
        ax.axhline(0, color='#212121', linewidth=0.8, linestyle='--', alpha=0.45)
        ax.set_xticks(range(len(stems)))
        ax.set_xticklabels([s.replace('_', '\n') for s in stems], fontsize=9)
        ax.set_ylabel('ΔG binding (kcal/mol)')
        ax.set_title('MM-PBSA Binding Free Energies\n'
                     'more negative = tighter binding', fontsize=12)
        for bar, val in zip(bars, vals):
            ypos = bar.get_height() + 0.8 if val >= 0 else bar.get_height() - 2.5
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        _save(fig, outpath)


# ─────────────────────────────────────────────────────────────────────────────
#  MM-PBSA result reader
# ─────────────────────────────────────────────────────────────────────────────
def read_mmpbsa(sdir):
    """
    Try to read ΔG from gmx_MMPBSA output files.
    Returns (mean_kcal, std_kcal) or (None, None).
    """
    candidates = [
        Path(sdir) / 'mmpbsa' / 'FINAL_RESULTS_MMPBSA.dat',
        Path(sdir) / 'mmpbsa' / 'FINAL_RESULTS_MMPBSA.csv',
        Path(sdir) / 'FINAL_RESULTS_MMPBSA.dat',
        Path(sdir) / 'FINAL_RESULTS_MMPBSA.csv',
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            text = p.read_text()
            m = re.search(r'TOTAL\s*=\s*([-\d.]+)\s*\+/-\s*([\d.]+)', text)
            if m:
                return float(m.group(1)), float(m.group(2))
            # Try CSV rows
            for line in text.splitlines():
                if 'TOTAL' in line and not line.startswith('#'):
                    nums = []
                    for tok in line.split(','):
                        try:
                            nums.append(float(tok.strip()))
                        except ValueError:
                            pass
                    if len(nums) >= 2:
                        return nums[-2], nums[-1]
        except Exception:
            pass
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    runs_dir  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./runs')
    plots_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('./plots')

    if not runs_dir.exists():
        sys.exit(f'ERROR: runs directory not found: {runs_dir}')

    plots_dir.mkdir(parents=True, exist_ok=True)
    for t in TARGETS:
        (plots_dir / t).mkdir(exist_ok=True)

    # ── Discover system directories ──────────────────────────────────────────
    all_stems = [f'{t}_{d}' for t in TARGETS for d in DESIGNS]
    sdirs = {s: runs_dir / s for s in all_stems if (runs_dir / s).is_dir()}
    if not sdirs:
        sys.exit(f'ERROR: no system directories found in {runs_dir}')
    print(f'Systems found: {", ".join(sorted(sdirs))}')

    # ── Load all XVG data ────────────────────────────────────────────────────
    data   = {}   # stem → {key: (x, y_1d)}
    stats  = {}   # stem → {metric_mean: (mean, std)}
    mmpbsa = {}   # stem → (dG_mean, dG_std)

    for stem, sdir in sdirs.items():
        data[stem]  = {}
        stats[stem] = {}

        def _load(xvg_name, key):
            f = xvg_path(sdir, xvg_name)
            if not f:
                return
            x, y = read_xvg(f)
            if not len(x):
                return
            y1 = first_col(y)
            data[stem][key] = (x, y1)
            stats[stem][f'{key}_mean'] = (float(np.mean(y1)), float(np.std(y1)))

        _load('rmsd_backbone.xvg', 'rmsd_bb')
        _load('rmsd_nanobody.xvg', 'rmsd_nb')
        _load('rmsd_target.xvg',   'rmsd_tgt')
        _load('gyrate.xvg',        'rg')
        _load('hbnum.xvg',         'hbonds')
        _load('mindist.xvg',       'mindist')
        _load('contacts.xvg',      'contacts')

        # RMSF — residue axis, not time
        f = xvg_path(sdir, 'rmsf_ca.xvg')
        if f:
            x, y = read_xvg(f)
            if len(x):
                data[stem]['rmsf'] = (x, first_col(y))

        dg, dg_std = read_mmpbsa(sdir)
        mmpbsa[stem] = (dg, dg_std)

    # ── Analysis table: (data_key, x_label, y_label, title, filename_stem) ──
    TIME_ANALYSES = [
        ('rmsd_bb',  'Time (ns)', 'RMSD (nm)',       'Backbone RMSD',             'rmsd_backbone'),
        ('rmsd_nb',  'Time (ns)', 'RMSD (nm)',       'Nanobody RMSD',             'rmsd_nanobody'),
        ('rmsd_tgt', 'Time (ns)', 'RMSD (nm)',       'Target RMSD',               'rmsd_target'),
        ('rg',       'Time (ns)', 'Rg (nm)',         'Radius of Gyration',        'gyrate'),
        ('hbonds',   'Time (ns)', 'H-bonds (count)', 'Interface H-bonds',         'hbonds'),
        ('mindist',  'Time (ns)', 'Min dist (nm)',   'Interface Min Distance',    'mindist'),
        ('contacts', 'Time (ns)', 'Contacts',        'Interface Contacts (<0.35 nm)', 'contacts'),
    ]

    # ── Individual plots ─────────────────────────────────────────────────────
    print('\n[1/3] Individual plots...')
    for stem, sdir in sorted(sdirs.items()):
        target, design = stem.rsplit('_', 1)
        color = COLORS.get(stem, '#78909C')
        ls    = LINESTYLES.get(design, '-')

        for key, xlabel, ylabel, title_sfx, fname in TIME_ANALYSES:
            if key not in data.get(stem, {}):
                continue
            x, y = data[stem][key]
            plot_time_series(
                x, y, xlabel, ylabel,
                f'{stem.upper()} — {title_sfx}',
                str(plots_dir / f'{stem}_{fname}.png'),
                color)

        if 'rmsf' in data.get(stem, {}):
            x, y = data[stem]['rmsf']
            plot_rmsf(x, y, f'{stem.upper()} — Cα RMSF',
                      str(plots_dir / f'{stem}_rmsf.png'), color)

    print(f'   → {plots_dir}/')

    # ── Comparison/overlay plots per target ──────────────────────────────────
    print('[2/3] Comparison plots...')
    for target in TARGETS:
        target_stems = [f'{target}_{d}' for d in DESIGNS]
        tdir = plots_dir / target

        for key, xlabel, ylabel, title_sfx, fname in TIME_ANALYSES:
            series = []
            for stem in target_stems:
                if key in data.get(stem, {}):
                    x, y  = data[stem][key]
                    d_num = stem.split('_')[-1]
                    series.append((stem, x, y, COLORS.get(stem, '#888'),
                                   LINESTYLES.get(d_num, '-')))
            if series:
                plot_time_series_overlay(
                    series, xlabel, ylabel,
                    f'{target.upper()} — {title_sfx} (designs 1–3)',
                    str(tdir / f'{fname}_comparison.png'))

        # RMSF overlay
        rmsf_series = []
        for stem in target_stems:
            if 'rmsf' in data.get(stem, {}):
                x, y  = data[stem]['rmsf']
                d_num = stem.split('_')[-1]
                rmsf_series.append((stem, x, y, COLORS.get(stem, '#888'),
                                    LINESTYLES.get(d_num, '-')))
        if rmsf_series:
            plot_rmsf_overlay(
                rmsf_series,
                f'{target.upper()} — Cα RMSF (designs 1–3)',
                str(tdir / 'rmsf_comparison.png'))

    print(f'   → {plots_dir}/<target>/')

    # ── Summary bar chart ────────────────────────────────────────────────────
    print('[3/3] Summary and MM-PBSA charts...')
    if stats:
        plot_summary(stats, str(plots_dir / 'summary_bars.png'))
        print(f'   summary_bars.png')

    if any(v[0] is not None for v in mmpbsa.values()):
        plot_mmpbsa(mmpbsa, str(plots_dir / 'mmpbsa_binding_energy.png'))
        print(f'   mmpbsa_binding_energy.png')
    else:
        print('   MM-PBSA results not found — run ./run_mmpbsa.sh first')

    # ── Print summary table ──────────────────────────────────────────────────
    print('\n' + '═' * 82)
    print(f'{"System":<10}  {"RMSD±σ (nm)":<14}  {"Rg±σ (nm)":<13}  '
          f'{"Hbonds±σ":<13}  {"Contacts±σ":<13}  {"ΔG (kcal/mol)"}')
    print('─' * 82)
    for stem in sorted(stats):
        s = stats[stem]
        def _fmt(k):
            if k in s: return f'{s[k][0]:.3f}±{s[k][1]:.3f}'
            return 'N/A'
        def _fmt0(k):
            if k in s: return f'{s[k][0]:.1f}±{s[k][1]:.1f}'
            return 'N/A'
        dg_val, dg_std = mmpbsa.get(stem, (None, None))
        dg_str = f'{dg_val:.1f}±{dg_std:.1f}' if dg_val is not None else 'N/A'
        print(f'{stem:<10}  {_fmt("rmsd_bb_mean"):<14}  {_fmt("rg_mean"):<13}  '
              f'{_fmt0("hbonds_mean"):<13}  {_fmt0("contacts_mean"):<13}  {dg_str}')
    print('═' * 82)
    print(f'\nAll charts → {plots_dir.resolve()}')


if __name__ == '__main__':
    main()
