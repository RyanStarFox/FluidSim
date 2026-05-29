#!/usr/bin/env python3
"""
plot_energy.py — Energy & performance comparison tool
======================================================
Reads energy.csv files from FLIP / APIC / PolyPIC runs and generates
comparison plots for the final report (Phase III).

Usage:
    # Single self-check (auto-run by framework.py)
    python plot_energy.py single output/flip/ratio_970/dam_break/energy.csv -o check.png

    # Cross-algorithm comparison
    python plot_energy.py compare \\
        output/flip/ratio_970/dam_break/energy.csv \\
        output/apic/dam_break/energy.csv \\
        output/polypic/ratio_970/dam_break/energy.csv \\
        -o comparison/

    # Cross-ratio sweep (FLIP internal analysis)
    python plot_energy.py sweep output/flip/ -s dam_break -o ratio_analysis/

    # All-in-one
    python plot_energy.py all \\
        --flip output/flip/ratio_970/ \\
        --apic output/apic/ \\
        --polypic output/polypic/ratio_970/ \\
        -o report/
"""

import argparse
import csv
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Style constants
COLORS = {
    'flip':    '#e74c3c',   # red
    'apic':    '#2ecc71',   # green
    'polypic': '#3498db',   # blue
}
LINE_WIDTH = 1.2
ALPHA = 0.85


def load_csv(path: Path) -> dict:
    """Load an energy.csv file. Returns dict with keys: frame, time, ek, ms, vort."""
    data = {'frame': [], 'time': [], 'ek': [], 'ms': [], 'vort': []}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['frame'].append(int(row['frame']))
            data['time'].append(float(row['time']))
            data['ek'].append(float(row['kinetic_energy']))
            data['ms'].append(float(row['frame_time_ms']))
            data['vort'].append(float(row['total_vorticity']))
    return data


def plot_single(csv_path: Path, output_path: Path):
    """Quick self-check: energy + normalized energy decay."""
    data = load_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Kinetic energy
    axes[0].plot(data['time'], data['ek'], color=COLORS['flip'],
                 linewidth=LINE_WIDTH)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Kinetic Energy')
    axes[0].set_title('Kinetic Energy vs Time')
    axes[0].grid(True, alpha=0.3)

    # Normalized energy
    ek0 = data['ek'][0] if data['ek'][0] > 1e-8 else 1.0
    norm_ek = [e / ek0 for e in data['ek']]
    axes[1].plot(data['time'], norm_ek, color=COLORS['flip'],
                 linewidth=LINE_WIDTH)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('E_k / E_k(0)')
    axes[1].set_title('Normalized Energy Decay')
    axes[1].grid(True, alpha=0.3)

    # Frame time
    axes[2].plot(data['frame'], data['ms'], color=COLORS['flip'],
                 linewidth=LINE_WIDTH)
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Time (ms)')
    axes[2].set_title('Per-Frame Computation Time')
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"[plot_single] Saved: {output_path}")


def plot_compare(paths: dict, scene: str, output_dir: Path):
    """Cross-algorithm comparison: FLIP vs APIC vs PolyPIC.

    Args:
        paths: {'flip': Path, 'apic': Path, 'polypic': Path}
        scene: scene name for plot titles
        output_dir: directory for output PNGs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {name: load_csv(p) for name, p in paths.items() if p.exists()}

    if not datasets:
        print("[ERROR] No CSV files found to compare")
        return

    # --- Figure 1: Kinetic energy ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in datasets.items():
        ax.plot(data['time'], data['ek'], color=COLORS[name],
                linewidth=LINE_WIDTH, alpha=ALPHA, label=name.upper())
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Kinetic Energy')
    ax.set_title(f'Kinetic Energy Conservation — {scene}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f'{scene}_energy.png', dpi=200)
    plt.close(fig)

    # --- Figure 2: Normalized energy decay ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in datasets.items():
        ek0 = data['ek'][0] if data['ek'][0] > 1e-8 else 1.0
        norm = [e / ek0 for e in data['ek']]
        ax.plot(data['time'], norm, color=COLORS[name],
                linewidth=LINE_WIDTH, alpha=ALPHA, label=name.upper())
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('E_k / E_k(0)')
    ax.set_title(f'Normalized Energy Decay — {scene}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f'{scene}_energy_normalized.png', dpi=200)
    plt.close(fig)

    # --- Figure 3: Frame time ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in datasets.items():
        ax.plot(data['frame'], data['ms'], color=COLORS[name],
                linewidth=LINE_WIDTH, alpha=ALPHA, label=name.upper())
    ax.set_xlabel('Frame')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Per-Frame Computation Time — {scene}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f'{scene}_performance.png', dpi=200)
    plt.close(fig)

    # --- Figure 4: Vorticity ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in datasets.items():
        ax.plot(data['time'], data['vort'], color=COLORS[name],
                linewidth=LINE_WIDTH, alpha=ALPHA, label=name.upper())
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total |Vorticity|')
    ax.set_title(f'Vorticity Decay (Numerical Dissipation) — {scene}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f'{scene}_vorticity.png', dpi=200)
    plt.close(fig)

    # --- Print summary stats ---
    print(f"\n[summary] {scene} — Mean frame time:")
    for name, data in datasets.items():
        avg_ms = np.mean(data['ms'])
        total_s = sum(data['ms']) / 1000.0
        print(f"  {name.upper():8s}: {avg_ms:.1f} ms/frame  |  total: {total_s:.1f} s")

    print(f"\n[summary] {scene} — Energy retention at t=5s:")
    for name, data in datasets.items():
        ek0 = data['ek'][0] if data['ek'][0] > 1e-8 else 1.0
        # Find closest frame to t=5s
        idx = min(range(len(data['time'])), key=lambda i: abs(data['time'][i] - 5.0))
        retention = data['ek'][idx] / ek0 * 100
        print(f"  {name.upper():8s}: {retention:.1f}%")

    print(f"\n[plot_compare] All plots saved to: {output_dir}")


def plot_sweep(base_dir: Path, scene: str, output_dir: Path):
    """Cross-ratio sweep: compare different flip_ratio values.

    Args:
        base_dir: e.g., output/flip/
        scene: 'dam_break' or 'liquid_pouring'
        output_dir: output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all ratio directories
    ratio_dirs = sorted(base_dir.glob('ratio_*'))
    if not ratio_dirs:
        print(f"[ERROR] No ratio_* dirs found under {base_dir}")
        return

    ratios = []
    datasets = []
    for rd in ratio_dirs:
        csv_path = rd / scene / 'energy.csv'
        if csv_path.exists():
            ratio_val = int(rd.name.split('_')[1]) / 1000.0
            ratios.append(ratio_val)
            datasets.append(load_csv(csv_path))

    if not datasets:
        print("[ERROR] No CSV files found in ratio sweep")
        return

    # Color map from blue (PIC, ratio=0) to red (FLIP, ratio=1)
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(ratios)))

    # --- Energy comparison ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for ratio, data, c in zip(ratios, datasets, colors):
        ax.plot(data['time'], data['ek'], color=c,
                linewidth=LINE_WIDTH, alpha=ALPHA,
                label=f'flip_ratio={ratio:.2f}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Kinetic Energy')
    ax.set_title(f'FLIP Ratio Sweep: Energy Conservation — {scene}')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f'{scene}_ratio_sweep_energy.png', dpi=200)
    plt.close(fig)

    # --- Normalized energy ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for ratio, data, c in zip(ratios, datasets, colors):
        ek0 = data['ek'][0] if data['ek'][0] > 1e-8 else 1.0
        norm = [e / ek0 for e in data['ek']]
        ax.plot(data['time'], norm, color=c,
                linewidth=LINE_WIDTH, alpha=ALPHA,
                label=f'flip_ratio={ratio:.2f}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('E_k / E_k(0)')
    ax.set_title(f'FLIP Ratio Sweep: Normalized Decay — {scene}')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f'{scene}_ratio_sweep_normalized.png', dpi=200)
    plt.close(fig)

    # --- Vorticity comparison ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for ratio, data, c in zip(ratios, datasets, colors):
        ax.plot(data['time'], data['vort'], color=c,
                linewidth=LINE_WIDTH, alpha=ALPHA,
                label=f'flip_ratio={ratio:.2f}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total |Vorticity|')
    ax.set_title(f'FLIP Ratio Sweep: Vorticity — {scene}')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f'{scene}_ratio_sweep_vorticity.png', dpi=200)
    plt.close(fig)

    # --- Energy retention vs ratio (bar chart) ---
    fig, ax = plt.subplots(figsize=(8, 4))
    retentions = []
    for data in datasets:
        ek0 = data['ek'][0] if data['ek'][0] > 1e-8 else 1.0
        ek_final = data['ek'][-1]
        retentions.append(ek_final / ek0 * 100)
    bars = ax.bar([f'{r:.2f}' for r in ratios], retentions,
                  color=colors, edgecolor='#333', linewidth=0.5)
    ax.set_xlabel('FLIP Ratio')
    ax.set_ylabel('Final E_k / Initial E_k (%)')
    ax.set_title(f'Energy Retention by FLIP Ratio — {scene}')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, retentions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f'{scene}_ratio_retention.png', dpi=200)
    plt.close(fig)

    print(f"\n[summary] {scene} — FLIP ratio sweep:")
    for ratio, data in zip(ratios, datasets):
        ek0 = data['ek'][0] if data['ek'][0] > 1e-8 else 1.0
        ek_final = data['ek'][-1]
        avg_ms = np.mean(data['ms'])
        print(f"  ratio={ratio:.2f}: retention={ek_final/ek0*100:.1f}%  "
              f"|  avg {avg_ms:.1f} ms/frame")

    print(f"\n[plot_sweep] All plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Energy & performance comparison for fluid sim results'
    )
    subparsers = parser.add_subparsers(dest='command', help='sub-commands')

    # single: self-check plot for one CSV
    p_single = subparsers.add_parser('single', help='Single-run self-check')
    p_single.add_argument('csv', type=Path, help='Path to energy.csv')
    p_single.add_argument('-o', '--output', type=Path, default=Path('energy.png'),
                          help='Output PNG path')

    # compare: cross-algorithm comparison
    p_cmp = subparsers.add_parser('compare', help='Compare FLIP vs APIC vs PolyPIC')
    p_cmp.add_argument('csvs', type=Path, nargs='+',
                       help='CSV files (order: FLIP APIC PolyPIC)')
    p_cmp.add_argument('-l', '--labels', nargs='+',
                       default=['flip', 'apic', 'polypic'],
                       help='Labels for CSV files')
    p_cmp.add_argument('-s', '--scene', default='dam_break',
                       help='Scene name for plot titles')
    p_cmp.add_argument('-o', '--output', type=Path, default=Path('comparison/'),
                       help='Output directory')

    # sweep: cross-ratio analysis
    p_sw = subparsers.add_parser('sweep', help='FLIP ratio sweep analysis')
    p_sw.add_argument('base_dir', type=Path,
                      help='Base directory (e.g., output/flip/)')
    p_sw.add_argument('-s', '--scene', default='dam_break',
                      help='Scene name')
    p_sw.add_argument('-o', '--output', type=Path, default=Path('sweep/'),
                      help='Output directory')

    # all: combined report
    p_all = subparsers.add_parser('all', help='Full report with all three algorithms')
    p_all.add_argument('--flip', type=Path, required=True,
                       help='FLIP base dir (e.g., output/flip/ratio_970/)')
    p_all.add_argument('--apic', type=Path, default=None,
                       help='APIC base dir')
    p_all.add_argument('--polypic', type=Path, default=None,
                       help='PolyPIC base dir')
    p_all.add_argument('-s', '--scene', default='dam_break',
                       help='Scene name')
    p_all.add_argument('-o', '--output', type=Path, default=Path('report/'),
                       help='Output directory')

    args = parser.parse_args()

    if args.command == 'single':
        plot_single(args.csv, args.output)

    elif args.command == 'compare':
        if len(args.csvs) != len(args.labels):
            print(f"ERROR: Got {len(args.csvs)} CSV files but {len(args.labels)} labels")
            sys.exit(1)
        paths = dict(zip(args.labels, args.csvs))
        plot_compare(paths, args.scene, args.output)

    elif args.command == 'sweep':
        plot_sweep(args.base_dir, args.scene, args.output)

    elif args.command == 'all':
        output_dir = args.output
        # Cross-algorithm for each scene
        for scene in ['dam_break', 'liquid_pouring']:
            paths = {}
            flip_csv = args.flip / scene / 'energy.csv'
            if flip_csv.exists():
                paths['flip'] = flip_csv
            if args.apic:
                apic_csv = args.apic / scene / 'energy.csv'
                if apic_csv.exists():
                    paths['apic'] = apic_csv
            if args.polypic:
                polypic_csv = args.polypic / scene / 'energy.csv'
                if polypic_csv.exists():
                    paths['polypic'] = polypic_csv
            if len(paths) >= 2:
                plot_compare(paths, scene, output_dir / scene)
            else:
                print(f"[skip] {scene}: need >=2 CSV files, got {len(paths)}")

        # FLIP internal ratio sweep
        plot_sweep(args.flip.parent, 'dam_break', output_dir / 'flip_sweep' / 'dam_break')
        plot_sweep(args.flip.parent, 'liquid_pouring', output_dir / 'flip_sweep' / 'liquid_pouring')

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
