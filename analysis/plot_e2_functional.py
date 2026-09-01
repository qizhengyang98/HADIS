"""E2 functional test: evidence that HADIS ran end to end on real GPUs.

E1 reproduces the paper's figure; E2 does not. Four workers cannot carry the
paper's load, so this plots what the functional test is actually there to show:
that real checkpoints served real queries, that the cascade routed between them,
that the controller re-planned during the run, and that model swaps happened and
what they cost.

    python analysis/plot_e2_functional.py                    # from hadis/logs/
    python analysis/plot_e2_functional.py --log-dir DIR
    python analysis/plot_e2_functional.py --out FILE.png

Panels are drawn from whatever the run produced; anything missing is left blank
with a note rather than shifting the others, so a partial run still plots.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ART = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ART, 'hadis', 'src'))

# One controller row per second, and E2 runs at wall-clock speed (no 10x
# compression), so a row is a second and the x axis is real minutes.
SEC_PER_ROW = 1
MODEL_COLOURS = {'sdxlltn': '#4C72B0', 'sd35turbo': '#DD8452',
                 'sd35med': '#55A868', 'sd35large': '#C44E52'}


def _read_csv(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return None
    return rows[0], rows[1:]


def _smooth(y, window):
    if window <= 1 or len(y) < window:
        return np.asarray(y, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(y, dtype=float), kernel, mode='same')


def _blank(ax, message):
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha='center', va='center',
            fontsize=9, color='0.45')
    ax.set_yticks([])


def panel_throughput(ax, log_dir, window):
    """Offered vs completed queries per second."""
    data = _read_csv(os.path.join(log_dir, 'slo_timeouts_per_second.csv'))
    if data is None:
        _blank(ax, 'slo_timeouts_per_second.csv not found')
        return 0
    _, rows = data
    arr = np.array([[float(v) for v in r[:4]] for r in rows if len(r) >= 4])
    t = np.arange(len(arr)) * SEC_PER_ROW / 60.0
    ax.plot(t, _smooth(arr[:, 3], window), color='0.35', lw=1.4, label='offered')
    ax.plot(t, _smooth(arr[:, 0], window), color='#4C72B0', lw=1.4, label='completed')
    ax.set_ylabel('queries / s')
    ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
    return len(arr)


def panel_slo(ax, log_dir, window):
    """Fraction of each second's queries that missed the SLO or were dropped."""
    data = _read_csv(os.path.join(log_dir, 'slo_timeouts_per_second.csv'))
    if data is None:
        _blank(ax, 'slo_timeouts_per_second.csv not found')
        return
    _, rows = data
    arr = np.array([[float(v) for v in r[:4]] for r in rows if len(r) >= 4])
    total = arr[:, 3]
    violated = arr[:, 1] + arr[:, 2]
    ratio = np.divide(violated, total, out=np.zeros(len(arr)), where=total != 0)
    t = np.arange(len(arr)) * SEC_PER_ROW / 60.0
    ax.plot(t, _smooth(ratio, window), color='#C44E52', lw=1.4)
    ax.set_ylabel('SLO violation')
    ax.set_ylim(-0.02, 1.02)
    served = int(arr[:, 0].sum())
    offered = int(total.sum())
    overall = 1 - served / offered if offered else 0.0
    ax.text(0.99, 0.92, f'{served}/{offered} within SLO  ({overall:.1%} violated)',
            transform=ax.transAxes, ha='right', va='top', fontsize=8, color='0.3')


def panel_mix(ax, log_dir, window):
    """Queries served per second by each model, showing the cascade actually routing."""
    data = _read_csv(os.path.join(log_dir, 'query_num_per_second.csv'))
    if data is None:
        _blank(ax, 'query_num_per_second.csv not found')
        return
    header, rows = data
    arr = np.array([[float(v) for v in r[:len(header)]] for r in rows
                    if len(r) >= len(header)])
    t = np.arange(len(arr)) * SEC_PER_ROW / 60.0
    for i, name in enumerate(header):
        ax.plot(t, _smooth(arr[:, i], window), lw=1.3, label=name,
                color=MODEL_COLOURS.get(name))
    ax.set_ylabel('queries / s\nby model')
    ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=4)


def panel_swaps(ax, log_dir, xmax):
    """Every model transition the workers performed, and what it cost."""
    paths = sorted(glob.glob(os.path.join(log_dir, 'model_swaps_*.csv')))
    points, labels = [], []
    for path in paths:
        data = _read_csv(path)
        if data is None:
            continue
        _, rows = data
        for r in rows:
            if len(r) >= 4:
                points.append((float(r[0]), float(r[3])))
                labels.append(f'{r[1]}->{r[2]}')
    if not points:
        _blank(ax, 'no model_swaps_*.csv, so no swap happened, or the workers '
                   'never changed model')
        ax.set_xlim(0, xmax)          # keep the frame identical to the panels above
        ax.set_ylabel('swap (s)')
        return
    t0 = min(p[0] for p in points)
    xs = [(p[0] - t0) / 60.0 for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, s=18, color='#8172B3', zorder=3)
    ax.set_ylabel('swap (s)')
    ax.set_ylim(0, max(ys) * 1.35)
    ax.text(0.99, 0.92, f'{len(ys)} swaps, median {np.median(ys):.2f} s, '
                        f'max {max(ys):.2f} s',
            transform=ax.transAxes, ha='right', va='top', fontsize=8, color='0.3')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log-dir', default=os.path.join(ART, 'hadis', 'logs'),
                    help='directory holding the controller CSVs and swap logs')
    ap.add_argument('--window', type=int, default=15,
                    help='smoothing window in seconds (default 15)')
    ap.add_argument('--out', default=os.path.join(ART, 'results', 'figures',
                                                  'e2_functional.png'))
    args = ap.parse_args()

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=False)
    n = panel_throughput(axes[0], args.log_dir, args.window)
    panel_slo(axes[1], args.log_dir, args.window)
    panel_mix(axes[2], args.log_dir, args.window)
    xmax = max(n * SEC_PER_ROW / 60.0, 1)
    panel_swaps(axes[3], args.log_dir, xmax)

    for ax in axes[:3]:
        ax.set_xlim(0, xmax)
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[3].set_xlabel('time (minutes)')
    axes[0].set_title('E2 functional test: real models on real GPUs', fontsize=11)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
