"""Paper Figure 7: end-to-end comparison on the real-world trace.

Left column, top to bottom: demand, FID, SLO violation ratio over the trace.
Right column: the legend, then average FID and average SLO violation bars.

    python analysis/plot_e1_end2end.py                     # from results/logs/
    python analysis/plot_e1_end2end.py --logs-dir DIR      # any layout of runs

Runs may be plotted as they are collected: whatever is present in results/logs/
is drawn and the rest is left blank. Colours, markers, legend entries, bar slots
and axis ranges are fixed per system, so the figure keeps the same shape and
simply fills in -- run HADIS, plot; run DiffServe, plot again; and so on until
all six reproduce the published figure.

Curves are drawn from the collected logs.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
from brokenaxes import brokenaxes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse_logs as P

ART = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fixed break for the average-SLO panel, so it looks identical no matter which
# systems have been collected. The lower band holds the adaptive systems, the
# upper one holds a saturated static baseline. The top sits slightly above the
# highest tick so that a bar landing near it is not clipped by its own axis;
# --slo-ylims adjusts both bands.

SMOOTH_WINDOW = 5

SLO_YLIMS = ((0.0, 0.20), (0.65, 0.78))
SLO_TICKS_LOW = [0.0, 0.05, 0.10, 0.15, 0.20]
SLO_TICKS_HIGH = [0.65, 0.70, 0.75]

# Name, colour, marker and FID source for each system. 
SYSTEMS = [
    dict(key='clipper_light', label='Clipper-Light', color='#1f77b4', marker='v',
         fid='mix'),
    dict(key='clipper_heavy', label='Clipper-Heavy', color='#ff7f0e', marker='^',
         fid='mix'),
    dict(key='infaas', label='INFaaS-Acc', color='#17becf', marker='s',
         fid='infaas'),
    dict(key='proteus', label='Proteus', color='#2ca02c', marker='o',
         fid='mix'),
    dict(key='diffserve', label='DiffServe', color='#d62728', marker='x',
         fid='mix'),
    dict(key='hadis', label='HADIS', color='#9467bd', marker='*',
         fid='hadis'),
]

def find_offset(log_dir, length):
    """First row where the system had started serving, so that the window is
    not padded with idle startup rows. Falls back to 0."""
    import pandas as pd
    totals = pd.read_csv(os.path.join(log_dir, 'slo_timeouts_per_second.csv'))['total'].values
    served = np.nonzero(totals)[0]
    if len(served) == 0:
        return 0
    return int(min(served[0], max(len(totals) - length, 0)))


def load_system(spec, log_dir, length, table, offset=None):
    """Load one run, or return None if it has not been collected (yet).

    Tolerates the states a partially-finished sweep produces: a missing
    directory, a directory with no CSVs, and a run that was stopped early and so
    has fewer rows than requested.
    """
    import pandas as pd

    if not os.path.isdir(log_dir):
        print(f'  {spec["label"]:14} not run yet')
        return None

    slo_csv = os.path.join(log_dir, 'slo_timeouts_per_second.csv')
    if not os.path.isfile(slo_csv):
        print(f'  {spec["label"]:14} no results in {log_dir}')
        return None

    off = find_offset(log_dir, length) if offset is None else offset

    # A run stopped early has fewer rows; use the largest whole number of
    # 5-row windows that is available rather than failing.
    available = len(pd.read_csv(slo_csv)) - off
    usable = min(length, (available // P.ROWS_PER_POINT) * P.ROWS_PER_POINT)
    if usable < 5 * P.ROWS_PER_POINT:
        print(f'  {spec["label"]:14} too short to plot '
              f'({available} rows after offset {off})')
        return None
    if usable < length:
        print(f'  {spec["label"]:14} short run: using {usable} of {length} rows')
    length = usable

    try:
        return _load(spec, log_dir, length, table, off)
    except Exception as exc:                      # malformed or truncated CSVs
        print(f'  {spec["label"]:14} could not be parsed: {exc}')
        return None


def _load(spec, log_dir, length, table, off):
    slo = P.parse_slo(log_dir, off, length)
    if spec['fid'] == 'hadis':
        fid = P.parse_hadis_fid(log_dir, off, length, table=table)
    elif spec['fid'] == 'infaas':
        fid = P.parse_infaas_fid(log_dir, off, length)
    else:
        fid = P.parse_mix_fid(log_dir, off, length)

    fid = P.smooth(fid, SMOOTH_WINDOW)
    slo = P.smooth(slo, SMOOTH_WINDOW)
    print(f'  {spec["label"]:14} offset={off:3}  FID {fid.mean():6.2f}  '
          f'SLO {slo.mean():.4f}  ({len(fid)}/{len(slo)} points)')
    return dict(fid=fid, slo=slo)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs-dir', default=os.path.join(ART, 'results', 'logs'),
                    help='directory holding one subdirectory per system')
    ap.add_argument('--length', type=int, default=360,
                    help='log rows to use per run (multiple of 5)')
    ap.add_argument('--table', default='hybrid', help='cascade table HADIS was run with')
    ap.add_argument('--offset', type=int, default=None,
                    help='fixed start row; default is auto-detected per run')
    ap.add_argument('--slo-ylims', default=None,
                    help='average-SLO axis break, as "lo_bot,lo_top,hi_bot,hi_top" '
                         f'(default {",".join(str(v) for pair in SLO_YLIMS for v in pair)})')
    ap.add_argument('--out', default=os.path.join(ART, 'results', 'figures', 'fig7_end2end.png'))
    args = ap.parse_args()

    if args.slo_ylims:
        vals = [float(v) for v in args.slo_ylims.split(',')]
        if len(vals) != 4:
            sys.exit('--slo-ylims needs four numbers: lo_bot,lo_top,hi_bot,hi_top')
        globals()['SLO_YLIMS'] = ((vals[0], vals[1]), (vals[2], vals[3]))

    print('Loading runs:')
    data = {}
    for spec in SYSTEMS:
        log_dir = os.path.join(args.logs_dir, spec['key'])
        data[spec['key']] = load_system(spec, log_dir, args.length, args.table,
                                        args.offset)

    # Draw whatever has been collected so far and leave the rest blank, keeping
    # every system's colour, marker, legend entry and bar slot fixed so the
    # figure has the same shape as it fills in run by run.
    present = [s for s in SYSTEMS if data[s['key']]]
    missing = [s for s in SYSTEMS if not data[s['key']]]
    print(f'\nCollected {len(present)} of {len(SYSTEMS)} systems'
          + (f'; still to run: {", ".join(s["label"] for s in missing)}' if missing else ''))
    if not present:
        print('WARNING: no runs found -- drawing the empty template. '
              'Run a system, then experiments/collect_logs.sh <name>.')

    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(3, 2, width_ratios=[2.2, 0.8], height_ratios=[1, 1, 1],
                           figure=fig, hspace=0.05, wspace=0.175)
    ax_demand = fig.add_subplot(gs[0, 0])
    ax_fid = fig.add_subplot(gs[1, 0], sharex=ax_demand)
    ax_slo = fig.add_subplot(gs[2, 0], sharex=ax_demand)
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis('off')

    # Demand
    demand = P.demand_curve(os.path.join(ART, 'hadis', 'traces', 'maf', 'shape',
                                         'original_invocations.csv'))
    ax_demand.plot(np.arange(len(demand)) * 10, demand, color='grey', linewidth=2.5)
    ax_demand.set_ylabel('Demand (QPS)', fontsize=20)
    ax_demand.tick_params(axis='y', labelsize=18)
    ax_demand.grid(True)

    for spec in present:
        d = data[spec['key']]
        for ax, series in ((ax_fid, d['fid']), (ax_slo, d['slo'])):
            ax.plot(P.time_axis(series), series, label=spec['label'], linewidth=2,
                    color=spec['color'], marker=spec['marker'], markevery=2,
                    markersize=11 if spec['marker'] == '*' else 9)

    ax_fid.set_ylabel('FID', fontsize=20)
    ax_fid.set_yticks([20, 22, 24, 26, 28, 30])
    ax_fid.set_ylim(18.5, 30.7)
    ax_fid.tick_params(axis='y', labelsize=18)
    ax_fid.grid(True)

    ax_slo.set_ylim(-0.05, 1.0)
    ax_slo.set_ylabel('SLO Violation Ratio', fontsize=20)
    ax_slo.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_slo.tick_params(axis='both', labelsize=18)
    ax_slo.grid(True)
    ax_slo.set_xlabel('Time (S)', fontsize=20)
    ax_demand.set_xlim(-50, 3550)      # fixed, so the axis does not move as runs are added
    for ax in (ax_demand, ax_fid):
        plt.setp(ax.get_xticklabels(), visible=False)

    # One legend entry per system, always, so the key is stable while results
    # accumulate; systems with no logs yet are greyed out and marked.
    from matplotlib.lines import Line2D
    handles, labels = [], []
    for spec in SYSTEMS:
        if data[spec['key']]:
            handles.append(Line2D([], [], color=spec['color'], marker=spec['marker'],
                                  linewidth=2,
                                  markersize=11 if spec['marker'] == '*' else 9))
            labels.append(spec['label'])
        else:
            handles.append(Line2D([], [], color='lightgrey', marker=spec['marker'],
                                  linewidth=2, markersize=8, alpha=0.6))
            labels.append(f'{spec["label"]} (not run)')
    ax_legend.legend(handles, labels, loc='center', fontsize=15, ncol=1,
                     labelspacing=0.3, handletextpad=0.5, frameon=True)

    # Average FID
    ax_avg_fid = fig.add_subplot(gs[1, 1])
    for i, spec in enumerate(SYSTEMS):
        d = data[spec['key']]
        if d is not None:
            ax_avg_fid.bar(i, d['fid'].mean(), color=spec['color'],
                           hatch='//', edgecolor='black')
    ax_avg_fid.set_xlim(-0.7, len(SYSTEMS) - 0.3)
    ax_avg_fid.set_ylim(19.5, 30.3)
    ax_avg_fid.set_yticks([20, 22, 24, 26, 28, 30])
    ax_avg_fid.grid(axis='y', linestyle='--', linewidth=0.7)
    ax_avg_fid.tick_params(axis='y', labelsize=18)
    ax_avg_fid.set_xticks([])
    ax_avg_fid.set_title('Average Stats', fontsize=20)

    # Average SLO violation, on a fixed broken axis: Clipper-Heavy sits an order
    # of magnitude above everything else. The limits do not adapt to the data, so
    # the panel is identical whichever systems have been collected.
    avg_slo = {s['key']: data[s['key']]['slo'].mean() for s in present}
    (lo_bot, lo_top), (hi_bot, hi_top) = SLO_YLIMS

    for key, value in avg_slo.items():
        if lo_top < value < hi_bot:
            print(f'  WARNING: {key} average SLO {value:.3f} falls in the axis break '
                  f'({lo_top}-{hi_bot}); its bar will look clipped. '
                  f'Adjust with --slo-ylims.')
        elif value > hi_top:
            print(f'  WARNING: {key} average SLO {value:.3f} is above the axis top '
                  f'({hi_top}); its bar will be cut off. Adjust with --slo-ylims.')

    bax = brokenaxes(ylims=SLO_YLIMS, hspace=0.2, subplot_spec=gs[2, 1], fig=fig)
    for i, spec in enumerate(SYSTEMS):
        if spec['key'] in avg_slo:
            bax.bar(i, avg_slo[spec['key']], color=spec['color'],
                    edgecolor='black', hatch='//', width=0.8)

    bottom = bax.axs[-1]
    bottom.set_xlim(-0.7, len(SYSTEMS) - 0.3)
    bottom.set_xticks(np.arange(len(SYSTEMS)))
    bottom.set_xticklabels([s['label'] for s in SYSTEMS], ha='right', rotation=20, fontsize=14)
    for label in bottom.get_xticklabels():
        label.set_transform(label.get_transform() +
                            mtransforms.ScaledTranslation(0.3, 0, fig.dpi_scale_trans))

    bax.axs[-1].set_yticks(SLO_TICKS_LOW)
    bax.axs[-1].set_yticklabels([f'{t:g}' for t in SLO_TICKS_LOW])
    bax.axs[0].set_yticks(SLO_TICKS_HIGH)
    bax.axs[0].set_yticklabels([f'{t:g}' for t in SLO_TICKS_HIGH])

    first = True
    for ax in bax.axs:
        if first:
            ax.spines['top'].set_visible(True)
            first = False
        ax.spines['right'].set_visible(True)
        ax.grid(axis='y', linestyle='--', linewidth=0.7)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_xlim(-0.7, len(SYSTEMS) - 0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f'\nWrote {args.out}')

    # Average statistics
    summary = os.path.join(os.path.dirname(args.out), 'summary.csv')
    with open(summary, 'w') as f:
        f.write('system,avg_fid,avg_slo_violation\n')
        print(f'\n{"system":15} {"avg FID":>9} {"avg SLO":>10}')
        for spec in SYSTEMS:
            d = data[spec['key']]
            if d is None:
                f.write(f'{spec["label"]},,\n')
                print(f'{spec["label"]:15} {"-":>9} {"-":>10}   (not run)')
                continue
            fid, slo = d['fid'].mean(), d['slo'].mean()
            f.write(f'{spec["label"]},{fid:.3f},{slo:.5f}\n')
            print(f'{spec["label"]:15} {fid:9.2f} {slo:10.4f}')
    print(f'\nWrote {summary}')

    hadis_slo = data['hadis']['slo'].mean() if data.get('hadis') else None
    if hadis_slo:
        print('\nSLO violation reduction vs. HADIS:')
        for spec in present:
            if spec['key'] == 'hadis':
                continue
            other = data[spec['key']]['slo'].mean()
            ratio = other / hadis_slo if hadis_slo else float('inf')
            print(f'  {spec["label"]:15} {ratio:6.1f}x')


if __name__ == '__main__':
    main()
