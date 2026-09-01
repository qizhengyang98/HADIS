"""Turn a run's controller CSVs into the two time series the paper plots.

Two points worth knowing about how FID is derived:

* rows where the MILP found no feasible plan (empty ``models`` -- the controller
  sheds load onto the lightest model) are mapped to the cascade table's
  worst-FID configuration. Dropping them instead would corrupt the reshape and
  shift every later point;
* the query-mix parser reads the number of model columns from the CSV header.

Every run logs one row per second, and each plotted point aggregates 5 rows.
Wall-clock is 10x compressed in profile-driven mode, so a point spans 50 s of
paper time; ``TIME_STEP_SEC`` below is that factor.
"""
import ast
import csv
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'hadis', 'src'))
from tables import load_cascade_table  # noqa: E402

ROWS_PER_POINT = 5
TIME_STEP_SEC = 50          # 5 log rows x 10x time compression

MODEL_ORDER = ['sdxlltn', 'sd35turbo', 'sd35med', 'sd35large']

# Single-model FIDs, and the FID of a two-model cascade as a function of the
# fraction of queries served by the heavier model (11 points, 0.0 to 1.0 in
# steps of 0.1). Offline profiling output.
FID_SINGLE = {0: 29.76, 1: 25.23, 2: 20.61, 3: 19.95}
FID_CASCADE = {
    (0, 1): [29.76, 28.63, 27.84, 27.22, 26.64, 26.06, 25.57, 25.16, 25.02, 25.02, 25.23],
    (0, 2): [29.76, 27.47, 26.03, 24.89, 23.89, 22.97, 22.13, 21.46, 21.05, 20.76, 20.61],
    (0, 3): [29.76, 27.46, 25.93, 24.69, 23.59, 22.46, 21.63, 20.83, 20.36, 19.97, 19.95],
    (1, 2): [25.23, 23.85, 22.94, 22.37, 21.79, 21.41, 21.18, 20.98, 20.81, 20.77, 20.61],
    (1, 3): [25.23, 23.96, 23.13, 22.51, 21.93, 21.40, 20.99, 20.63, 20.34, 20.13, 19.95],
    (2, 3): [20.61, 20.26, 20.02, 19.75, 19.48, 19.42, 19.45, 19.44, 19.49, 19.68, 19.95],
}


def _interpolate(values):
    """Densify a curve to steps of 0.05 by inserting midpoints."""
    v = np.asarray(values, dtype=float)
    mids = ((v[:-1] + v[1:]) / 2).tolist()
    out = []
    for a, b in zip(v, mids):
        out.extend([a, round(b, 2)])
    out.append(v[-1])
    return out


FID_LOOKUP = {k: (_interpolate(v) if isinstance(v, list) else v)
              for k, v in {**FID_SINGLE, **FID_CASCADE}.items()}


def smooth(raw, window):
    """Moving average, 'valid' mode (shortens the series by window-1)."""
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(raw, dtype=float), kernel, mode='valid')


def _slice(rows, offset, length):
    window = rows[offset:offset + length]
    if len(window) < length:
        raise ValueError(f'need {length} rows from offset {offset}, got {len(window)}')
    return window


def parse_slo(log_dir, offset, length):
    """SLO violation ratio per 5-second window: (timeout + drop) / total."""
    results = pd.read_csv(os.path.join(log_dir, 'slo_timeouts_per_second.csv'))
    results = np.array(_slice(results.values.tolist(), offset, length))
    n = length // ROWS_PER_POINT
    violations = results[:, [1, 2]].reshape(n, ROWS_PER_POINT, 2).sum(axis=1).sum(axis=1)
    total = np.sum(results[:, 3].reshape(n, ROWS_PER_POINT), axis=1)
    return np.divide(violations, total,
                     out=np.zeros_like(violations, dtype=float),
                     where=total != 0)


def parse_hadis_fid(log_dir, offset, length, table='hybrid', verbose=True):
    """FID from the cascade configuration the controller selected each second."""
    comb_config, fid_config = load_cascade_table(table)
    worst = int(np.argmax(fid_config))     # all-lightest configuration

    configs = []
    with open(os.path.join(log_dir, 'cascade_config_per_second.csv')) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            raw = row[0].strip()
            models = tuple() if raw == '' else tuple(ast.literal_eval(raw))
            configs.append([models, float(row[1]), float(row[2])])
    configs = _slice(configs, offset, length)

    fids, shed, unmatched = [], 0, 0
    for cfg in configs:
        if cfg[0] == tuple():
            # MILP infeasible: the controller sheds load onto the lightest model
            # with no escalation, which is the worst-FID row of the table.
            fids.append(fid_config[worst])
            shed += 1
            continue
        for idx, entry in enumerate(comb_config):
            if (cfg[0] == tuple(int(m) for m in entry[0])
                    and cfg[1] == entry[1] and cfg[2] == entry[2]):
                fids.append(fid_config[idx])
                break
        else:
            fids.append(fid_config[worst])
            unmatched += 1

    if verbose and (shed or unmatched):
        print(f'    {os.path.basename(log_dir)}: {shed} shed row(s), '
              f'{unmatched} unmatched row(s) -> worst-FID configuration '
              f'({fid_config[worst]})')

    fids = np.asarray(fids).reshape(length // ROWS_PER_POINT, ROWS_PER_POINT)
    return fids.mean(axis=1)


def _query_mix(log_dir, offset, length):
    """Per 5-second window, the mix of models that served the queries.

    Returns a list of (key, share) where key is a model index (single model) or
    an ordered pair (two-model cascade) and share is the fraction served by the
    heavier of the two, rounded to 0.05.
    """
    df = pd.read_csv(os.path.join(log_dir, 'query_num_per_second.csv'))
    columns = list(df.columns)
    # This codebase logs all four models; a two-column log is read as the
    # (sd35turbo, sd35large) pair.
    model_idx = ([MODEL_ORDER.index(c) for c in columns] if set(columns) <= set(MODEL_ORDER)
                 else [1, 3])
    counts = np.array(_slice(df.values.tolist(), offset, length))
    counts = counts.reshape(-1, ROWS_PER_POINT, len(columns)).sum(axis=1)

    totals = counts.sum(axis=1, keepdims=True)
    shares = np.zeros_like(counts, dtype=float)
    nonzero = (totals != 0)[:, 0]
    shares[nonzero] = counts[nonzero] / totals[nonzero]
    shares = np.round(shares / 0.05) * 0.05

    mix = []
    for row in shares:
        active = np.where(row > 0)[0]
        if len(active) == 0:
            mix.append(((-1, -1), 0))
        elif len(active) == 1:
            mix.append((model_idx[active[0]], row[active[0]]))
        else:
            first, second = sorted(active)[:2]
            mix.append(((model_idx[first], model_idx[second]),
                        round(row[max(first, second)], 2)))
    return mix


def parse_mix_fid(log_dir, offset, length):
    """FID for the systems that log no cascade configuration (Proteus,
    DiffServe, Clipper): infer it from which models served the queries."""
    fids = []
    for key, share in _query_mix(log_dir, offset, length):
        if isinstance(key, tuple):
            if key == (-1, -1):
                fids.append(FID_LOOKUP[0])          # idle window
            else:
                curve = FID_LOOKUP[key]
                fids.append(curve[int(round(share / 0.05))])
        else:
            fids.append(FID_LOOKUP[key])
    return np.asarray(fids, dtype=float)


def parse_infaas_fid(log_dir, offset, length):
    """INFaaS-Acc hosts one variant at a time: use the dominant model."""
    df = pd.read_csv(os.path.join(log_dir, 'query_num_per_second.csv'))
    counts = np.array(_slice(df.values.tolist(), offset, length))
    counts = counts.reshape(length // ROWS_PER_POINT, ROWS_PER_POINT, len(df.columns)).sum(axis=1)
    return np.asarray([FID_LOOKUP[i] for i in np.argmax(counts, axis=1)], dtype=float)


def demand_curve(shape_csv, points=330, qps_min=1.0, qps_max=6.4):
    """The demand the trace was generated from, in paper QPS units."""
    invocations = pd.read_csv(shape_csv)['invocations'].values.astype(float)
    scaled = (invocations - invocations.min()) / (invocations.max() - invocations.min())
    return scaled[:points] * (qps_max - qps_min) + qps_min


def time_axis(series, step=TIME_STEP_SEC):
    return np.arange(len(series)) * step
