"""Generate a request trace from the Microsoft Azure Functions demand shape.

Combines the two original steps (1_shape_preserving_transformation.py and
2_generate_trace_from_shape.py) into one parameterised script, so the same code
produces both the profile-driven and the real-execution traces.

The shape (original_invocations.csv, 347 points) is rescaled to [qps_min,
qps_max] and each point is expanded into Poisson arrivals over
`seconds_per_point` seconds. Output lines are "<arrival_ms>,<prompt_index>".
"""
import argparse
import os

import numpy as np
import pandas as pd

MAX_PROMPT_ID = 5000     # prompts available in traces/text_imagenet1k_hr_5k.txt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shape', default=os.path.join(os.path.dirname(__file__),
                                                    'original_invocations.csv'))
    ap.add_argument('--qps-min', type=float, required=True)
    ap.add_argument('--qps-max', type=float, required=True)
    ap.add_argument('--seconds-per-point', type=float, default=1.0,
                    help='wall-clock seconds each shape point spans')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    invocations = pd.read_csv(args.shape)['invocations'].values.astype(float)
    scaled = (invocations - invocations.min()) / (invocations.max() - invocations.min())
    qps = scaled * (args.qps_max - args.qps_min) + args.qps_min

    span_ms = args.seconds_per_point * 1000.0
    trace, offset = [], 0.0
    for rate in qps:
        requests = int(round(rate * args.seconds_per_point))
        if requests > 0:
            # Poisson arrivals within the interval: exponential inter-arrivals
            gaps = np.rint(rng.exponential(scale=span_ms / requests, size=requests))
            current = 0.0
            for gap in gaps:
                current += gap
                if current < span_ms:
                    trace.append(offset + current)
        offset += span_ms

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        for arrival in trace:
            f.write(f'{arrival},{rng.integers(1, MAX_PROMPT_ID)}\n')

    duration = offset / 1000.0
    print(f'{args.out}: {len(trace)} requests over {duration:.0f} s '
          f'({len(trace) / duration:.2f} QPS average, '
          f'{args.qps_min}-{args.qps_max} QPS demand)')


if __name__ == '__main__':
    main()
