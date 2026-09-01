import math
from collections import defaultdict


# ── Trade-off knob ────────────────────────────────────────────────────────── #

# QUALITY_BIAS ∈ [0.0, 1.0]
#   0.0  → SLO-first:     prefer lighter models, minimise SLO violations at the
#           cost of higher FID (lower image quality).
#   1.0  → Quality-first: prefer heavier models, minimise FID at the cost of
#           higher SLO violation rate.
#   0.5  → Balanced (original behaviour).
#
# Set to 0.5, the documented original. INFaaS-Acc is a quality-first baseline --
# it deploys the highest-quality variant whose latency satisfies the SLO -- so an
# SLO-first bias does not represent it faithfully: the planner spends less time on
# the heavier variants and therefore violates the SLO less often than the design
# it stands in for.
QUALITY_BIAS = 0.5

# ── Derived constants (do not edit directly) ──────────────────────────────── #

# How much queuing pressure contributes to the latency estimate.
# High value → sensitive to queuing → steps down sooner → fewer SLO violations.
# Range: [0.05 (quality-first) … 0.40 (SLO-first)]
QUEUE_WEIGHT = 0.40 - 0.35 * QUALITY_BIAS

# Step DOWN when exp_lat > STEP_DOWN_MULTIPLIER × SLO.
# < 1.0 → preemptive downgrade before SLO is breached → fewer violations.
# > 1.0 → tolerate latency past the SLO to preserve quality → more violations.
# Range: [0.85 (SLO-first) … 1.15 (quality-first)]
STEP_DOWN_MULTIPLIER = 0.85 + 0.30 * QUALITY_BIAS

# Step UP when exp_lat < STEP_UP_THRESHOLD × SLO.
# High threshold → steps up more aggressively → better quality, more violations.
# Range: [0.80 (SLO-first) … 0.98 (quality-first)]
STEP_UP_THRESHOLD = 0.80 + 0.18 * QUALITY_BIAS

# Emergency jump to the lightest model when queue exceeds this depth.
# High value → tolerates longer queues before downgrading → better quality.
# Range: [20 (SLO-first) … 100 (quality-first)]
CRITICAL_QUEUE = int(20 + 80 * QUALITY_BIAS)

# ── Persistent state ──────────────────────────────────────────────────────── #

_state = {
    'current_idx': 0,   # index into by_fid; 0 = best quality (sd35large)
}


# ── Helpers ───────────────────────────────────────────────────────────────── #

def _expected_latency(model, bs, total_queue, num_workers, model_latency_values):
    """
    Estimated end-to-end latency = execution latency
                                  + QUEUE_WEIGHT × (queue / total throughput).
    The reduced QUEUE_WEIGHT tolerates mild queuing before triggering a switch.
    """
    lat = model_latency_values.get((model, bs))
    if lat is None:
        return float('inf')
    total_tp = num_workers * bs / lat
    q_delay = QUEUE_WEIGHT * total_queue / total_tp if total_tp > 0 else float('inf')
    return lat + q_delay


def _best_bs(model, model_latency_values, slo):
    """Largest batch size whose raw execution latency fits within the SLO."""
    for bs in reversed([1, 2, 4, 8, 16, 32]):
        lat = model_latency_values.get((model, bs))
        if lat is not None and lat <= slo:
            return bs
    return None


# ── Main function ─────────────────────────────────────────────────────────── #

def solve_infaas_accuracy(model_latency_values, total_workers, slo, ewma_demand, fid_weighting,
                          demand_per_model, queue_length_per_model):
    """
    INFaaS-Accuracy: queue-reactive single-model selection.

    Objective: minimise FID (use the best quality model possible), allowing
    mild SLO violations via the reduced QUEUE_WEIGHT on the latency estimate.

    Decision logic (evaluated every scheduling tick):

        total_queue > CRITICAL_QUEUE          → jump to sdxlltn immediately
        exp_lat > SLO                         → step DOWN one quality level
        exp_lat < STEP_UP_THRESHOLD × SLO    → step UP   one quality level
        otherwise                             → hold current model

    Output format matches solve_proteus_milp.
    """
    global _state

    if ewma_demand == 0:
        return None

    demand_weight = 1.0
    model_names = sorted(set(m for (m, _) in model_latency_values))
    # by_fid ascending: [sd35large(0), sd35med(1), sd35turbo(2), sdxlltn(3)]
    by_fid = sorted(model_names, key=lambda m: fid_weighting.get(m, float('inf')))
    worst_idx = len(by_fid) - 1
    num_avail = total_workers - 1

    _state['current_idx'] = min(_state['current_idx'], worst_idx)

    total_queue = sum(queue_length_per_model.get(m, 0) for m in model_names)

    # ── Resolve batch size for current model ──────────────────────────────── #
    current_model = by_fid[_state['current_idx']]
    bs = _best_bs(current_model, model_latency_values, slo)
    if bs is None:
        # Raw execution already exceeds SLO → step down immediately
        _state['current_idx'] = min(_state['current_idx'] + 1, worst_idx)
        current_model = by_fid[_state['current_idx']]
        bs = _best_bs(current_model, model_latency_values, slo)
        if bs is None:
            print("[INFaaS] No model can satisfy the SLO.")
            return None

    # ── Expected latency with reduced queue weight ────────────────────────── #
    exp_lat = _expected_latency(current_model, bs, total_queue, num_avail, model_latency_values)

    # ── Switching decision ────────────────────────────────────────────────── #
    prev_idx = _state['current_idx']

    if total_queue > CRITICAL_QUEUE:
        # Too many queries in queue: jump to sdxlltn to drain immediately.
        print(f'[INFaaS] CRITICAL queue={total_queue:.0f} > {CRITICAL_QUEUE}: jump to sdxlltn')
        _state['current_idx'] = worst_idx

    elif exp_lat > slo * STEP_DOWN_MULTIPLIER:
        # Expected latency exceeds the (bias-adjusted) step-down threshold.
        if _state['current_idx'] < worst_idx:
            _state['current_idx'] += 1
            print(f'[INFaaS] Step DOWN  {by_fid[prev_idx]} → {by_fid[_state["current_idx"]]} '
                  f'(exp={exp_lat:.2f}s > {slo * STEP_DOWN_MULTIPLIER:.2f}s)')

    elif exp_lat < slo * STEP_UP_THRESHOLD:
        # Expected latency well below SLO: step up one quality level.
        if _state['current_idx'] > 0:
            _state['current_idx'] -= 1
            print(f'[INFaaS] Step UP    {by_fid[prev_idx]} → {by_fid[_state["current_idx"]]} '
                  f'(exp={exp_lat:.2f}s < {slo * STEP_UP_THRESHOLD:.2f}s)')

    else:
        print(f'[INFaaS] Hold       {current_model} '
              f'(exp={exp_lat:.2f}s, queue={total_queue:.1f})')

    # ── Re-resolve after a potential index change ─────────────────────────── #
    current_model = by_fid[_state['current_idx']]
    bs = _best_bs(current_model, model_latency_values, slo)
    if bs is None:
        print(f'[INFaaS] Model {current_model} infeasible after switch.')
        return None

    print(f'[INFaaS] → model={current_model}, workers={num_avail}, batch={bs}, '
          f'FID={fid_weighting.get(current_model):.2f}, '
          f'exp_lat={exp_lat:.2f}s, queue={total_queue:.1f}')
    return {
        "device_allocation": {current_model: num_avail},
        "batch_sizes": {current_model: bs},
    }
