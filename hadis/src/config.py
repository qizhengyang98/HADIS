"""Global configuration for the HADIS testbed.

Two run modes, selected by the ``--profile-driven`` flag that every launch script
accepts (see ``start_*.sh``):

  profile-driven (E1)   simulated execution, model latencies scaled down 10x,
                        6 s SLO, 347 s trace.  16 workers on one node.
  real           (E2)   real diffusion models on real GPUs, paper latencies,
                        60 s SLO, ~3470 s trace.  4 workers, one per node.

Every mode-dependent constant lives here so that the components cannot drift
apart.  All components must be launched in the *same* mode; each one logs its
mode at startup.
"""
import json
import os


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# config.py lives in <repo>/src/, and every component is launched from
# <repo>/src/<component>/, so resolve everything against the repo root instead
# of relying on the caller's working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_path(*parts):
    return os.path.join(REPO_ROOT, *parts)


# --------------------------------------------------------------------------- #
# Cascade
# --------------------------------------------------------------------------- #

CAS_EXEC = 'multi'
CASCADE_MODEL_ORDER = ['sdxlltn', 'sd35turbo', 'sd35med', 'sd35large']

# Index of the light/heavy pair DiffServe (-ap 4) is fixed to, and of the models
# Clipper-Light (-ap 0) / Clipper-Heavy (-ap 1) pin every worker to.
DIFFSERVE_MODEL_PAIR = (1, 3)          # sd35turbo -> sd35large
CLIPPER_LIGHT_MODEL = 0                # sdxlltn
CLIPPER_HEAVY_MODEL = 3                # sd35large


def get_model_order():
    return CASCADE_MODEL_ORDER


def get_cas_exec():
    return CAS_EXEC


def set_cas_exec(cascade):
    global CAS_EXEC
    CAS_EXEC = cascade


# --------------------------------------------------------------------------- #
# Run mode
# --------------------------------------------------------------------------- #

PROFILE_DRIVEN = False
DO_SIMULATE = False

# Wall-clock compression applied in profile-driven mode.  The paper runs at
# 1.0; the testbed runs 10x faster so a ~1 hour trace replays in ~6 minutes.
PROFILE_DRIVEN_TIME_SCALE = 0.1


def get_profile_driven():
    return PROFILE_DRIVEN


def set_profile_driven(enabled=True):
    """Profile-driven mode implies simulated execution on the workers."""
    global PROFILE_DRIVEN, DO_SIMULATE
    PROFILE_DRIVEN = bool(enabled)
    if PROFILE_DRIVEN:
        DO_SIMULATE = True


def get_mode_name():
    return 'profile-driven' if PROFILE_DRIVEN else 'real'


def get_time_scale():
    return PROFILE_DRIVEN_TIME_SCALE if PROFILE_DRIVEN else 1.0


def get_do_simulate():
    return DO_SIMULATE


def set_do_simulate_true():
    global DO_SIMULATE
    DO_SIMULATE = True


def set_do_simulate_false():
    global DO_SIMULATE
    DO_SIMULATE = False


# --------------------------------------------------------------------------- #
# Profiled model latencies
# --------------------------------------------------------------------------- #

# Per-image latency on an L40S at the models' default step counts, as reported
# in the paper (SDXL-Lightning 2 steps, SD3.5-large-turbo 4, SD3.5-medium 50,
# SD3.5-large 50).  Profile-driven mode divides these by 10.
BASE_LATENCIES_SEC = {
    'sdxlltn': 0.5,
    'sd35turbo': 1.3,
    'sd35med': 13.0,
    'sd35large': 27.0,
}

ALLOWED_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

# Batching discount: a batch of b costs b * base * 0.9 (b > 1).  NOTE this is
# optimistic for the heavy models: the one real measurement we have
# (sd35_bench/results_partial_L40S_32GBram.txt: SD3.5-large, 1024^2, 28 steps)
# is *super*-linear: bs2 = 2.21x bs1, bs4 = 4.47x bs1.  Harmless in
# profile-driven mode (the simulator sleeps on the same table the planner
# optimises against) but it makes the planner optimistic in real mode, hence
# MAX_BATCH_SIZE_REAL below.
BATCH_DISCOUNT = 0.9

# Real mode only: SD3.5-large already peaks at 30.4 GB (bs1) / 36.0 GB (bs4) on
# a 46 GB L40S, so cap the heavy models before the MILP can pick a batch that
# will not fit.
MAX_BATCH_SIZE_REAL = {'sd35med': 2, 'sd35large': 2}


def _scaled_base(unit):
    """Base per-image latency for the active mode, in seconds or milliseconds.

    Rounded to 6 decimals so that the 10x scaling reproduces the literals the
    code used to hard-code (0.5 * 0.1 -> 0.05, not 0.05000000000000001).
    """
    factor = get_time_scale() * (1000.0 if unit == 'ms' else 1.0)
    return {m: round(v * factor, 6) for m, v in BASE_LATENCIES_SEC.items()}


def _batch_latency(base, batch):
    # Reproduces the original expression exactly, rounding included.
    if batch == 1:
        return round(base * batch, 1)
    return round(base * batch * BATCH_DISCOUNT, 1)


# --------------------------------------------------------------------------- #
# Real-execution checkpoints
# --------------------------------------------------------------------------- #

# The checkpoints the cascade actually runs in real mode. Shared by the worker
# (src/worker/model_re.py) and the profiler (experiments/profile_gpu.py) so a
# profile always describes the models that will serve.
#
# `guidance` is the classifier-free guidance scale; None means "leave the
# pipeline's default". The two turbo/distilled variants are guidance-free.
MODEL_SPECS = {
    'sdxlltn':   dict(repo='ByteDance/SDXL-Lightning', kind='sdxl_lightning',
                      ckpt='sdxl_lightning_2step_unet.safetensors', steps=2, guidance=0.0,
                      approx_gib=6.5),
    'sd35turbo': dict(repo='stabilityai/stable-diffusion-3.5-large-turbo', kind='sd3',
                      steps=4, guidance=0.0, approx_gib=27.4),
    'sd35med':   dict(repo='stabilityai/stable-diffusion-3.5-medium', kind='sd3',
                      steps=50, guidance=7.0, approx_gib=16.6),
    'sd35large': dict(repo='stabilityai/stable-diffusion-3.5-large', kind='sd3',
                      steps=50, guidance=7.0, approx_gib=27.4),
}


def get_model_cache():
    """The HuggingFace cache holding every checkpoint, or None for HF's default.

    The artifact assumes all four variants live in ONE cache directory. Set
    HADIS_MODEL_CACHE (or pass --cache / --cache-dir) to point at it; leave it
    unset to use HuggingFace's own cache, which is one directory too.
    """
    return os.environ.get('HADIS_MODEL_CACHE') or None


# --------------------------------------------------------------------------- #
# Measured profile (real mode only)
# --------------------------------------------------------------------------- #

# experiments/profile_gpu.py writes profiles/<gpu>.json on the serving hardware.
# When one is present, real mode plans with those measurements instead of the
# reference latencies above, because a reviewer's GPU is not ours. Selected with
# HADIS_PROFILE, or the single file in profiles/ if there is exactly one.
#
# This is consulted ONLY when PROFILE_DRIVEN is False, so the profile-driven
# experiment cannot reach it.
PROFILE_DIR = repo_path('profiles')
INFEASIBLE_LATENCY_S = 1e5     # a batch that does not fit: priced out of the MILP

_PROFILE_CACHE = {}


def get_measured_profile():
    """The measured profile for this machine, or None."""
    if PROFILE_DRIVEN:
        return None
    if 'p' in _PROFILE_CACHE:
        return _PROFILE_CACHE['p']

    path = os.environ.get('HADIS_PROFILE')
    if not path:
        found = sorted(f for f in os.listdir(PROFILE_DIR)
                       if f.endswith('.json')) if os.path.isdir(PROFILE_DIR) else []
        path = os.path.join(PROFILE_DIR, found[0]) if len(found) == 1 else None
    profile = None
    if path and os.path.isfile(path):
        with open(path) as f:
            profile = json.load(f)
        profile['_path'] = path
    _PROFILE_CACHE['p'] = profile
    return profile


def _measured_latency(model, batch):
    """Measured seconds for (model, batch), or None if there is no profile.

    Batches the profile could not fit are priced at INFEASIBLE_LATENCY_S so the
    planners' latency constraints exclude them without any solver change.
    """
    profile = get_measured_profile()
    if not profile:
        return None
    entry = profile.get('models', {}).get(model)
    if not entry:
        return None
    measured = {int(b): v['latency_s'] for b, v in entry.get('batches', {}).items()
                if not v.get('oom') and 'latency_s' in v}
    if not measured:
        return INFEASIBLE_LATENCY_S
    if batch in measured:
        return measured[batch]
    if batch > max(measured):
        return INFEASIBLE_LATENCY_S
    # below the largest measured batch but not measured: scale the nearest
    # per-image cost, which is how batching behaves within the fitted range
    nearest = min(measured, key=lambda b: abs(b - batch))
    return round(measured[nearest] / nearest * batch, 4)


def get_controller_runtimes():
    """{(model, batch): seconds}, the table the resource allocators plan with.

    NOTE: rounding to one decimal *in seconds* gives 100 ms granularity, which
    is coarse relative to the scaled-down light models. In profile-driven mode
    the planner therefore treats sdxlltn@bs1 as 0.1 s while a worker takes
    0.05 s, and sd35turbo@bs1 as 0.1 s against an actual 0.13 s, so the planner
    is conservative for the light models. At TIME_SCALE = 1 (real mode) the
    rounding is a no-op: 0.5 / 1.3 / 13 / 27 all survive it.
    """
    base = _scaled_base('s')
    table = {}
    for model in base:
        for batch in ALLOWED_BATCH_SIZES:
            measured = _measured_latency(model, batch)
            table[(model, batch)] = (measured if measured is not None
                                     else _batch_latency(base[model], batch))
    return table


def get_model_runtimes():
    """{(model, model, batch): milliseconds}, the table the workers use to
    sleep (simulated mode) and to decide whether a queued query would expire.

    Includes the interpolated odd batch sizes (3, 5, 6, 7) that serviceQueue
    can produce when the queue is shorter than the configured batch size.
    """
    base = _scaled_base('ms')
    runtimes = {}
    for model in base:
        for batch in ALLOWED_BATCH_SIZES:
            measured = _measured_latency(model, batch)
            runtimes[(model, model, batch)] = (measured * 1000 if measured is not None
                                               else _batch_latency(base[model], batch))
    for low, high in [(2, 4), (2, 8), (4, 8), (6, 8)]:
        for model in base:
            mid = int((low + high) / 2)
            runtimes[(model, model, mid)] = (runtimes[(model, model, low)] +
                                             runtimes[(model, model, high)]) / 2
    return runtimes


def get_max_batch_size(model):
    """Largest batch the planner may pick for a model in the active mode."""
    if PROFILE_DRIVEN:
        return max(ALLOWED_BATCH_SIZES)
    profile = get_measured_profile()
    if profile and model in profile.get('models', {}):
        measured = profile['models'][model].get('max_batch')
        if measured:
            return measured
    return MAX_BATCH_SIZE_REAL.get(model, max(ALLOWED_BATCH_SIZES))


def get_allowed_batch_sizes(model=None):
    if model is None:
        return list(ALLOWED_BATCH_SIZES)
    cap = get_max_batch_size(model)
    return [b for b in ALLOWED_BATCH_SIZES if b <= cap]


# --------------------------------------------------------------------------- #
# Control loop
# --------------------------------------------------------------------------- #

# How often the controller re-solves the allocation, in TRACE seconds.
#
# Its event loop ticks once per wall-clock second and re-plans every Nth tick.
# The interval must scale with the mode: 50 trace-seconds is 5 s of wall clock
# under profile-driven mode's 10x compression, but 50 s in real mode. The
# original code hardcoded "every 5 ticks", which is right only for the
# compressed run. In real mode it re-planned every 5 s, far faster than a
# model swap completes, so workers were reassigned while still mid-swap and the
# plan oscillated instead of converging.
ALLOC_INTERVAL_TRACE_SEC = 50


def get_alloc_interval_ticks(period_sec=1):
    """Event-loop ticks between re-allocations, for the active mode."""
    ticks = ALLOC_INTERVAL_TRACE_SEC * get_time_scale() / float(period_sec)
    return max(1, int(round(ticks)))


# --------------------------------------------------------------------------- #
# Application (latency SLO) and workload trace
# --------------------------------------------------------------------------- #

APP_JSON_PROFILE_DRIVEN = repo_path('traces', 'apps', 'multi_models.json')      # 6 s SLO
APP_JSON_REAL = repo_path('traces', 'apps', 'multi_models_60s.json')            # 60 s SLO

TRACE_PROFILE_DRIVEN = repo_path('traces', 'maf', 'multi_dynamic_testbed',
                                 'trace_64qps.txt')          # peaks at 64 QPS, 347 s
TRACE_REAL = repo_path('traces', 'maf', 'multi_realtime',
                       'trace_1_6qps.txt')                # peaks at 1.6 QPS, ~3470 s

PROMPT_FILE = repo_path('traces', 'text_imagenet1k_hr_5k.txt')


def get_app_json():
    return APP_JSON_PROFILE_DRIVEN if PROFILE_DRIVEN else APP_JSON_REAL


def get_default_trace():
    return TRACE_PROFILE_DRIVEN if PROFILE_DRIVEN else TRACE_REAL


def get_prompt_file():
    return PROMPT_FILE


def get_latency_slo_usec():
    """Latency SLO in microseconds, read from the same app JSON the controller
    and load balancer register, so the client cannot stamp a different one."""
    with open(get_app_json()) as f:
        return int(json.load(f)['latencySLOInMSec']) * 1000


# --------------------------------------------------------------------------- #
# Startup banner
# --------------------------------------------------------------------------- #

def mode_banner(component):
    return (f'[{component}] mode={get_mode_name()} '
            f'time_scale={get_time_scale()} '
            f'simulate={get_do_simulate()} '
            f'slo={get_latency_slo_usec() / 1e6:g}s '
            f'app={os.path.basename(get_app_json())}')
