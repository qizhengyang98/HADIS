"""Measure everything the HADIS scheduler needs, on this machine's GPU.

The planner's decisions are only as good as the numbers it plans with, and those
are hardware-specific. Run this once on the GPU you intend to serve from; it
writes a profile that the controller and workers read in real-execution mode, so
no latency has to be assumed.

What it measures, in order

  1. host memory to hold every variant page-locked: the steady state a worker
     settles at and the higher peak reached while loading;
  2. model transition time: page-locked host <-> GPU, for the pairs the cascade
     can switch between, using the same pipelined swap the worker runs;
  3. inference latency per model per batch size, and the peak GPU memory each
     needs: the MILP's throughput (batch/latency), its latency constraint, and
     the largest batch that actually fits;
  4. router overhead per query;
  5. discriminator overhead per image.

Every variant is loaded and page-locked before anything is timed. That pays the
loading peak once, up front, and means the swap and inference costs are measured
in the state a worker actually serves in: all mirrors resident, one on the GPU.

Usage
  python experiments/profile_gpu.py                     # full profile
  python experiments/profile_gpu.py --quick             # 1 rep, small batches
  python experiments/profile_gpu.py --models sdxlltn sd35turbo
  python experiments/profile_gpu.py --out profiles/my_gpu.json

The four variants are 77.9 GiB of weights and measured 79.5 GB resident / 89 GB
peak while loading, so the node needs ~100 GB of RAM. The full profile takes
roughly an hour; --quick takes a few minutes and is enough to check the setup.
Results go to hadis/profiles/<gpu>.json unless --out says otherwise.
"""
import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
HADIS = os.path.join(ART, 'hadis')
sys.path.insert(0, os.path.join(HADIS, 'src'))
sys.path.insert(0, os.path.join(HADIS, 'src', 'worker'))

import config                                    # noqa: E402
from model_store import (ModelStore, GIB, release_heap,      # noqa: E402
                         detect_host_memory_gb, HOST_MEMORY_NOTE)
from model_loader import load_pipeline                       # noqa: E402

# Checkpoints for the four cascade variants, at the step counts they are served
# at. Kept here rather than in the serving code so that profiling and serving
# cannot disagree about what was measured.
MODEL_SPECS = config.MODEL_SPECS

PROMPT = 'a photograph of an astronaut riding a horse on a dusty road'


def log(msg):
    print(msg, flush=True)


def peak_host_gb():
    """Highest RSS this process has reached (never decreases)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def current_host_gb():
    """RSS right now, the steady state once loading is done."""
    for line in open('/proc/self/status'):
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) / 1048576
    return 0.0


def time_inference(pipe, name, batch, reps):
    """Median latency and peak memory for one (model, batch). None if it OOMs."""
    spec = MODEL_SPECS[name]
    prompts = [PROMPT] * batch
    kwargs = dict(prompt=prompts, num_inference_steps=spec['steps'])
    if spec['guidance'] is not None:
        kwargs['guidance_scale'] = spec['guidance']

    try:
        with torch.no_grad():                     # warm-up, not measured
            pipe(**kwargs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            with torch.no_grad():
                pipe(**kwargs)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return dict(latency_s=round(statistics.median(times), 4),
                    latency_min_s=round(min(times), 4),
                    per_image_s=round(statistics.median(times) / batch, 4),
                    peak_alloc_gib=round(torch.cuda.max_memory_allocated() / GIB, 2),
                    peak_reserved_gib=round(torch.cuda.max_memory_reserved() / GIB, 2),
                    reps=reps)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    except RuntimeError as exc:                   # older torch raises plain RuntimeError
        if 'out of memory' not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        return None


def profile_discriminator(reps, batches):
    """CLIP discriminator forward, per image."""
    sys.path.insert(0, os.path.join(HADIS, 'discriminator'))
    try:
        import clip
        from model import CLIPDiscriminator
    except ImportError as exc:
        return {'error': f'not measured: {exc}'}

    device = 'cuda'
    clip_model, preprocess = clip.load('ViT-B/32', device=device)
    disc = CLIPDiscriminator(clip_model.float()).to(device)
    # Head-only checkpoint by default; the encoder comes from clip.load above,
    # so strict=False is correct and its missing keys are expected.
    for name in ('CLIP_discriminator_head.pt', 'CLIP_discriminator.pt'):
        weights = os.path.join(HADIS, 'discriminator', name)
        if os.path.isfile(weights):
            disc.load_state_dict(torch.load(weights, map_location=device), strict=False)
            break
    disc.eval()

    out = {}
    for batch in batches:
        images = torch.randn(batch, 3, 224, 224, device=device)
        with torch.no_grad():
            disc(images)
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            with torch.no_grad():
                disc(images)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        out[str(batch)] = dict(batch_s=round(med, 5), per_image_ms=round(med / batch * 1000, 3))
    del disc, clip_model
    torch.cuda.empty_cache()
    return out


def profile_router(reps):
    """Rule-based router: score a prompt and compare against the threshold."""
    import numpy as np
    import pandas as pd
    feats = pd.read_csv(config.repo_path('router', 'prompt_features.csv')).iloc[:, 1:]
    weight = np.array([0.85, 0.32, 0.72, 0.99, 0.76, 0.07, 0.78, 0.01])
    scores = (((feats - feats.mean()) / feats.std()) @ weight).to_numpy()

    times = []
    for i in range(reps):                          # what isHardByRouter does per query
        t0 = time.perf_counter()
        ordered = np.sort(scores)
        idx = max(int(len(ordered) * (1 - 0.5)) - 1, 0)
        _ = scores[i % len(scores)] >= ordered[idx]
        times.append(time.perf_counter() - t0)
    return dict(per_query_ms=round(statistics.median(times) * 1000, 4),
                prompts=len(scores), reps=reps)


def load_one(name, args, store, profile):
    """Phase 1: load one variant from disk and page-lock its host mirror.

    Every model is mirrored before any inference runs, so the loading peak is
    paid once up front and the latencies measured afterwards are measured in the
    state a worker actually serves in: all mirrors resident, one on the GPU.
    """
    t0 = time.perf_counter()
    pipe = load_pipeline(name, args.cache_dir,
                         allow_download=not args.no_download, log=log)
    disk_s = time.perf_counter() - t0
    mirror = store.add(name, pipe)
    profile['models'][name] = {
        'repo': MODEL_SPECS[name]['repo'],
        'steps': MODEL_SPECS[name]['steps'],
        'params_gib': round(mirror.nbytes / GIB, 2),
        'buffers_gib': round(mirror.buffer_bytes / GIB, 3),
        'tensors': len(mirror),
        'pinned': mirror.pinned,
        'disk_to_host_s': round(disk_s, 1),
        'pin_s': round(mirror.pin_seconds, 1),
        'batches': {},
    }
    log(f"  {profile['models'][name]['params_gib']} GiB params + "
        f"{profile['models'][name]['buffers_gib']} GiB buffers, "
        f"disk->host {disk_s:.0f}s, pin {mirror.pin_seconds:.1f}s, "
        f"pinned={mirror.pinned}")


def measure_one(name, args, store, profile):
    """Phase 2: bring one mirror onto the GPU and time its batches."""
    entry = profile['models'][name]
    entry['host_to_gpu_s'] = round(store.activate(name), 3)
    log(f"  host->gpu {entry['host_to_gpu_s']:.2f}s")
    pipe = store.mirrors[name].pipe
    for batch in args.batch_sizes:
        result = time_inference(pipe, name, batch, args.reps)
        if result is None:
            log(f'  batch {batch:>2}: OOM, max feasible batch is below this')
            entry['batches'][str(batch)] = {'oom': True}
            break
        entry['batches'][str(batch)] = result
        log(f"  batch {batch:>2}: {result['latency_s']:8.3f} s "
            f"({result['per_image_s']:.3f} s/image, peak {result['peak_alloc_gib']} GiB)")

    feasible = [int(b) for b, v in entry['batches'].items() if not v.get('oom')]
    entry['max_batch'] = max(feasible) if feasible else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=list(MODEL_SPECS),
                    choices=list(MODEL_SPECS))
    ap.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 2, 4, 8])
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--quick', action='store_true',
                    help='1 rep, batches 1 and 2 only, a few minutes, to check the setup')
    ap.add_argument('--cache-dir', default=None,
                    help='the HuggingFace cache holding all four checkpoints. Read '
                         'offline first, so nothing already on disk is re-downloaded; '
                         'anything missing is fetched into it. Defaults to '
                         '$HADIS_MODEL_CACHE, else the HuggingFace default cache.')
    ap.add_argument('--no-download', action='store_true',
                    help='fail if a checkpoint is not already cached, instead of '
                         'downloading it (~78 GiB for the full set)')
    ap.add_argument('--chunk-mib', type=int, default=512, help='swap chunk size')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.quick:
        args.reps, args.batch_sizes = 1, [1, 2]
    if args.cache_dir is None:
        args.cache_dir = config.get_model_cache()
    if not torch.cuda.is_available():
        sys.exit('No CUDA device visible; this profile must be taken on the serving GPU.')

    gpu = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / GIB
    profile = {
        'meta': {
            'gpu': gpu,
            'vram_gib': round(total_vram, 2),

            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            # Flags only: the invocation carries absolute cache paths and the
            # machine's name, neither of which belongs in a shared profile.
            'command': ' '.join(
                ('<dir>' if i and sys.argv[i - 1] in ('--cache-dir', '--out')
                 else os.path.basename(a) if i == 0 else a)
                for i, a in enumerate(sys.argv)),
        },
        'models': {}, 'swap': {}, 'discriminator': {}, 'router': {},
    }
    log(f'GPU: {gpu} ({total_vram:.1f} GiB) | torch {torch.__version__}')

    # Mirrors are always pinned; see model_store for why pageable was rejected.
    host_gb, mem_source = detect_host_memory_gb()
    log(f'Host memory: {host_gb:.0f} GB ({mem_source}):  {HOST_MEMORY_NOTE}')
    if host_gb and host_gb < 100:
        log(f'  WARNING: {host_gb:.0f} GB may not be enough; the mirrors alone are '
            f'~80 GB and loading needs one model of headroom on top.')
    profile['meta'].update(host_memory_gb=round(host_gb, 1) if host_gb else None,
                           host_memory_source=mem_source)

    # --- 1: load and page-lock every variant, and measure what that costs ---
    #
    # Largest first. Peak host memory while mirroring a model is everything
    # already page-locked, plus that model twice over (the loaded copy and the
    # slab being built from it), so the last model loaded sets the peak and the
    # biggest one last would be the worst order.
    load_order = sorted(args.models,
                        key=lambda n: MODEL_SPECS[n].get('approx_gib', 0), reverse=True)
    log(f'\nload order (largest first): {" -> ".join(load_order)}')
    store = ModelStore(chunk_bytes=args.chunk_mib * 1024 ** 2)
    for name in load_order:
        log(f'\n== loading {name} ==')
        try:
            load_one(name, args, store, profile)
        except Exception as exc:               # one model must not lose the profile
            import traceback
            log(f'  FAILED: {type(exc).__name__}: {exc}')
            traceback.print_exc()
            profile['models'][name] = {'error': f'{type(exc).__name__}: {exc}'}
        # Hand the loaded copy's pages back before starting the next model --
        # otherwise glibc's arena carries them and the peak compounds.
        release_heap()
        log(f'  host RSS now {current_host_gb():.1f} GB '
            f'(peak {peak_host_gb():.1f} GB)')

    # What a worker settles at once every mirror is built: this, not the model
    # bytes and not the loading peak, is what a node has to sustain. Recorded
    # here, before anything else allocates, so it measures the mirrors alone.
    release_heap()
    profile['meta']['model_bytes_gib'] = round(store.total_host_bytes() / GIB, 2)
    profile['meta']['steady_host_rss_gib'] = round(current_host_gb(), 1)
    profile['meta']['peak_host_rss_gib'] = round(peak_host_gb(), 1)
    log(f"\nall mirrors resident: {profile['meta']['steady_host_rss_gib']:.1f} GB "
        f"(peak {profile['meta']['peak_host_rss_gib']:.1f} GB while loading)")

    # --- 2: transition cost between the models the cascade switches between --
    if len(store.mirrors) > 1:
        log('\n== model transitions (pipelined, page-locked) ==')
        names = [n for n in args.models if n in store.mirrors]
        for i, a_name in enumerate(names):
            for b_name in names[i + 1:]:
                store.activate(a_name)
                t_ab = store.activate(b_name)
                t_ba = store.activate(a_name)
                profile['swap'][f'{a_name}->{b_name}'] = round(t_ab, 3)
                profile['swap'][f'{b_name}->{a_name}'] = round(t_ba, 3)
                log(f'  {a_name:>10} <-> {b_name:<10} '
                    f'{t_ab * 1000:7.0f} ms / {t_ba * 1000:7.0f} ms')

    # --- 3: batched inference latency, with every mirror resident ------------
    for name in args.models:
        if name not in store.mirrors:
            continue
        log(f'\n== {name} ==')
        try:
            measure_one(name, args, store, profile)
        except Exception as exc:
            import traceback
            log(f'  FAILED: {type(exc).__name__}: {exc}')
            traceback.print_exc()
            profile['models'][name]['error'] = f'{type(exc).__name__}: {exc}'
            torch.cuda.empty_cache()

    # --- 4: the per-query overheads -----------------------------------------
    log('\n== router ==')
    profile['router'] = profile_router(reps=200)
    log(f"  {profile['router']['per_query_ms']} ms per query")

    log('\n== discriminator ==')
    profile['discriminator'] = profile_discriminator(
        reps=max(args.reps, 3), batches=[b for b in args.batch_sizes if b <= 8])
    for b, v in profile['discriminator'].items():
        if isinstance(v, dict):
            log(f"  batch {b:>2}: {v['per_image_ms']} ms per image")

    # Everything a worker holds, not just the mirrors: CLIP and the router
    # tables are resident during serving too.
    release_heap()
    profile['meta']['final_host_rss_gib'] = round(current_host_gb(), 1)
    profile['meta']['final_peak_host_rss_gib'] = round(peak_host_gb(), 1)

    out = args.out or os.path.join(HADIS, 'profiles',
                                   gpu.lower().replace(' ', '_').replace('/', '_') + '.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(profile, f, indent=2)
    log(f'\nWrote {out}')
    log(f"\n  model weights            : {profile['meta']['model_bytes_gib']:.1f} GiB")
    log(f"  resident once loaded     : {profile['meta']['steady_host_rss_gib']:.1f} GB"
        f"   <- what a worker node must sustain")
    log(f"  peak while loading       : {profile['meta']['peak_host_rss_gib']:.1f} GB")
    log(f"  with CLIP + router too   : {profile['meta']['final_host_rss_gib']:.1f} GB"
        f" (peak {profile['meta']['final_peak_host_rss_gib']:.1f} GB)")


if __name__ == '__main__':
    main()
