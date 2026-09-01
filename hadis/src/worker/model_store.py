"""CPU-resident diffusion models with a pipelined CPU<->GPU swap.

A worker holds every model variant in pinned host memory and keeps exactly one
of them on the GPU. When the controller reassigns a worker, the outgoing model
is copied back to the host while the incoming one is copied in, chunk by chunk
on two CUDA streams, so that:

  * the two directions overlap on the full-duplex PCIe link, and
  * peak GPU memory is max(model) + one chunk rather than the sum, because each
    outgoing chunk's GPU tensors are released as soon as its copy completes.

Naively keeping two models resident does not fit: two SD3.5 mirrors need more
memory than a 46 GB card has once a CUDA context is up.

Used by the real-execution worker and by experiments/profile_gpu.py, so that the
swap cost reported by the profiler is measured on the same code path that runs
during an experiment.
"""
import ctypes
import gc
import logging
import os
import time

import torch

try:                                   # glibc only; harmless if unavailable
    _LIBC = ctypes.CDLL('libc.so.6')
except OSError:
    _LIBC = None


def gpu_free_gib():
    """(free, total) device memory in GiB, for the whole card rather than just torch's view."""
    try:
        free, total = torch.cuda.mem_get_info()
        return free / GIB, total / GIB
    except Exception:
        return 0.0, 0.0


def current_host_rss_gib():
    """This process's resident set right now, in GiB."""
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return 0.0


def release_heap():
    """Return freed CPU memory to the OS.

    Each model is loaded as ordinary CPU tensors before being mirrored into
    pinned memory. Python frees the originals, but glibc keeps the arena, so the
    process's RSS keeps the freed bytes, roughly 12 GB per model here. Under a
    cgroup that is charged to the job and gets it OOM-killed well before the
    pinned mirrors themselves would.
    """
    gc.collect()
    if _LIBC is not None:
        _LIBC.malloc_trim(0)

GIB = 1024 ** 3
DEFAULT_CHUNK_BYTES = 512 * 1024 ** 2      # 512 MiB; see profile_gpu.py --chunk-mib

# --------------------------------------------------------------------------- #
# Pinned host mirrors
# --------------------------------------------------------------------------- #
#
# The mirrors are always page-locked. Pageable host memory was measured as an
# alternative and rejected: it saves about 30 GB resident but makes every model
# transition roughly 4x slower, which is the cost that matters while a worker is
# not serving.
#
# Measured on an L40S node for the four variants (77.9 GiB of fp16 weights):
#
#     mirrors                  transition (mean/worst)  resident   loading peak
#     per-tensor pin_memory()      1.22 s / 1.40 s      115.3 GiB     125 GB
#     pageable (no pinning)        4.79 s / 8.89 s       85.2 GiB      91 GB
#     registered slab (current)    1.22 s / 1.40 s        79.5 GB      89 GB
#
# Pinning tensor by tensor cost far more than the model bytes suggest, because
# CUDA's caching host allocator rounds every request up to a power-of-two
# bucket: 6.46 GiB became 8, 16.60 GiB became 32, 27.43 GiB became 32. One slab
# per model does not fix that on its own, since it is still one rounded allocation.
# So each mirror is ordinary memory page-locked in place with cudaHostRegister,
# which does no rounding: 1.02x the weights instead of 1.48x, at the same DMA
# speed. On a 128 GB node that is the difference between completing and being
# OOM-killed on the fourth model.
#
# Pinned pages are unevictable, so a host that cannot hold the footprint is
# OOM-killed rather than swapped: size the node from HOST_MEMORY_NOTE below,
# not from the model sizes.
#
# Measured for the four cascade variants on an L40S; see profiles/<gpu>.json for
# this machine's numbers.
HOST_MEMORY_NOTE = (
    'four variants page-locked measured 79.5 GB resident, 89 GB peak while '
    'loading (77.9 GiB of weights, so 1.02x)')


def detect_host_memory_gb():
    """Host memory actually available to this process, and where that came from.

    Not /proc/meminfo alone: under a scheduler or container the process is
    capped well below the machine's RAM, and exceeding the cap is what gets it
    killed. This node reports 503 GB of RAM while allowing 128 GB.
    """
    candidates = []

    for path in ('/sys/fs/cgroup/memory.max',                    # cgroup v2
                 '/sys/fs/cgroup/memory/memory.limit_in_bytes'):  # cgroup v1
        try:
            with open(path) as f:
                raw = f.read().strip()
            if raw and raw != 'max':
                value = int(raw) / GIB
                if value < 1e6:            # v1 uses a huge sentinel for "no limit"
                    candidates.append((value, os.path.basename(path)))
        except (OSError, ValueError):
            pass

    slurm = os.environ.get('SLURM_MEM_PER_NODE')                  # MB
    if slurm and slurm.isdigit():
        candidates.append((int(slurm) / 1024, 'SLURM_MEM_PER_NODE'))

    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemTotal:'):
                candidates.append((int(line.split()[1]) / 1048576, 'MemTotal'))
                break
    except OSError:
        pass

    if not candidates:
        return None, 'unknown'
    return min(candidates)                 # the tightest limit is the binding one


def pipeline_modules(pipe):
    """The submodules of a diffusers pipeline that hold parameters worth moving."""
    names = ('transformer', 'unet', 'text_encoder', 'text_encoder_2',
             'text_encoder_3', 'vae')
    return [getattr(pipe, n) for n in names if getattr(pipe, n, None) is not None]


def pipeline_parameters(pipe):
    return [p for module in pipeline_modules(pipe) for p in module.parameters()]


def pipeline_buffers(pipe):
    """Registered buffers (CLIP position_ids, normalisation statistics, ...).

    These are not parameters, so they are not part of the swapped set, but the
    modules still index them during a forward pass: leaving them on the host
    while the weights are on the GPU raises "expected all tensors to be on the
    same device". They are small, a few hundred MB across all variants, so
    they are moved to the GPU once and stay there.
    """
    return [(module, name, buf)
            for module in pipeline_modules(pipe)
            for name, buf in module.named_buffers(recurse=True)
            if buf is not None]


class ModelMirror:
    """One model's parameters: a pinned host copy, and optionally a GPU copy.

    The host tensors are the authoritative storage. ``attach`` points the
    pipeline's parameters at whichever copy is currently live, so inference runs
    against the GPU tensors without the pipeline knowing a swap happened.
    """

    def __init__(self, name, pipe):
        self.name = name
        self.pipe = pipe
        self.params = pipeline_parameters(pipe)
        self.host = []
        self.gpu = None                     # list aligned with self.params, or None
        self.nbytes = sum(p.numel() * p.element_size() for p in self.params)

        t0 = time.perf_counter()
        self.slab = None
        self.registered = False
        self._mirror_into_slab()
        self.pin_seconds = time.perf_counter() - t0
        self.pinned = all(h.is_pinned() for h in self.host if h.numel())

        # Buffers are not swapped: park them on the GPU for the model's lifetime
        self.buffer_bytes = 0
        for module, name, buf in pipeline_buffers(pipe):
            if buf.device.type != 'cuda':
                _set_buffer(module, name, buf.to('cuda'))
            self.buffer_bytes += buf.numel() * buf.element_size()

        # (parameters were already released above, as each tensor was mirrored)

    def __len__(self):
        return len(self.params)

    def _mirror_into_slab(self):
        """Mirror the whole model into one page-locked slab of ordinary memory.

        Pinning tensor by tensor is what made the host footprint ~1.5x the model
        bytes: CUDA's caching host allocator rounds every request up to a
        power-of-two bucket, and a transformer's shapes waste a third of each
        one. One slab per model does not fix it either: measured on this node,
        cudaHostAlloc rounded 6.46 GiB to 8, 16.60 GiB to 32, 27.43 GiB to 32.

        So the slab is allocated as *ordinary* memory and page-locked in place
        with cudaHostRegister, which does no rounding at all: 1.04x instead of
        1.48x, at the same DMA speed (23.3 GiB/s registered vs 21.5 GiB/s from
        the allocator). Each parameter becomes an aligned view into it.

        Registration failing is not fatal, because the slab is still valid memory, so
        the mirror degrades to pageable (correct, but ~4x slower to swap) and
        says so. Failing to allocate the slab is fatal: there is no cheaper way
        to hold the model, since pinning tensor by tensor would need ~1.5x the
        memory that just could not be found.
        """
        sizes = [p.data.numel() * p.data.element_size() for p in self.params]
        offsets, total = [], 0
        for n in sizes:
            offsets.append(total)
            total += _align(n)
        try:
            slab = torch.empty(total, dtype=torch.uint8)
        except (RuntimeError, MemoryError) as exc:
            raise MemoryError(
                f'{self.name}: could not allocate a {total / GIB:.1f} GiB host slab '
                f'({exc}). {HOST_MEMORY_NOTE}.') from exc

        # Copy first, register second: the copies fault the slab's pages in, and
        # each source is dropped as it is taken so the peak stays near one model.
        empty = torch.empty(0)
        for p, off, n in zip(self.params, offsets, sizes):
            src = p.data.detach()
            if n:
                view = slab[off:off + n].view(src.dtype).view(src.shape)
                view.copy_(src if src.is_contiguous() else src.contiguous())
                self.host.append(view)
            else:
                self.host.append(src.clone())      # 0-element: nothing to pin
            p.data = empty                         # release the loaded copy now

        self.slab = slab                           # keeps the views' storage alive
        # Page-locking host memory can cost device memory for the mappings. It
        # measured 0.2% on one L40S, but that is driver- and IOMMU-dependent, so
        # measure it here rather than assume it.
        # HADIS_PIN_MIRRORS=0 skips page-locking entirely. The slab is still
        # one contiguous allocation and every parameter still a view into it, so
        # correctness is unchanged; only the DMA is pageable, which measured
        # ~4x slower per swap (4.79 s vs 1.22 s mean). Use it to rule
        # page-locking in or out as the consumer of device memory.
        if os.environ.get('HADIS_PIN_MIRRORS', '1') == '0':
            self.registered = False
            logging.info(f'[model_store] {self.name}: HADIS_PIN_MIRRORS=0, mirror left '
                         f'pageable ({total / GIB:.2f} GiB); swaps will be ~4x slower')
            return True
        free_before, _ = gpu_free_gib()
        err = torch.cuda.cudart().cudaHostRegister(slab.data_ptr(), total, 0)
        free_after, dev_total = gpu_free_gib()
        self.registered = int(err) == 0
        logging.info(f'[model_store] {self.name}: registered {total / GIB:.2f} GiB host; '
                     f'gpu free {free_before:.2f} -> {free_after:.2f} GiB of {dev_total:.2f} '
                     f'(cost {free_before - free_after:+.2f} GiB)')
        if not self.registered:
            logging.warning(f'[model_store] {self.name}: cudaHostRegister failed '
                            f'({err}); mirror is pageable and will swap slowly')
    def close(self):
        """Undo the page-locking. Registered memory must be released before the
        slab is freed, so this is called from __del__ as well."""
        if getattr(self, 'registered', False):
            self.registered = False
            try:
                torch.cuda.cudart().cudaHostUnregister(self.slab.data_ptr())
            except Exception:                      # interpreter teardown
                pass

    def __del__(self):
        self.close()

    def attach(self, tensors):
        """Point the pipeline's parameters at ``tensors`` (GPU or host)."""
        for p, t in zip(self.params, tensors):
            p.data = t

    def to_gpu(self, stream=None):
        """Copy the host mirror onto the GPU and attach it. Returns seconds."""
        t0 = time.perf_counter()
        gpu = [torch.empty(h.shape, dtype=h.dtype, device='cuda') for h in self.host]
        ctx = torch.cuda.stream(stream) if stream is not None else _NullCtx()
        with ctx:
            for g, h in zip(gpu, self.host):
                g.copy_(h, non_blocking=True)
        torch.cuda.synchronize()
        self.gpu = gpu
        self.attach(gpu)
        return time.perf_counter() - t0

    def to_host(self):
        """Copy the GPU tensors back and release them. Returns seconds."""
        if self.gpu is None:
            return 0.0
        t0 = time.perf_counter()
        for h, g in zip(self.host, self.gpu):
            h.copy_(g, non_blocking=True)
        torch.cuda.synchronize()
        self.release_gpu()
        return time.perf_counter() - t0

    def release_gpu(self):
        self.gpu = None
        self.attach(self.host)


def _align(nbytes, alignment=256):
    """Round up so every view starts on a boundary valid for its dtype."""
    return (nbytes + alignment - 1) // alignment * alignment


def _set_buffer(module, dotted_name, tensor):
    """Assign a buffer given its dotted name relative to ``module``."""
    parts = dotted_name.split('.')
    target = module
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], tensor)


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class ModelStore:
    """Holds every variant in pinned host memory; one is resident on the GPU."""

    def __init__(self, chunk_bytes=DEFAULT_CHUNK_BYTES, swap_log=None):
        self.mirrors = {}
        self.resident = None
        self.chunk_bytes = chunk_bytes
        self.swap_log = swap_log
        self.stream_in = torch.cuda.Stream()
        self.stream_out = torch.cuda.Stream()
        self._log_header_written = False

    def add(self, name, pipe):
        mirror = ModelMirror(name, pipe)
        self.mirrors[name] = mirror
        release_heap()                 # hand the loaded copy's arena back
        logging.info(f'[model_store] pinned {name}: {len(mirror)} tensors, '
                     f'{mirror.nbytes / GIB:.2f} GiB params + '
                     f'{mirror.buffer_bytes / GIB:.2f} GiB buffers on GPU, '
                     f'{mirror.pin_seconds:.1f}s'
                     f'{"" if mirror.pinned else " (NOT pinned, transfers will be slower)"}')
        return mirror

    def total_host_bytes(self):
        return sum(m.nbytes for m in self.mirrors.values())

    def activate(self, name):
        """Make ``name`` the resident model, swapping out whatever is there.

        Returns the elapsed seconds (0.0 if it was already resident).
        """
        if self.resident == name:
            return 0.0
        if name not in self.mirrors:
            raise KeyError(f'{name} is not in the store: {sorted(self.mirrors)}')

        torch.cuda.synchronize()
        # Return the previous model's inference workspace to the driver: the
        # caching allocator holds it, and the incoming model's tensors have
        # different shapes, so the freed blocks are not reusable.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        if self.resident is None:
            self.mirrors[name].to_gpu()
        else:
            self._pipelined_swap(self.mirrors[self.resident], self.mirrors[name])

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        moved = (self.mirrors[name].nbytes +
                 (self.mirrors[self.resident].nbytes if self.resident else 0))
        self._record(self.resident, name, moved, elapsed)
        self.resident = name
        return elapsed

    def _pipelined_swap(self, out_mirror, in_mirror):
        """Walk both models in chunks: one chunk out, one chunk in, per step."""
        out_gpu = out_mirror.gpu
        n_out, n_in = len(out_gpu), len(in_mirror.host)
        in_gpu = [None] * n_in
        i_out = i_in = 0

        while i_out < n_out or i_in < n_in:
            # queue one chunk of the outgoing model, host <- gpu
            freed, nbytes = [], 0
            with torch.cuda.stream(self.stream_out):
                while i_out < n_out and nbytes < self.chunk_bytes:
                    g = out_gpu[i_out]
                    if g is not None:
                        out_mirror.host[i_out].copy_(g, non_blocking=True)
                        nbytes += g.numel() * g.element_size()
                        freed.append(i_out)
                    i_out += 1

            # allocate this chunk's incoming tensors on the default stream first,
            # then queue their copies, gpu <- host
            batch, nbytes = [], 0
            while i_in < n_in and nbytes < self.chunk_bytes:
                h = in_mirror.host[i_in]
                in_gpu[i_in] = torch.empty(h.shape, dtype=h.dtype, device='cuda')
                batch.append(i_in)
                nbytes += h.numel() * h.element_size()
                i_in += 1
            with torch.cuda.stream(self.stream_in):
                for j in batch:
                    in_gpu[j].copy_(in_mirror.host[j], non_blocking=True)

            self.stream_out.synchronize()
            self.stream_in.synchronize()
            # The outgoing copies have landed: release their GPU memory so the
            # incoming model can reuse it. Dropping the list reference is not
            # enough, because attach() also pointed the module's parameter at the same
            # tensor, and that second reference would keep the whole outgoing
            # model alive until the swap finished, defeating the point of
            # chunking (both models resident = more than the card holds).
            for j in freed:
                out_mirror.params[j].data = out_mirror.host[j]
                out_gpu[j] = None

        out_mirror.release_gpu()
        in_mirror.gpu = in_gpu
        in_mirror.attach(in_gpu)

    def _record(self, src, dst, moved_bytes, elapsed):
        peak_alloc = torch.cuda.max_memory_allocated() / GIB
        peak_reserved = torch.cuda.max_memory_reserved() / GIB
        logging.info(f'[model_store] {src} -> {dst}: {elapsed * 1000:.0f} ms, '
                     f'{moved_bytes / GIB:.2f} GiB, peak {peak_alloc:.2f} GiB allocated')
        if not self.swap_log:
            return
        try:
            new = not self._log_header_written
            with open(self.swap_log, 'a') as f:
                if new:
                    f.write('timestamp,from,to,bytes,ms,peak_alloc_gib,peak_reserved_gib\n')
                    self._log_header_written = True
                f.write(f'{time.time():.3f},{src},{dst},{moved_bytes},'
                        f'{elapsed * 1000:.1f},{peak_alloc:.2f},{peak_reserved:.2f}\n')
        except OSError as exc:
            logging.warning(f'[model_store] could not write {self.swap_log}: {exc}')
