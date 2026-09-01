"""Load a real diffusion variant onto the CPU, from a local cache when possible.

Shared by the real-execution worker (``model_re.py``) and the profiler
(``experiments/profile_gpu.py``) so that the checkpoint a profile measures is
the checkpoint that serves. Every pipeline is returned on the CPU: ``ModelStore``
owns the transfer to the GPU, and four variants cannot sit on one card at once.
"""
import logging
import os

import torch

from config import MODEL_SPECS


def missing_locally(exc):
    """True if this exception means "not in this cache", not "load failed".

    Distinguishing the two matters: treating a real failure as a cache miss
    sends the loader on to the next cache, which then downloads tens of GB
    instead of reporting the problem.
    """
    if isinstance(exc, OSError):              # HF raises OSError subclasses
        return True
    return type(exc).__name__ in ('LocalEntryNotFoundError', 'EntryNotFoundError',
                                  'RepositoryNotFoundError')


def load_pipeline(name, cache_dir=None, allow_download=True, log=logging.info):
    """Load one variant onto the CPU from ``cache_dir``.

    The artifact assumes every checkpoint lives in one cache directory; None
    means HuggingFace's own (HF_HOME, else ~/.cache/huggingface). The cache is
    tried **without touching the network** first, so a checkpoint already on
    disk is never re-downloaded, and an error that is not "absent from this
    cache" is raised rather than being retried as a miss. Treating a real
    failure as a miss is what silently re-downloads tens of GB.
    """
    try:
        return load_pipeline_from(name, cache_dir, local_files_only=True)
    except Exception as exc:
        if not missing_locally(exc):
            raise
    where = cache_dir or 'the default HuggingFace cache'
    if not allow_download:
        raise FileNotFoundError(
            f'{name} ({MODEL_SPECS[name]["repo"]}) is not in {where} '
            f'and downloading is disabled')
    log(f'  not cached; downloading {MODEL_SPECS[name]["repo"]} into {where}')
    return load_pipeline_from(name, cache_dir, local_files_only=False)


def load_pipeline_from(name, cache_dir, local_files_only=False):
    from diffusers import (StableDiffusion3Pipeline, StableDiffusionXLPipeline,
                           UNet2DConditionModel, EulerDiscreteScheduler)
    spec = MODEL_SPECS[name]
    if spec['kind'] == 'sd3':
        return StableDiffusion3Pipeline.from_pretrained(
            spec['repo'], torch_dtype=torch.float16, cache_dir=cache_dir,
            local_files_only=local_files_only)

    # SDXL-Lightning is the SDXL base with a distilled UNet swapped in
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    base = 'stabilityai/stable-diffusion-xl-base-1.0'
    unet = UNet2DConditionModel.from_config(
        UNet2DConditionModel.load_config(base, subfolder='unet', cache_dir=cache_dir,
                                         local_files_only=local_files_only)).to(torch.float16)
    unet.load_state_dict(load_file(
        hf_hub_download(spec['repo'], spec['ckpt'], cache_dir=cache_dir,
                        local_files_only=local_files_only), device='cpu'))
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base, unet=unet, torch_dtype=torch.float16, variant='fp16', cache_dir=cache_dir,
        local_files_only=local_files_only)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing='trailing')
    return pipe
