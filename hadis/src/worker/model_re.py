"""Real-execution variant of ``model.py`` (E2).

A copy of the profile-driven model, changed only where real GPUs and real
checkpoints demand it. ``model.py`` is left untouched: it is what the
profile-driven experiment (E1) runs and is already validated.

What differs, and why:

  * ``pipe_load`` takes checkpoints from ``config.MODEL_SPECS``. The
    original sd35* branches point at SD2/SD3 repositories
    (``stabilityai/sd-turbo``, ``stable-diffusion-3-medium``,
    ``stable-diffusion-3-large``) and never assign a guidance scale, so
    they raise UnboundLocalError before they can serve.
  * Pipelines are loaded onto the CPU and handed to ``ModelStore``. The
    original ends ``pipe_load`` with ``pipe.to("cuda")`` and
    ``loadAllModels`` puts all four variants on the card at once, which no
    single GPU has room for.
  * A model change is a swap, not a re-load: the incoming variant is
    copied in from its page-locked host mirror while the outgoing one is
    copied back, and every swap is timed into logs/model_swaps.csv.
  * The discriminator is the CLIP checkpoint that ships with the artifact,
    not ``<modelDir>/multi.pt``, which does not exist.
"""
import os
import sys
sys.path.append('..')
import logging
import pickle
import time
import threading
# import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import efficientnet_v2_s
from typing import List
from enum import Enum
# from multiprocessing import Process
import torch.multiprocessing as mp
from common.query import Query, QueryResults
from PIL import Image
from diffusers import DiffusionPipeline, DDIMScheduler, StableDiffusionPipeline, LCMScheduler, AutoPipelineForText2Image, StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torchvision.transforms as transforms
import config
from config import get_cas_exec, get_do_simulate, get_model_order
from config import MODEL_SPECS, get_model_cache
from model_loader import load_pipeline
from model_store import ModelStore, release_heap, current_host_rss_gib, gpu_free_gib


class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=2, bias=True)
        )
        self.net = efficientnet_v2_s(weights='IMAGENET1K_V1')
        self.net.classifier = classifier

    def forward(self, x):
        return self.net(x)

transform  = transforms.Compose([transforms.Resize(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.507395516207, ),(0.255128989415, ))
                                ])


USECONDS_IN_SEC = 1000 * 1000
MSECONDS_IN_SEC = 1000
# Default multiple of a query's latency SLO within which this worker is willing
# to start it. The controller overrides it per allocation policy (see
# SLO_ADMISSION_FACTOR in controller.py); this value is what every policy except
# Clipper-Heavy uses.
SLO_FACTOR = 1.5
lock = threading.Lock()

# Serialises GPU work against model swaps. serviceQueueLoop runs inference in one
# thread of the model process while the IPC thread handles LOAD_MODEL, and a swap
# frees the outgoing model's GPU tensors. Without this a swap arriving mid-batch
# pulls the weights out from under a forward pass.
#
# Module scope rather than an attribute because LoadedModel is pickled into the
# spawned child and a threading lock is not picklable.
swap_lock = threading.RLock()
ALLOW_RANDOM = False # true for ablation study


class ModelState(Enum):
    READY = 1
    NO_MODEL_LOADED = 2

# This class is responsible for loading a model variant and running inference
# on it
class LoadedModel:
    def __init__(self, pipe1, pipe2):
        self.model = None
        self.modelName = None
        self.g_scale = None
        self.n_steps = None
        self.conf_thres = None
        self.router_thres = None
        self.batch_size = None
        self.is_lightweight = None
        self.slo_admission_factor = SLO_FACTOR
        self.discriminator = None
        self.infer_level = None
        self.conf_dist = None
        self.preprocess = None
        self.ort_predict = None
        self.postprocess = None
        # Make sure that LoadedModel's appID and worker's appID are always in sync
        # i.e., there are no methods that modify LoadedModel's appID but do not
        # change worker's appID, and vice versa
        self.appID = None
        self.task = None
        self.queue = []
        self.modelDir = config.repo_path('discriminator')

        # Host mirrors of every variant; exactly one is on the GPU at a time.
        self.store = None
        self.swap_log_path = None

        self.label_to_task = None

        self.state = ModelState.NO_MODEL_LOADED

        # TODO: should these be hard-coded?
        self.model_names = ['sd21', 'sdxl', 'sdv15', 'sdxl-turbo', 'sdv15-lcm', 'sdxl-lcm', 
                          'sdxl-lightning', 'sdxs', 'tinysd', 'sdturbo', 
                          'sd35-turbo', 'sd35-medium', 'sd35-large']
        self.scheduler_names = ['dpms++', 'default', 'ddim']
        
        # CHANGED: support arbitrary staged models via dictionary
        self.stage_model_args = {}  # e.g., {0: (pipe, g_scale, steps), 1: (...), ...}
        
        self.do_simulate = get_do_simulate()
        self.pipeline = get_cas_exec()
        # __init__ runs in the worker daemon process, but readIPCMessages runs in
        # a spawned child that re-imports config with its defaults; carry the mode
        # across explicitly so lookups in the child agree with the daemon.
        self.profile_driven = config.get_profile_driven()
        self.live_discriminator = config.get_live_discriminator()
        self.discriminatorPreprocess = None
        logging.info(config.mode_banner('model'))

        self.conf_dist_path = config.repo_path('discriminator', 'confidence_scores')
        self.router_dist_path = config.repo_path('router', 'prompt_features.csv')

        # {(model, model, batch): milliseconds}, including the interpolated odd
        # batch sizes serviceQueue can ask for.
        self.profiled_runtimes = config.get_model_runtimes()

        self.readPipe, _ = pipe1
        _, self.writePipe = pipe2
        
        self.serviceQueueThread = None
        self.pipeProcess = mp.Process(target=self.readIPCMessages, args=((pipe1, pipe2,)))
        self.pipeProcess.start()


    # Load a new model
    def load(self, modelName, schedulerName, infer_level, router_thres, conf_thres, batch_size, is_lightweight, appID, task):
        # previousModel = self.model
        loadedFrom = None

        # TODO: check whether modelName and schedulerName are in self.model_names and self.scheduler_names
        try:
            self.router_thres = float(router_thres)
            self.conf_thres = float(conf_thres)
            self.batch_size = int(batch_size)
            self.is_lightweight = int(is_lightweight)
            loadedFrom, loadingTime = self.loadFromStorage(modelName, schedulerName, infer_level)
            
            # TODO: make this asynchronous, we do not want to wait on model unloading
            # TODO: or should it be synchronous and we wait for requests of currently
            #       loaded model to finish before we load new model?
            # if previousModel is not None:
            #     self.unload(previousModel)

            self.appID = appID
            self.task = task
            self.state = ModelState.READY

            # logging.info(f'self.state: {self.state}')
            # logging.info(f'self.model: {self.model}')
            # logging.info(f'self.discriminator: {self.discriminator}')
            
            return loadedFrom, loadingTime
        except Exception as e:
            raise e
            
            
    def loadAllModels(self):
        """Mirror every variant in page-locked host memory, then warm each one.

        The whole cascade is resident on the host (measured 79.5 GB for the four
        variants) and exactly one variant is on the GPU; switching between them
        is a swap, not a load from disk.
        """
        available_models = get_model_order()
        self.store = ModelStore()

        for level, model_name in enumerate(available_models):
            t0 = time.time()
            pipe, gscale, steps = self.pipe_load(model_name)
            self.stage_model_args[level] = (pipe, gscale, steps)
            mirror = self.store.add(model_name, pipe)
            # Hand the loaded pageable copy's pages back before the next model.
            # Python frees them but glibc keeps the arena, so without this the
            # peak compounds by roughly one model per variant: measured 142 GB
            # instead of 89 GB across the four, which overruns a 128 GB node.
            release_heap()
            logging.info(f'loadAllModels: {model_name} mirrored in '
                         f'{time.time() - t0:.1f}s, {mirror.nbytes / 1024 ** 3:.1f} GiB, '
                         f'pinned={mirror.pinned}, host RSS {current_host_rss_gib():.1f} GB, '
                         f'gpu free {gpu_free_gib()[0]:.2f} GiB')

        # One warm-up pass each: the first call on a pipeline compiles kernels
        # and allocates workspaces, which would otherwise land on the first
        # real query and blow its SLO.
        for level, model_name in enumerate(available_models):
            self.store.activate(model_name)
            self.model, self.g_scale, self.n_steps = self.stage_model_args[level]
            free, dev_total = gpu_free_gib()
            logging.info(f'loadAllModels: warming {model_name}: gpu free {free:.2f} of '
                         f'{dev_total:.2f} GiB, torch reserved '
                         f'{torch.cuda.memory_reserved() / 1024 ** 3:.2f} GiB, '
                         f'allocated {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GiB')
            self.model(**self.get_inputs('Warm-up pass', 1))
            # Return the warm-up's cached blocks to the driver. Inference peaks
            # well above the weights (SDXL's VAE decode alone transiently takes
            # ~6.5 GiB) and the caching allocator keeps every block it ever
            # took: measured 40.1 GiB reserved against 17.4 GiB actually in use,
            # leaving 3.6 GiB free on a 44 GiB card. Without this the next
            # variant's activation has to fit in whatever is left.
            torch.cuda.empty_cache()
            logging.info(f'loadAllModels: warmed {model_name}, gpu reserved '
                         f'{torch.cuda.memory_reserved() / 1024 ** 3:.1f} GiB, '
                         f'free {torch.cuda.mem_get_info()[0] / 1024 ** 3:.1f} GiB')
        release_heap()
        self.discriminator = self.loadDiscriminator()

        # Logged last, and only once everything a query needs is on the GPU:
        # start_worker_re.sh waits for this line, so anything that can still
        # fail must happen before it or the launcher reports a worker ready
        # that is in fact about to die.
        logging.info(f'loadAllModels: {len(available_models)} variants ready, '
                     f'{self.store.total_host_bytes() / 1024 ** 3:.1f} GiB mirrored, '
                     f'host RSS {current_host_rss_gib():.1f} GB, '
                     f'gpu free {gpu_free_gib()[0]:.2f} GiB')
        
    def loadFromStorage(self, modelName, schedulerName, infer_level):
        # TODO: is there anything else to do?
        loadingTimeStart = time.time()
        if modelName == 'sink':
            self.model = 'sink'
            return 'storage', 0
            
        available_models = get_model_order()
        if not self.stage_model_args:
            if self.do_simulate:
                for level, model_name in enumerate(available_models):
                    self.stage_model_args[level] = (model_name, 0, 1)
                self.discriminator = 'discriminator'
            else:
                self.loadAllModels()
        
        if modelName == self.modelName:
            pass
        else:
            previousModel = self.modelName
            # The switch and the swap are one atomic step: between rebinding
            # self.model and moving its weights onto the GPU the pipeline would
            # otherwise be pointing at host tensors, and any batch that started
            # in that window would fail with a device mismatch.
            waitStart = time.time()
            with swap_lock:
                waited = time.time() - waitStart
                self.modelName = modelName
                if int(infer_level) in self.stage_model_args:
                    self.model, self.g_scale, self.n_steps = self.stage_model_args[int(infer_level)]
                # Bring the incoming variant onto the GPU from its host mirror,
                # copying the outgoing one back in the same pipelined pass.
                if self.store is not None and modelName in self.store.mirrors:
                    swap_s = self.store.activate(modelName)
                    logging.info(f'SWAP: {previousModel} -> {modelName} in '
                                 f'{swap_s:.3f}s (waited {waited:.3f}s for in-flight batch)')
                    self.logSwap(previousModel, modelName, swap_s, waited)
            if self.pipeline == 'multi':
                # discriminator scores
                conf_dist = np.loadtxt(os.path.join(self.conf_dist_path, f'scores_model_{infer_level}.txt'))
                if conf_dist.ndim > 1:
                    conf_dist = conf_dist[:, 1] # real scores
                # router scores
                feat_weight = np.array([0.85, 0.32, 0.72, 0.99, 0.76, 0.07, 0.78, 0.01])
                prompt_feats = pd.read_csv(self.router_dist_path)
                feats = prompt_feats.iloc[:, 1:]
                feat_mean = feats.mean()
                feat_std = feats.std()
                norm_feats = (feats - feat_mean) / feat_std
                router_dist = norm_feats @ feat_weight
            else:
                conf_dist = np.loadtxt(self.conf_dist_path)
                router_dist = None
            self.conf_dist = conf_dist.reshape(-1,4).mean(axis=1)
            self.router_dist = router_dist

        loadingTime = int((time.time() - loadingTimeStart) * USECONDS_IN_SEC)
        self.infer_level = int(infer_level)

        return 'storage', loadingTime

    
    def loadDiscriminator(self):
        """Build the CLIP discriminator and load its trained weights.

        The checkpoint is a *state_dict*, not a pickled module: model.py's
        `torch.load(...).cuda()` assumes the latter and fails with
        "'collections.OrderedDict' object has no attribute 'cuda'".

        Only the 2-layer head is trained. The ViT-B/32 image encoder is frozen
        (its forward runs under torch.no_grad), so the artifact ships the head
        alone at 0.5 MB and reconstructs the encoder from clip.load, instead of
        carrying a 336 MB copy of weights clip.load downloads anyway. A full
        checkpoint still loads if one is present.

        The module is imported by path because discriminator/model.py shares a
        filename with this package's model.py.
        """
        import importlib.util
        import clip

        discDir = self.modelDir
        spec = importlib.util.spec_from_file_location(
            'hadis_discriminator_model', os.path.join(discDir, 'model.py'))
        discModule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(discModule)

        clipModel, preprocess = clip.load('ViT-B/32', device='cpu')
        # The input the discriminator was trained and scored with. The stored
        # confidence scores were computed on images prepared this way.
        self.discriminatorPreprocess = preprocess
        discriminator = discModule.CLIPDiscriminator(clipModel.float())

        head = os.path.join(discDir, 'CLIP_discriminator_head.pt')
        full = os.path.join(discDir, 'CLIP_discriminator.pt')
        path = head if os.path.isfile(head) else full
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'no discriminator weights in {discDir} '
                f'(expected CLIP_discriminator_head.pt or CLIP_discriminator.pt)')
        stateDict = torch.load(path, map_location='cpu')

        missing, unexpected = discriminator.load_state_dict(stateDict, strict=False)
        # Encoder keys missing is expected for the head-only checkpoint: they are
        # already correct, straight from clip.load. Anything else is not.
        headMissing = [k for k in missing if not k.startswith('clip_image_encoder.')]
        if headMissing or unexpected:
            raise RuntimeError(
                f'{os.path.basename(path)} does not fit CLIPDiscriminator: '
                f'missing {headMissing}, unexpected {unexpected}')

        discriminator = discriminator.cuda().eval()
        logging.info('loadDiscriminator: escalation uses the '
                     + ('live discriminator output' if self.live_discriminator
                        else 'precomputed per-prompt scores'))
        logging.info(f'loadDiscriminator: CLIP discriminator ready on GPU '
                     f'({os.path.basename(path)})')
        return discriminator

    def logSwap(self, fromModel, toModel, seconds, waited=0.0):
        """Append one row to logs/model_swaps.csv.

        Evidence that the swap path ran, and the only record of what a
        transition costs during a real run. The paper says worker roles are
        swapped at runtime but reports no cost for it.
        """
        if self.swap_log_path is None:
            logDir = config.repo_path('logs')
            os.makedirs(logDir, exist_ok=True)
            self.swap_log_path = os.path.join(
                logDir, f'model_swaps_{os.environ.get("HADIS_WORKER_PORT", "worker")}.csv')
            if not os.path.exists(self.swap_log_path):
                with open(self.swap_log_path, 'w') as f:
                    f.write('timestamp,from_model,to_model,seconds,'
                            'waited_s,peak_alloc_gib,peak_reserved_gib\n')
        try:
            with open(self.swap_log_path, 'a') as f:
                f.write(f'{time.time():.6f},{fromModel},{toModel},{seconds:.4f},'
                        f'{waited:.4f},'
                        f'{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f},'
                        f'{torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}\n')
        except OSError as exc:                 # logging must never kill a worker
            logging.warning(f'logSwap: could not write {self.swap_log_path}: {exc}')

    def pipe_load(self, model_name: str, scheduler='default'):
        """Load one variant onto the CPU.

        Returns (pipe, guidance_scale, num_inference_steps) like the original,
        but the pipeline stays on the host: ModelStore owns the GPU transfer.
        """
        if model_name not in MODEL_SPECS:
            raise ValueError(f'{model_name} is not a real-execution variant; '
                             f'expected one of {sorted(MODEL_SPECS)}')
        spec = MODEL_SPECS[model_name]
        pipe = load_pipeline(model_name, get_model_cache(), log=logging.info)
        return pipe, spec['guidance'], spec['steps']

    def get_inputs(self, data_prompts, batch_size=1):
        generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
        prompts = data_prompts if isinstance(data_prompts, List) else batch_size * [data_prompts]
        if self.g_scale is not None:
            return {"prompt": prompts, 
                "generator": generator, 
                "num_inference_steps": self.n_steps, 
                "guidance_scale": self.g_scale}
        else:
            return {"prompt": prompts, 
                "generator": generator, 
                "num_inference_steps": self.n_steps}

    
    # Unload the currently loaded model
    def unload(self, model):
        # TODO: Stop its execution thread and remove model from GPU memory
        # TODO: Should we wait for its current requests to complete? (empty queue)
        #       If yes, should this block before the next model is loaded? Otherwise
        #       GPU might automatically unload this model to load new one, load this one
        #       again to execute its requests, resulting in thrashing
        pass
        # Join will not work if the queueProcess does not finish on its own,
        # we may have to interrupt it
        self.queueProcess.join()
        raise Exception('unload is not yet implemented')

            
    def serviceQueue(self):
        # TODO: this should only be called from the readQueue process to avoid
        #       self.queue synchronization issues
        # Should we use semaphore on self.queue anyway? What about callbacks to this
        # function? Which process do they execute in?
        # Possibile options:
        # 1. Not enough requests in queue, return
        # 2. Pop requests from queue and serve (what does the callback do?)
        # 3. For other algorithms, perhaps set an interrupt timer to this function

        if len(self.queue) == 0:
            # Nothing to do
            return
        
        # If there are any requests in queue, serve each of them one-by-one
        # with batch size of 1
        while len(self.queue) >= 1:
            try:
                popped = []
                for i in range(self.batch_size):
                    bs = np.min([self.batch_size, len(popped)+len(self.queue)])
                    bs = bs if bs in [1,2,3,4,5,6,7,8,16,32] else 4 # TODO: don't hard code here
                    if len(self.queue) > 0:
                        query = self.queue.pop(0)
                        # drop a query if it would expire by executing all the requests in a queue in a batch
                        # or with the maximum batch size
                        # estimate_remaining_runtime = query.processingTime + self.profiled_runtimes[(self.modelName, self.modelName, bs)] / MSECONDS_IN_SEC
                        # Estimated remaining runtime = processing time of previous worker (if there is)
                        #                                 + the time since the query is added to queue
                        #                                 + the runtime with a given batch size
                        inqueue_time = time.time()-query.timestamp
                        estimate_remaining_runtime = query.processingTime + inqueue_time + self.profiled_runtimes[(self.modelName, self.modelName, bs)] / MSECONDS_IN_SEC
                        expire_time = query.latencySLOInUSec / USECONDS_IN_SEC * self.slo_admission_factor
                        logging.info(f"expire_time: {expire_time}, processingTime: {query.processingTime}, inqueue time: {inqueue_time}, estimate_remaining_runtime: {estimate_remaining_runtime}")
                        if expire_time < estimate_remaining_runtime:
                            logging.info(f"Drop request: {query.requestID}, estimate remaining runtime: {estimate_remaining_runtime}, expire_time: {expire_time}")
                            continue
                        popped.append(query)

                        event = {'event': 'WORKER_DEQUEUED_QUERY',
                                'requestID': query.requestID, 'queryID': query.queryID,
                                'userID': query.userID, 'appID': query.applicationID,
                                'task': self.task, 'sequenceNum': query.sequenceNum,
                                'timestamp': time.time()}
                        # logging.info(f'EVENT,{str(event)}')
                        logging.info(f'EVENT: WORKER_DEQUEUED_QUERY, sequenceNum: {query.sequenceNum}')

                with lock:
                    self.writePipe.send(f'QUEUE_SIZE,{len(self.queue)}')
                    logging.info(f'Check send QUEUE_SIZE')
            except Exception as e:
                logging.error(f"Model: Error serviceQueue - {e}")

            # # Batch size of 1
            # popped = [popped]

            logging.info(f'Check executeBatch START')
            if len(popped) >= 1:
                self.executeBatch(popped)
            else:
                logging.info('No query in batch ...')
            logging.info(f'Check executeBatch END')
            pass

        return
    

    def executeBatch(self, queries):
        # Check if model is ready to execute
        if self.state == ModelState.NO_MODEL_LOADED:
            logging.error(f'\texecuteBatch: no model is currently loaded, cannot '
                          f'execute request')
            return
        elif self.state == ModelState.READY:
            pass
        else:
            logging.error(f'Model state {self.state} not handled by executeBatch()')
            return

        # Extract data from list of Query objects
        data_prompts = list(map(lambda x: x.prompt, queries)) # prompts
        data_array = list(map(lambda x: x.data, queries)) # images
        conf_idx = list(map(lambda x: x.sequenceNum, queries))

        # Run the inference
        try:
            batch_size = len(data_prompts) # The batch_size here is 'How many images are generated per prompt', not equal to self.batch_size
            logging.info(f"Start image generation for batch size {batch_size}")
            start_time = time.time()
            # Held for the whole GPU section: a swap must not land between the
            # forward pass and the discriminator, which also runs on the GPU.
            with swap_lock:
                if self.do_simulate:
                    bs = batch_size if batch_size in [1,2,3,4,5,6,7,8,16,32] else 4
                    time.sleep(self.profiled_runtimes[(self.modelName, self.modelName, bs)] / MSECONDS_IN_SEC)
                    results = torch.randn(bs, 3, 224, 224)
                else:
                    results = self.model(**self.get_inputs(data_prompts, batch_size))
                inferenceModel = self.modelName
            inference_time = time.time() - start_time
            print(f'\tProcess 2, inference time: {(inference_time):.6f}')
            logging.info(f'\tInference time: {(inference_time):.6f}')

            # Verify the qualify of the image by the discriminator
            start_time = time.time()
            results_qualified = [] # 1 for qualified, and 0 for non-qualified which need to be sent to 2nd level workers
            
            if isinstance(results, torch.Tensor):
                image_tensors = results
            else:
                prep = self.discriminatorPreprocess or transform
                image_tensors = torch.stack([prep(results.images[i]) for i in range(batch_size)])
            image_tensors = image_tensors.cuda()
            if self.do_simulate:
                time.sleep(0.01)
            else:
                with swap_lock, torch.no_grad():
                    logits, _ = self.discriminator(image_tensors)
                # Probability of the "real" class, the same quantity as the
                # stored scores, moved to the CPU for the threshold comparison.
                live_scores = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            if ALLOW_RANDOM: # Random assiging confidence score
                rng = np.random.default_rng()
                conf_scores = [rng.uniform(0.0,1.0) for i in conf_idx]
                abs_conf_thres = self.conf_thres
            else:
                # The threshold is a percentile of the stored score distribution,
                # which is what the planner assumes, in both modes.
                sorted_dist = np.sort(self.conf_dist)
                index = max(int(len(sorted_dist) * self.conf_thres) - 1, 0)
                abs_conf_thres = sorted_dist[index]
                if self.live_discriminator and not self.do_simulate:
                    conf_scores = live_scores
                else:
                    conf_scores = self.conf_dist[conf_idx] # directly get data from pre-computed files
            results_qualified = [1 if cs>=abs_conf_thres else 0 for cs in conf_scores]
            logging.info(f'abs_conf_thres: {abs_conf_thres}, conf_score: {conf_scores}')
            
            print(f'\tProcess 2, varification time: {(time.time() - start_time):.6f}')
            print(f'\tVerification results: {results_qualified}')
            logging.info(f'Verification results: {results_qualified}')
            print(f'\tProcess 2, sending completed inference at {time.time()}')
        except Exception as e:
            logging.error(f"Model: Error executionBatch - {e}")
        
        with lock:
            logging.info(f'Check send COMPLETED_INFERENCE')
            self.writePipe.send('COMPLETED_INFERENCE')
            logging.info(f'Check send COMPLETED_INFERENCE DONE')
            for i in range(batch_size):
                queries[i].resultQualified = results_qualified[i]
                # query_results = QueryResults(queries[i].queryID, data_prompts[i], results.images[i], results_qualified[i])
                logging.info(f'Check send query {i+1}/{batch_size} COMPLETED_INFERENCE')
                self.writePipe.send(queries[i])
                logging.info(f'Check send query {i+1}/{batch_size} COMPLETED_INFERENCE DONE')
            logging.info(f'Check send DONE_SENDING')
            self.writePipe.send('DONE_SENDING')
            logging.info(f'Check send DONE_SENDING DONE')

        for query in queries:
            event = {'event': 'WORKER_COMPLETED_QUERY',
                    'requestID': query.requestID, 'queryID': query.queryID,
                    'userID': query.userID, 'appID': query.applicationID,
                    'task': self.task, 'sequenceNum': query.sequenceNum,
                    'timestamp': time.time()}
            # logging.info(f'EVENT,{str(event)}')
            logging.info(f'EVENT: WORKER_COMPLETED_QUERY, sequenceNum: {query.sequenceNum}')

        return
    
    
    # Seconds to idle when the queue is empty. serviceQueue() returns at once
    # when there is nothing to do, so without this the loop spins at full speed
    # and never yields the GIL voluntarily. A model swap issues ~2400
    # Python-level copy_() calls, each of which must then wait out a GIL switch
    # interval: measured 1.2 s -> 43-46 s per swap, recovering to 1.2 s the
    # moment the spinning stops. It also burned a core per worker doing nothing.
    #
    # 2 ms is far below anything that matters here (the fastest model takes
    # 400 ms) and keeps pickup latency negligible.
    QUEUE_IDLE_SEC = 0.002

    def serviceQueueLoop(self):
        logging.info('ServiceQueue event loop waiting')
        logging.info(f'Profiled_runtimes: {self.profiled_runtimes}')
        while True:
            if not self.queue:
                time.sleep(self.QUEUE_IDLE_SEC)
                continue
            self.serviceQueue()

    
    # Simulated process
    # TODO: update the actual process of router [i.e., from ../../router/promptFeature.py]
    def isHardByRouter(self, query):
        if self.pipeline == 'multi' and self.is_lightweight == 1:
            if ALLOW_RANDOM: # Random assiging router score
                abs_router_thres = 1 - self.router_thres
                rng = np.random.default_rng()
                router_score = rng.uniform(0.0, 1.0)
            else: # Router score from pre-computed file
                sorted_dist = np.sort(self.router_dist)
                index = max(int(len(sorted_dist) * (1-self.router_thres)) - 1, 0)
                abs_router_thres = sorted_dist[index]
                router_score = self.router_dist[query.sequenceNum]
            is_hard = True if router_score >= abs_router_thres else False
            return is_hard
        else:
            return False
      

    def set_slo_admission_factor(self, value):
        """Empty string means the controller sent nothing: keep the default."""
        self.slo_admission_factor = float(value) if value not in ('', 'None') else SLO_FACTOR
        return self.slo_admission_factor


    def readIPCMessages(self, pipe1, pipe2):
        # Runs in the spawned child process: restore the run mode picked up from
        # the daemon before anything reads it from config.
        config.set_profile_driven(self.profile_driven)
        config.set_cas_exec(self.pipeline)
        if self.do_simulate:
            config.set_do_simulate_true()

        # for aligning the log names of model and worker
        workerPort = self.readPipe.recv()
        # logfile_name = f'../../logs/model_{time.time()}.log'
        logfile_name = f'../../logs/model_{workerPort}.log'
        logging.basicConfig(filename=logfile_name, level=logging.INFO,
                            format='%(asctime)s %(levelname)-8s %(message)s')
        
        readPipe, _ = pipe1
        _, writePipe = pipe2
        # TODO: This is busy waiting. Is there a better way to do this?
        while True:
            message = readPipe.recv()

            logging.info(f'\tProcess 2, readQueue: message: {message}')

            if message == 'QUERY':
                if self.model == 'sink':
                    continue
                query = readPipe.recv()
                self.queue.append(query)

                logging.info(f'\tProcess 2, readQueue: Appended query to queue from '
                             f'readQueue, time: {time.time()}')
                
                # self.serviceQueue()
            
            elif message == 'REQUEST':
                if self.model == 'sink':
                    continue
                request = readPipe.recv()
                # TODO: construct queries from request and put them in queue
                #       it is better to do that here than in the worker daemon process

                # TODO: Initial request has task ID 0
                # TODO: replace this application's defined task
                # TODO: for intermediate task, it should use that task information
                # TODO: Perhaps this task information should be passed as part of
                #       the request


                print(f'before preprocessing,request: {request}')
                # logging.info(f'before preprocessing, request: {request}')
                print(f'before preprocessing, request.prompt: {request.prompt}')
                # logging.info(f'before preprocessing, request.prompt: {request.prompt}')
                print(f'before preprocessing, request.data: {request.data}')
                # logging.info(f'before preprocessing, request.data: {request.data}')
                # queries = self.preprocess(request, self.dataset)
                queries = [request]
                # TODO: add request.data which is images produced by 1st-level workers
                # for img2img in 2nd-level workers
                for query in queries:
                    query_is_hard = self.isHardByRouter(query)
                    if query_is_hard:
                        with lock:
                            logging.info(f'Check send COMPLETED_INFERENCE [by router]')
                            writePipe.send('COMPLETED_INFERENCE_BY_ROUTER')
                            logging.info(f'Check send COMPLETED_INFERENCE DONE [by router]')
                            query.resultQualified = 0 # directly send to next stage
                            writePipe.send(query)
                            logging.info(f'Check send DONE_SENDING [by router]')
                            writePipe.send('DONE_SENDING')
                            logging.info(f'Check send DONE_SENDING DONE [by router]')
                    else:
                        self.queue.append(query)

                        event = {'event': 'WORKER_ENQUEUED_QUERY',
                                'requestID': query.requestID, 'queryID': query.queryID,
                                'userID': query.userID, 'appID': query.applicationID,
                                'task': self.task, 'modelVariant': self.modelName,
                                'sequenceNum': query.sequenceNum,
                                'timestamp': time.time()}
                        # logging.info(f'EVENT,{str(event)}')
                        logging.info(f'EVENT: WORKER_ENQUEUED_QUERY, sequenceNum: {query.sequenceNum}')

                        with lock:
                            logging.info('Check send QUEUED_QUERY')
                            writePipe.send(f'QUEUED_QUERY,{len(self.queue)}')
                            logging.info('Check send QUEUED_QUERY DONE')
                            writePipe.send(query)
                            logging.info('Check send query [QUEUED_QUERY] DONE')

                logging.info(f'\tProcess 2, readQueue: Appended query to queue from '
                             f'readQueue, time: {time.time()}')

                # self.serviceQueue()

            elif message == 'UPDATE_THRES_LEVEL':
                update_message = readPipe.recv()
                (infer_level, router_thres, conf_thres, batch_size, is_lightweight,
                 slo_admission_factor) = update_message.split(',')
                self.set_slo_admission_factor(slo_admission_factor)
                self.infer_level = int(infer_level)
                self.router_thres = float(router_thres)
                self.conf_thres = float(conf_thres)
                self.batch_size = int(batch_size)
                self.is_lightweight = int(is_lightweight)
                
            elif message == 'LOAD_MODEL':
                load_model_message = readPipe.recv()
                (modelName, schedulerName, infer_level, router_thres, conf_thres, batch_size,
                 is_lightweight, appID, task, slo_admission_factor) = load_model_message.split(',')
                self.set_slo_admission_factor(slo_admission_factor)
                self.childrenTasks = readPipe.recv()
                self.label_to_task = readPipe.recv()
                print(f'\tchildrenTasks: {self.childrenTasks}')
                print(f'\tlabel_to_task: {self.label_to_task}')
                print((f'\tinfer_level: {infer_level}'))
                print(f'Model loaded, Worker is ready.')
                # logging.info(f'\tchildrenTasks: {self.childrenTasks}')
                # logging.info(f'\tlabel_to_task: {self.label_to_task}')
                logging.info(f"Check LOAD_MODEL, model: {modelName}")
                logging.info(f'\tinfer_level: {infer_level}')

                logging.info("Check load and loadFromStorage")
                loadedFrom, loadingTime = self.load(modelName, schedulerName, infer_level, router_thres, conf_thres, batch_size, is_lightweight, appID, task)

                logging.info(f'\tProcess 2, readQueue: loaded model {modelName} from '
                             f'{loadedFrom} in time {loadingTime} micro-seconds')
                
                with lock:
                    logging.info("Check send LOAD_MODEL_RESPONSE")
                    writePipe.send('LOAD_MODEL_RESPONSE')
                    writePipe.send(f'{modelName},{loadedFrom},{loadingTime}')
                
                if self.serviceQueueThread is None and not modelName == 'sink':
                    self.serviceQueueThread = threading.Thread(target=self.serviceQueueLoop)
                    self.serviceQueueThread.start()
    

    def inference(self):
        # TODO: If a query in the queue does not belong to the appID, remove it
        raise Exception('inference is not yet implemented')

