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
        self.modelDir = '../../models'

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
        # CHANGED: use global config-based list instead of hardcoding
        available_models = get_model_order()  # <- reference from config

        for level, model_name in enumerate(available_models):
            self.stage_model_args[level] = self.pipe_load(model_name)  # default scheduler used

        for level, (pipe, gscale, steps) in self.stage_model_args.items():
            self.model, self.g_scale, self.n_steps = pipe, gscale, steps
            self.model(**self.get_inputs("Warm-up pass", 1))
        
        modelPath = os.path.join(self.modelDir, f'{self.pipeline}.pt')
        self.discriminator = torch.load(modelPath).cuda().eval()
        
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
            self.modelName = modelName
            if int(infer_level) in self.stage_model_args:
                self.model, self.g_scale, self.n_steps = self.stage_model_args[int(infer_level)]
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

    
    def pipe_load(self, model_name: str, scheduler='default'):
        hf_cache_dir = "../../models"
        if model_name == 'sd21':
            model_id = "stabilityai/stable-diffusion-2-1-base"
            scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
            pipe = DiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            g_scale = None
            n_steps = 50
        elif model_name == 'sdxl':
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", cache_dir=hf_cache_dir)
            g_scale = None
            n_steps = 50
        elif model_name == 'sdv15':
            model_id = "runwayml/stable-diffusion-v1-5"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", cache_dir=hf_cache_dir)
            g_scale = None
            n_steps = 50
        elif model_name == 'sdxl-turbo':
            model_id = "stabilityai/sdxl-turbo"
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            g_scale = 0.0
            n_steps = 1
        elif model_name == 'sdv15-lcm':
            model_id = "Lykon/dreamshaper-7"
            adapter_id = "latent-consistency/lcm-lora-sdv1-5"
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights(adapter_id)
            pipe.fuse_lora()
            g_scale = 0
            n_steps = 4
        elif model_name == 'sdxl-lcm':
            unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", torch_dtype=torch.float16, variant="fp16")
            pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16")
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            g_scale = 8.0
            n_steps = 4
        elif model_name in ['sdxl-lightning', 'sdxlltn']:
            base = "stabilityai/stable-diffusion-xl-base-1.0"
            repo = "ByteDance/SDXL-Lightning"
            ckpt = "sdxl_lightning_2step_unet.safetensors"
            unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
            unet.load_state_dict(load_file(hf_hub_download(repo, ckpt, cache_dir=hf_cache_dir), device="cuda"))
            pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16", cache_dir=hf_cache_dir)
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
            g_scale = 0
            n_steps = 2
        elif model_name == 'sdturbo':
            model_id = "stabilityai/sd-turbo"
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", cache_dir=hf_cache_dir)
            g_scale = 0.0
            n_steps = 1
        elif model_name == 'sdxs':
            model_id = "IDKiro/sdxs-512-0.9"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir=hf_cache_dir)
            g_scale = 0
            n_steps = 1
        elif model_name == 'tinysd':
            model_id = "segmind/tiny-sd"
            pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            g_scale = None
            n_steps = 30
        elif model_name == 'sd35turbo':
            model_id = "stabilityai/sd-turbo"
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", cache_dir=hf_cache_dir)
            n_steps = 4
        elif model_name == 'sd35med':
            model_id = "stabilityai/stable-diffusion-3-medium"
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", cache_dir=hf_cache_dir)
            n_steps = 50
        elif model_name == 'sd35large':
            model_id = "stabilityai/stable-diffusion-3-large"
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", cache_dir=hf_cache_dir)
            n_steps = 50
        if not scheduler == 'default':
            if scheduler == 'dpms++':
                pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
                n_steps = 20
        pipe.to("cuda")
        # print(pipe)
        return pipe, g_scale, n_steps
    
    
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
            if self.do_simulate:
                bs = batch_size if batch_size in [1,2,3,4,5,6,7,8,16,32] else 4
                time.sleep(self.profiled_runtimes[(self.modelName, self.modelName, bs)] / MSECONDS_IN_SEC)
                results = torch.randn(bs, 3, 224, 224)
            else:
                results = self.model(**self.get_inputs(data_prompts, batch_size))
            inference_time = time.time() - start_time
            print(f'\tProcess 2, inference time: {(inference_time):.6f}')
            logging.info(f'\tInference time: {(inference_time):.6f}')

            # Verify the qualify of the image by the discriminator
            start_time = time.time()
            results_qualified = [] # 1 for qualified, and 0 for non-qualified which need to be sent to 2nd level workers
            
            if isinstance(results, torch.Tensor):
                image_tensors = results
            else:
                image_tensors = torch.stack([transform(results.images[i]) for i in range(batch_size)])
            image_tensors = image_tensors.cuda()
            softmax = nn.Softmax()
            if self.do_simulate:
                time.sleep(0.01)
            else:
                conf_scores = self.discriminator(image_tensors)
                # conf_scores = softmax(conf_scores)
                # conf_fake, conf_real = conf_scores[:,0], conf_scores[:,1]
            if ALLOW_RANDOM: # Random assiging confidence score
                rng = np.random.default_rng()
                conf_scores = [rng.uniform(0.0,1.0) for i in conf_idx]
                abs_conf_thres = self.conf_thres
            else:
                conf_scores = self.conf_dist[conf_idx] # directly get data from pre-computed files
                sorted_dist = np.sort(self.conf_dist)
                index = max(int(len(sorted_dist) * self.conf_thres) - 1, 0)
                abs_conf_thres = sorted_dist[index]
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
    
    
    def serviceQueueLoop(self):
        # time.sleep(5)
        logging.info('ServiceQueue event loop waiting')
        logging.info(f'Profiled_runtimes: {self.profiled_runtimes}')
        while True:
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

