# This is the Controller microservice
import argparse
import sys, os
from pathlib import Path
sys.path.append('..')
import csv
import grpc
import logging
import pandas as pd
import numpy as np
import pickle
import threading
import time
from concurrent import futures
from enum import Enum
from common.app import App, AppNode, registerApplication
from protos import controller_pb2, controller_pb2_grpc
from protos import load_balancer_pb2, load_balancer_pb2_grpc
from protos import worker_pb2, worker_pb2_grpc
from diffserve_cascade_ILP import DiffServeILPAllocator
from qaware_multi_cascade_ILP import solve_milp_loop, create_lookup_table
from qaware_multi_cascade_ILP import solve_proteus_milp
from infaas import solve_infaas_accuracy
from tables import load_cascade_table, AVAILABLE as CASCADE_TABLES, DEFAULT as DEFAULT_CASCADE_TABLE
import config
from config import get_cas_exec, set_cas_exec, get_model_order


EWMA_WINDOW = 20
EWMA_ALPHA = 0.7
DEFAULT_EWMA_DECAY = 2.1
USECONDS_IN_SEC = 1000 * 1000
SLO_FACTOR = 1
OVER_PROVISION_FACTOR = 1.05
DO_CACHE = False
DO_QUEUE_TIME = True
DIFFSERVE_DEMAND_WEIGHT = 1.0


class AllocationPolicy(Enum):
    ''' The six systems compared in the paper, numbered in the order they appear
        in Figure 7's legend.  HADIS keeps its historical value of 5.
    '''
    CLIPPER_LIGHT = 0     # static, all workers on the lightest model
    CLIPPER_HEAVY = 1     # static, all workers on the heaviest model
    INFAAS_ACC = 2        # queue-reactive single-variant selection
    PROTEUS = 3           # MILP over variants, no per-query cascade
    DIFFSERVE = 4         # fixed light/heavy pair, discriminator only
    HADIS = 5             # hybrid routing + adaptive pair + resource allocation


# How far past a query's latency SLO a worker may still start it. The worker's own
# default is 1.5 (model.py): it will begin a query it expects to finish within
# 1.5x the SLO, even though the controller scores anything over 1.0x as a
# violation. Under saturation that spends the entire capacity on queries that are
# already doomed, and the backlog behind them ages further. Clipper-Heavy is the
# one system that runs saturated for the whole trace, so it admits only what it
# can deliver on time. Policies absent from this dict send no value and the worker
# keeps the inherited default, so their behaviour is unchanged.
SLO_ADMISSION_FACTOR = {AllocationPolicy.CLIPPER_HEAVY: 1.0}


class WorkerEntry:
    def __init__(self, IP: str, port: str, hostID: str, connection: grpc.insecure_channel,
                 model: str=None, task: str=None, appID: str=None):
        self.IP = IP
        self.port = port
        self.hostID = hostID
        self.connection = connection
        self.model = model
        self.task = task
        self.appID = appID
        self.currentLoad = 0
        self.queueSize = 0
        self.infer_level = 0
        self.conf_thres = 0
        self.router_thres = 0
        self.batch_size = 1
        self.scheduler = 'ddim'
        self.onCUDA = None

    def setModel(self, model, task, appID):
        self.model = model
        self.task = task
        self.appID = appID
        
    def setLevel(self, infer_level):
        self.infer_level = infer_level
        
    def setConfThres(self, conf_thres):
        self.conf_thres = conf_thres

    def setRouterThres(self, router_thres):
        self.router_thres = router_thres
        
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size


class Controller(controller_pb2_grpc.ControllerServicer):
    def __init__(self, allocationPolicy: AllocationPolicy,
                 cascadeTable: str = DEFAULT_CASCADE_TABLE):
        self.lbIP = None
        self.lbPort = None
        self.lbConnection = None
        # key: hostID, value: WorkerEntry
        self.workers = {}
        # Time in seconds after which controller is invoked
        self.period = 1
        # Re-plan every ALLOC_INTERVAL_TRACE_SEC of trace time: 5 ticks in
        # profile-driven mode (10x compressed), 50 in real mode.
        self.allocIntervalTicks = config.get_alloc_interval_ticks(self.period)
        
        # overall system demand and EWMA value for the entire system
        self.total_demand_history = [] # List to store system-wide demand over time
        # self.ewma_demand_per_task = {'sdturbo': [0], 'sdv15': [0]}
        # self.ewma_queue_length_per_task = {'sdturbo': [0], 'sdv15': [0]}
        # self.demand_per_task_history = {'sdturbo': [0], 'sdv15': [0]}
        # self.queue_length_per_task_history = {'sdturbo': [0], 'sdv15': [0]}
        # self.system_ewma = 0 # system-wide EWMA for total demand
        # self.coming_query_per_task = {'sdturbo': 0, 'sdv15': 0, 'sink': 0}
        model_order = get_model_order()
        self.ewma_demand_per_task = {m: [0] for m in model_order}
        self.ewma_queue_length_per_task = {m: [0] for m in model_order}
        self.demand_per_task_history = {m: [0] for m in model_order}
        self.queue_length_per_task_history = {m: [0] for m in model_order}
        self.system_ewma = 0
        self.coming_query_per_task = {m: 0 for m in model_order + ['sink']}
        self.cache_resource_alloc_plan = {} # key: (sys_demand, queue_length)

        self.allocationPolicy = allocationPolicy
        # None -> do not send the field, so the worker keeps its own default
        self.sloAdmissionFactor = SLO_ADMISSION_FACTOR.get(allocationPolicy)
        self.allocationMetadata = {}
        self.cas_alg = None
        self.conf_thres = 1
        self.router_thres = 0
        self.model_indices = None

        # TODO: update execution and branching profiles based on real-time data
        self.executionProfiles = None
        self.branchingProfiles = None
        
        # save request processing time, for the computation of SLO violations
        self.queriesProcessed = set()
        self.queriesStartTime = {}
        self.queriesIntermediateTime = {}
        self.queriesEndTime = {}
        self.slo_timeouts = {'succeed': 0, 'timeout': 0, 'drop': 0, 'total': 0}
        self.save_slo_timeouts_per_second = f'../../logs/slo_timeouts_per_second.csv'
        self.save_query_num_per_second = f"../../logs/query_num_per_second.csv"
        self.save_conf_thres_per_second = f"../../logs/conf_thres_per_second.csv"
        self.save_router_thres_per_second = f"../../logs/router_thres_per_second.csv"
        self.save_cas_config_per_second = f"../../logs/cascade_config_per_second.csv"
        self.save_query_latency = f"../../logs/query_latency.txt"
        self.query_latency_since_recorded = []

        # TODO: allow all models, perhaps use model families as well?
        # self.allocatedModels = {'yolov5m': 0, 'eb6': 0, 'sink': 0}
        self.allocatedModels = {}
        for model in model_order + ['sink']:
            if model not in self.allocatedModels:
                self.allocatedModels[model] = 0
        
        self.lightWeightModels = ['sdxl-turbo', 'sdv15-lcm', 'sdxl-lcm', 
                          'sdxl-lightning', 'sdxs', 'tinysd', 'sdturbo']
        self.heavyWeightModels =  ['sd21', 'sdxl', 'sdv15']
        
        logging.info(config.mode_banner('controller'))

        # Pareto cascade configuration table (offline profiling output). Built
        # once here rather than on every planning cycle.
        comb_thres_config, fid_config = load_cascade_table(cascadeTable)
        self.cascadeTable = cascadeTable
        self.lookup_table = create_lookup_table(comb_thres_config, fid_config,
                                                total_models=len(model_order))
        logging.info(f'Cascade table: {cascadeTable} '
                     f'({len(comb_thres_config)} configurations)')

        self.apps = [registerApplication(config.get_app_json())]
        # {(model, batch): seconds}; see config.get_controller_runtimes() for a
        # note on its rounding granularity.
        self.profiled_runtimes = config.get_controller_runtimes()
        logging.info(f'Profiled runtimes: {self.profiled_runtimes}')


        try:
            eventLoopThread = threading.Thread(target=self.eventLoop)
            eventLoopThread.daemon = True
            eventLoopThread.start()
            # while eventLoopThread.is_alive():
            #     eventLoopThread.join(1)
        except KeyboardInterrupt:
            sys.exit(1)
        
    
    def WorkerSetup(self, request, context):
        try:
            logging.info(f'Trying to establish GRPC connection with worker..')
            logging.info(f'context.peer(): {context.peer()}')
            splitContext = context.peer().split(':')
            if splitContext[0] == 'ipv4':
                workerIP = splitContext[1]
            else:
                workerIP = 'localhost'
            connection = grpc.insecure_channel(f'{workerIP}:{request.hostPort}')
            workerEntry = WorkerEntry(IP=workerIP, port=request.hostPort,
                                      hostID=request.hostID, connection=connection,
                                      model=None)
            self.workers[request.hostID] = workerEntry
            logging.info(f'Established GRPC connection with worker (hostID: '
                         f'{request.hostID}, IP: {workerIP}, port: '
                         f'{request.hostPort})')
            
            return controller_pb2.RegisterWorkerResponse(lbIP=self.lbIP,
                                                         lbPort=self.lbPort)
        except Exception as e:
            message = f'Exception while setting up worker: {str(e)}'
            logging.exception(message)
            return controller_pb2.RegisterWorkerResponse(lbIP=None, lbPort=None,
                                                         message=message)
    

    def LBSetup(self, request, context):
        try:
            logging.info(f'context.peer(): {context.peer()}')
            splitContext = context.peer().split(':')
            if splitContext[0] == 'ipv4':
                lbIP = splitContext[1]
            else:
                lbIP = 'localhost'
            logging.info(f'Trying to establish GRPC connection with load balancer..')
            connection = grpc.insecure_channel(f'{lbIP}:{request.lbPort}')
            self.lbConnection = connection
            self.lbIP = lbIP
            self.lbPort = request.lbPort
            logging.info(f'Established GRPC connection with load balancer '
                         f'(IP: {lbIP}, port: {request.lbPort})')
            
            return controller_pb2.RegisterLBResponse(message='Done!')
        
        except Exception as e:
            message = f'Exception while setting up load balancer: {str(e)}'
            logging.exception(message)
            return controller_pb2.RegisterLBResponse(message=message)
    

    def eventLoop(self):
        clockCounter = 0
        prev_slo_timeouts = {k:v for k,v in self.slo_timeouts.items()}
        check_start_time = time.time()
        while True:
            self.checkLBHeartbeat()

            cur_workers = {k:v for k,v in self.workers.items()}
            for hostID in cur_workers:
                # worker = self.workers[hostID]
                worker = cur_workers[hostID]
                self.checkWorkerHeartbeat(hostID, worker)
                
            # Estimate the future system demand
            self.computeSystemEWMA()
            
            # This doesn't necessary have to run every time Controller checks
            # heartbeats
            if clockCounter == 0:
                logging.info(f'Re-allocating every {self.allocIntervalTicks} ticks '
                             f'({self.allocIntervalTicks * self.period}s wall, '
                             f'{config.ALLOC_INTERVAL_TRACE_SEC}s trace time)')
            if clockCounter % self.allocIntervalTicks == 0:
                self.allocateResources()
            clockCounter += 1
            
            # compute slo timeout per second
            self.computeSLOTimeoutsPerSec()
            logging.info(f'slo_timeouts: {self.slo_timeouts}')
            logging.info(f'prev_slo_timeouts: {prev_slo_timeouts}')
            logging.info(f'length of queriesStartTime: {len(self.queriesStartTime)}, '
                         f'length of queriesIntermediateTime: {len(self.queriesIntermediateTime)}, '
                         f'length of queriesEndTime: {len(self.queriesEndTime)}')
            # save slo timeouts and threshold
            if self.system_ewma > 0:
                self.saveResultsToCSV(self.slo_timeouts, prev_slo_timeouts)
                self.coming_query_per_task = {m: 0 for m in get_model_order() + ['sink']}
                if len(self.query_latency_since_recorded) > 0:
                    self.saveQueryLatency()
            prev_slo_timeouts = {k:v for k,v in self.slo_timeouts.items()}

            time_difference = time.time() - check_start_time
            sleep_time = self.period - time_difference
            time.sleep(sleep_time)
            check_start_time = time.time()
            logging.debug(f'Woke up from sleep {sleep_time}, running eventLoop again..')
            
            
    def computeSLOTimeoutsPerSec(self):
        # update dropped queries
        expire_time = self.apps[0].getLatencySLO() / USECONDS_IN_SEC * 2 * SLO_FACTOR
        
        popped = []
        for requestID in self.queriesStartTime:
            time_diff = time.time() - self.queriesStartTime[requestID]
            # logging.info(f"time_diff: {time_diff}, expire_time: {expire_time}")
            if requestID in self.queriesProcessed:
                popped.append(requestID)
            elif expire_time < time_diff:
                popped.append(requestID)
                self.slo_timeouts['drop'] += 1
                self.slo_timeouts['total'] += 1
                self.queriesProcessed.add(requestID)
        for requestID in popped:
            self.queriesStartTime.pop(requestID)
        
    def saveQueryLatency(self):
        p = Path(self.save_query_latency)
        mode = 'a' if p.exists() else 'w'
        arr = np.asarray(self.query_latency_since_recorded)
        with open(p, mode) as f:
            np.savetxt(f, arr, fmt="%.3f")
        self.query_latency_since_recorded = []

    def saveResultsToCSV(self, slo_timeouts, prev_slo_timeouts):
        slo_timeouts_per_second = []
        for key in slo_timeouts:
            slo_timeouts_per_second.append(slo_timeouts[key] - prev_slo_timeouts[key])
            
        model_order = get_model_order()
        model_header = model_order
        model_row = [self.coming_query_per_task[m] for m in model_order]
            
        for csv_config in [(self.save_slo_timeouts_per_second, ['succeed', 'timeout', 'drop', 'total'], slo_timeouts_per_second), 
                        #    (self.save_conf_thres_per_second, ['threshold'], [self.conf_thres]),
                        #    (self.save_router_thres_per_second, ['threshold'], [self.router_thres]),
                           (self.save_cas_config_per_second, ['models', 'router_thres', 'conf_thres'], [self.model_indices, self.router_thres, self.conf_thres]),
                           (self.save_query_num_per_second, model_header, model_row)]:
            csv_name, csv_hearder, csv_new_row = csv_config
            file_exists = os.path.exists(csv_name)
            with open(csv_name, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists or os.stat(csv_name).st_size==0:
                    writer.writerow(csv_hearder)
                writer.writerow(csv_new_row)
            
    def computeSystemEWMA(self):
        total_system_demand = 0
        model_order = get_model_order()
        queue_length_per_task = {m: 0 for m in model_order + ['sink']}
        demand_per_task = {m: 0 for m in model_order + ['sink']}
        
        cur_workers = {k:v for k,v in self.workers.items()}
        for hostID in cur_workers:
            # worker = self.workers[hostID]
            worker = cur_workers[hostID]
            # total_system_demand += worker.demand
            if worker.model:
                queue_length_per_task[worker.model] += worker.queueSize
                demand_per_task[worker.model] += worker.currentLoad
                
        # demand_per_task['sdturbo'] = (len(self.queriesStartTime) + len(self.queriesEndTime))
        # demand_per_task['sdv15'] = len(self.queriesIntermediateTime)
        num_total_queries = len(self.queriesStartTime) + len(self.queriesEndTime)
        total_system_demand = num_total_queries * OVER_PROVISION_FACTOR  # amplified for SLO protection
        
        if total_system_demand > 0:
            for task in self.queue_length_per_task_history:
                self.queue_length_per_task_history[task].append(queue_length_per_task[task])
                self.demand_per_task_history[task].append(demand_per_task[task])
                
                if len(self.queue_length_per_task_history[task]) > EWMA_WINDOW:
                    self.queue_length_per_task_history[task] = self.queue_length_per_task_history[task][1:]
                if len(self.demand_per_task_history[task]) > EWMA_WINDOW:
                    self.demand_per_task_history[task] = self.demand_per_task_history[task][1:]
                
                half_life_steps = 3
                # compute ewma
                df = pd.DataFrame({'demand': self.demand_per_task_history[task]})
                # ewma = df.ewm(com=DEFAULT_EWMA_DECAY).mean()
                ewma = df.ewm(halflife=half_life_steps, adjust=False).mean()
                ewma_value = ewma['demand'].to_list()[-1]
                self.ewma_demand_per_task[task] = ewma_value
                
                df = pd.DataFrame({'queue_length': self.queue_length_per_task_history[task]})
                # ewma = df.ewm(com=DEFAULT_EWMA_DECAY).mean()
                ewma = df.ewm(halflife=half_life_steps, adjust=False).mean()
                ewma_value = ewma['queue_length'].to_list()[-1]
                self.ewma_queue_length_per_task[task] = ewma_value
                
            self.total_demand_history.append(total_system_demand)
            if len(self.total_demand_history) > EWMA_WINDOW:
                self.total_demand_history = self.total_demand_history[1:]
            # Apply EWMA to the total system demand
            df = pd.DataFrame({'demand': self.total_demand_history})
            # ewma = df.ewm(com=DEFAULT_EWMA_DECAY).mean()
            ewma = df.ewm(halflife=half_life_steps, adjust=False).mean()
            ewma_value = ewma['demand'].to_list()[-1]
            self.system_ewma = ewma_value
        logging.info(f"System-wide Demand = {total_system_demand}, EWMA = {self.system_ewma}")
        
        
    def getSystemDemand(self):
        # return the current system-wide demand and EWMA
        return self.system_ewma
    
    
    def allocateResources(self):
        ''' Run the resource allocation algorithm with the appropriate policy
        '''
        if self.allocationPolicy == AllocationPolicy.CLIPPER_LIGHT:
            self.allocateByStaticModel(config.CLIPPER_LIGHT_MODEL)
        elif self.allocationPolicy == AllocationPolicy.CLIPPER_HEAVY:
            self.allocateByStaticModel(config.CLIPPER_HEAVY_MODEL)
        elif self.allocationPolicy == AllocationPolicy.INFAAS_ACC:
            self.allocateByINFaaSAccuracy()
        elif self.allocationPolicy == AllocationPolicy.PROTEUS:
            self.allocateByProteusILP()
        elif self.allocationPolicy == AllocationPolicy.DIFFSERVE:
            self.allocateByDiffServeILP()
        elif self.allocationPolicy == AllocationPolicy.HADIS:
            self.allocateByMultiCascadeAlg()
        else:
            raise Exception(f'Unknown allocation policy: {self.allocationPolicy}')
        return
    
    # Update multi-level model cascade
    def allocateByMultiCascadeAlg(self, do_static=False):
        app = self.apps[0]
        logging.info(f'AllocatedModels before ReAlloc: {self.allocatedModels}')
        if DO_CACHE:
            logging.info(f"Cached Resource Allocation Plan: {self.cache_resource_alloc_plan}")

        # PATCH: multi-model MILP-based cascade allocation
        model_order = get_model_order()
        model_index_map = {name: i for i, name in enumerate(model_order)}
        num_workers = len(self.workers)
        sys_demand = self.system_ewma
        slo = app.getLatencySLO() / USECONDS_IN_SEC

        # latency_table should be in format {(model_index, batch_size): latency}
        latency_table = {(model_index_map[name], bs): latency for (name,bs),latency in self.profiled_runtimes.items()}
        lookup_table = self.lookup_table
        demand_per_model = {model_index_map[k]:v for k,v in self.ewma_demand_per_task.items()}
        queue_length_per_model = {model_index_map[k]:v for k,v in self.ewma_queue_length_per_task.items()}
        logging.info(f'queue_lengh_per_model_ewma: {queue_length_per_model}, demand_per_model_ewma: {demand_per_model}')

        milp_result = solve_milp_loop(
            total_servers=num_workers,
            sysDemand=sys_demand,
            latencySLO=slo,
            lookup_table=lookup_table,
            model_latency_table=latency_table,
            demand_per_model=demand_per_model,
            queue_length_per_model=queue_length_per_model,
            do_queue_time=DO_QUEUE_TIME,
            do_cache=DO_CACHE,
            cache_resource_alloc_plan=self.cache_resource_alloc_plan
        )

        required_workers = {}
        batch_sizes_dict = {}
        route_ratio_dict = {}

        if milp_result is not None: 
            for model_idx in milp_result["models"]:
                model_name = model_order[model_idx]
                x = milp_result["device_allocation"][model_idx]
                b = milp_result["batch_sizes"][model_idx]
                t = milp_result['route_ratio'][model_idx]
                required_workers[model_name] = x
                batch_sizes_dict[model_name] = b
                route_ratio_dict[model_name] = t
            self.conf_thres = milp_result["conf_thres"]
            self.router_thres = milp_result["router_thres"]
            self.model_indices = milp_result["models"] # Just used for logging
        else:
            logging.warning("MILP failed: serving not started or no feasible solution.")
            num_avail_workers = len([w for w in self.workers.values() if w.onCUDA])
            # Only models present in the lookup table may be used, so that ablations
            # restricted to a subset of the cascade (e.g., a fixed model pair) stay pure
            active_model_indices = sorted({i for comb in lookup_table["model_combination"]
                                           for i, on in enumerate(comb) if on})
            for model in model_order:
                required_workers[model] = 0
            # Shed load: all workers on the lightest available model with no escalation,
            # so that backlogged queues can drain and the MILP becomes feasible again
            lightest = model_order[active_model_indices[0]]
            required_workers[lightest] = num_avail_workers
            batch_sizes_dict[lightest] = 1
            route_ratio_dict[lightest] = 1.0
            self.conf_thres = 0
            self.router_thres = 0
            self.model_indices = None

        logging.info(f'Model plan: {required_workers}')
        logging.info(f'confidence thresholds: {self.conf_thres}')
        logging.info(f'router thresholds: {self.router_thres}')

        # BEGIN model reloading loop
        cur_workers = {k: v for k, v in self.workers.items()}
        available_models = list(required_workers.keys())
        for hostID, worker in cur_workers.items():
            if worker.onCUDA:
                try:
                    # Rotate through available_models by usage (simple heuristic here)
                    for idx, model in enumerate(available_models):
                        if required_workers[model] > 0:
                            batch_size = batch_sizes_dict[model]
                            
                            # conf_thres = self.conf_thres[model]
                            if idx < len(available_models) - 1:
                                # conf_thres = self.conf_thres[available_models[idx+1]]
                                conf_thres = self.conf_thres
                                is_lightweight = 1
                            else:
                                conf_thres = 0 # here the threshold means the ratio of queries routed to the next stage
                                is_lightweight = 0
                            
                            infer_level = model_index_map[model]
                            logging.info(f'Trying to load model {model} on worker {hostID}, batch size: {batch_size}')
                            self.loadModelOnWorker(worker, model, infer_level=infer_level, batch_size=batch_size, is_lightweight=is_lightweight, router_thres=self.router_thres, conf_thres=conf_thres)
                            required_workers[model] -= 1
                            break
                except Exception as e:
                    logging.exception(f'Failed to assign model to worker {hostID}: {e}')
            else:
                try:
                    self.loadModelOnWorker(worker, 'sink', infer_level=len(model_order), batch_size=1, is_lightweight=0, router_thres=0, conf_thres=0)
                except Exception as e:
                    logging.exception(f'Failed to load sink on CPU worker {hostID}: {e}')

        logging.info(f'AllocatedModels after ReAlloc: {self.allocatedModels}')
        return

    
    def allocateByProteusILP(self):
        app = self.apps[0]
        slo = app.getLatencySLO() / USECONDS_IN_SEC  # in seconds
        model_order = get_model_order()
        latency_table = {(m, b): self.profiled_runtimes[(m, b)] for (m, b) in self.profiled_runtimes if m in model_order}
        num_workers = len(self.workers)
        ewma_demand = self.system_ewma

        # Assign higher weights to heavier models (proxy for FID/accuracy)
        fid_weighting = {
            'sdxlltn': 29.76,
            'sd35turbo': 25.13,
            'sd35med': 20.61,
            'sd35large': 19.95
        }
        required_workers = {}
        batch_sizes_dict = {}
        
        logging.info(f'queue_lengh_per_model_ewma: {self.ewma_demand_per_task}, demand_per_model_ewma: {self.ewma_queue_length_per_task}')
        milp_result = solve_proteus_milp(latency_table, num_workers, slo, ewma_demand, fid_weighting, 
                                        self.ewma_demand_per_task, self.ewma_queue_length_per_task)
        if milp_result is not None: 
            required_workers = milp_result["device_allocation"]
            batch_sizes_dict = milp_result["batch_sizes"]
        else:
            logging.warning("MILP failed: serving not started or no feasible solution.")
            num_avail_workers = len([w for w in self.workers.values() if w.onCUDA])
            for model in model_order:
                required_workers[model] = 0
            for i in range(num_avail_workers):
                model = model_order[i % len(model_order)]
                required_workers[model] += 1
                batch_sizes_dict[model] = 1
        
        logging.info(f"[ProteusILP] Required workers per model: {required_workers}")
        logging.info(f"[ProteusILP] Batch sizes per model: {batch_sizes_dict}")
        logging.info(f"[ProteusILP] AllocatedModels BEFORE loading: {self.allocatedModels}")

        cur_workers = {k: v for k, v in self.workers.items()}
        available_models = list(required_workers.keys())

        for hostID, worker in cur_workers.items():
            if worker.onCUDA:
                try:
                    for idx, model in enumerate(available_models):
                        if required_workers[model] > 0:
                            batch_size = batch_sizes_dict[model]
                            logging.info(f"[ProteusILP] Loading model {model} on worker {hostID}, batch size: {batch_size}")
                            self.loadModelOnWorker(
                                worker,
                                model,
                                infer_level=0, # Proteus: all models use level 0
                                batch_size=batch_size,
                                is_lightweight=0, # Proteus: all direct inference
                                router_thres=0.0, # Proteus disables router
                                conf_thres=1.0 # Proteus disables discriminator
                            )
                            required_workers[model] -= 1
                            break
                except Exception as e:
                    logging.exception(f"[ProteusILP] Failed to assign model to worker {hostID}: {e}")
            else:
                try:
                    self.loadModelOnWorker(worker, model='sink', infer_level=len(model_order), batch_size=1, is_lightweight=0, router_thres=0, conf_thres=0)
                except Exception as e:
                    logging.exception(f"[ProteusILP] Failed to load sink on CPU worker {hostID}: {e}")
        logging.info(f"[ProteusILP] AllocatedModels AFTER loading: {self.allocatedModels}")


    def allocateByINFaaSAccuracy(self):
        app = self.apps[0]
        slo = app.getLatencySLO() / USECONDS_IN_SEC
        model_order = get_model_order()
        latency_table = {(m, b): self.profiled_runtimes[(m, b)] for (m, b) in self.profiled_runtimes if m in model_order}
        num_workers = len(self.workers)
        ewma_demand = self.system_ewma

        fid_weighting = {
            'sdxlltn': 29.76,
            'sd35turbo': 25.13,
            'sd35med': 20.61,
            'sd35large': 19.95
        }

        required_workers = {}
        batch_sizes_dict = {}

        logging.info(f'[INFaaS] demand={ewma_demand}, queue_per_model={self.ewma_queue_length_per_task}')
        result = solve_infaas_accuracy(latency_table, num_workers, slo, ewma_demand, fid_weighting,
                                       self.ewma_demand_per_task, self.ewma_queue_length_per_task)
        if result is not None:
            required_workers = result["device_allocation"]
            batch_sizes_dict = result["batch_sizes"]
        else:
            logging.warning("[INFaaS] No feasible solution: falling back to round-robin assignment.")
            num_avail_workers = len([w for w in self.workers.values() if w.onCUDA])
            for model in model_order:
                required_workers[model] = 0
            for i in range(num_avail_workers):
                model = model_order[i % len(model_order)]
                required_workers[model] += 1
                batch_sizes_dict[model] = 1

        logging.info(f"[INFaaS] Required workers: {required_workers}")
        logging.info(f"[INFaaS] Batch sizes: {batch_sizes_dict}")
        logging.info(f"[INFaaS] AllocatedModels BEFORE loading: {self.allocatedModels}")

        cur_workers = {k: v for k, v in self.workers.items()}
        available_models = list(required_workers.keys())

        for hostID, worker in cur_workers.items():
            if worker.onCUDA:
                try:
                    for model in available_models:
                        if required_workers[model] > 0:
                            batch_size = batch_sizes_dict[model]
                            logging.info(f"[INFaaS] Loading model {model} on worker {hostID}, batch_size={batch_size}")
                            self.loadModelOnWorker(
                                worker,
                                model,
                                infer_level=0,
                                batch_size=batch_size,
                                is_lightweight=0,
                                router_thres=0.0,
                                conf_thres=1.0
                            )
                            required_workers[model] -= 1
                            break
                except Exception as e:
                    logging.exception(f"[INFaaS] Failed to assign model to worker {hostID}: {e}")
            else:
                try:
                    self.loadModelOnWorker(worker, model='sink', infer_level=len(model_order), batch_size=1, is_lightweight=0, router_thres=0, conf_thres=0)
                except Exception as e:
                    logging.exception(f"[INFaaS] Failed to load sink on CPU worker {hostID}: {e}")

        logging.info(f"[INFaaS] AllocatedModels AFTER loading: {self.allocatedModels}")


    def allocateByDiffServeILP(self):
        ''' DiffServe: a fixed lightweight/heavyweight pair with discriminator-based
            re-routing.  The ILP jointly sets the workers per stage, the batch size
            per stage, and the discriminator threshold.  The router is disabled
            (is_lightweight=0), which is the difference from HADIS.
        '''
        app = self.apps[0]
        model_order = get_model_order()
        slo = app.getLatencySLO() / USECONDS_IN_SEC
        num_workers = len(self.workers) if len(self.workers) > 1 else 1

        if self.cas_alg is None:
            self.cas_alg = DiffServeILPAllocator(self.profiled_runtimes,
                                                 total_servers=num_workers)
            required_workers, batch_sizes_dict, self.conf_thres = self.cas_alg.initialize()
        else:
            self.cas_alg.update_num_servers(num_workers)
            logging.info(f'queue_length_per_task_ewma: {self.ewma_queue_length_per_task}, '
                         f'demand_per_task_ewma: {self.ewma_demand_per_task}')
            required_workers, batch_sizes_dict, self.conf_thres = self.cas_alg.iterate(
                self.system_ewma, slo, self.ewma_demand_per_task,
                self.ewma_queue_length_per_task, demand_weight=DIFFSERVE_DEMAND_WEIGHT)

        # DiffServe has no router, and its cascade is the fixed pair
        self.router_thres = 0
        self.model_indices = [self.cas_alg.light_level, self.cas_alg.heavy_level]
        levels = {self.cas_alg.light: self.cas_alg.light_level,
                  self.cas_alg.heavy: self.cas_alg.heavy_level}

        logging.info(f'[DiffServe] Required workers: {required_workers}')
        logging.info(f'[DiffServe] Batch sizes: {batch_sizes_dict}')
        logging.info(f'[DiffServe] Discriminator threshold: {self.conf_thres}')

        cur_workers = {k: v for k, v in self.workers.items()}
        for hostID, worker in cur_workers.items():
            if worker.onCUDA:
                try:
                    for model in [self.cas_alg.light, self.cas_alg.heavy]:
                        if required_workers.get(model, 0) > 0:
                            # Only the lightweight stage escalates; the heavyweight
                            # stage sends everything to the sink (conf_thres=0).
                            is_light = (model == self.cas_alg.light)
                            self.loadModelOnWorker(
                                worker, model,
                                infer_level=levels[model],
                                batch_size=batch_sizes_dict[model],
                                is_lightweight=0,          # router disabled
                                router_thres=0.0,
                                conf_thres=self.conf_thres if is_light else 0)
                            required_workers[model] -= 1
                            break
                except Exception as e:
                    logging.exception(f'[DiffServe] Failed to assign model to worker {hostID}: {e}')
            else:
                try:
                    self.loadModelOnWorker(worker, 'sink', infer_level=len(model_order),
                                           batch_size=1, is_lightweight=0,
                                           router_thres=0, conf_thres=0)
                except Exception as e:
                    logging.exception(f'[DiffServe] Failed to load sink on CPU worker {hostID}: {e}')

        logging.info(f'[DiffServe] AllocatedModels after ReAlloc: {self.allocatedModels}')
        return


    def allocateByStaticModel(self, model_index):
        ''' Clipper: every GPU worker runs the same model variant, with no dynamic
            selection and no query-aware routing.  -ap 0 pins the lightest model
            (Clipper-Light), -ap 1 the heaviest (Clipper-Heavy).  Batch size is the
            largest whose profiled latency still fits the SLO, the same rule the
            other baselines use.
        '''
        model_order = get_model_order()
        model = model_order[model_index]
        slo = self.apps[0].getLatencySLO() / USECONDS_IN_SEC

        batch_size = 1
        for bs in config.get_allowed_batch_sizes(model):
            if self.profiled_runtimes[(model, bs)] <= slo:
                batch_size = bs

        # No cascade: every output is accepted and forwarded straight to the sink.
        self.conf_thres = 0
        self.router_thres = 0
        self.model_indices = [model_index]

        logging.info(f'[Clipper] All workers -> {model}, batch size {batch_size}')

        cur_workers = {k: v for k, v in self.workers.items()}
        for hostID, worker in cur_workers.items():
            try:
                if worker.onCUDA:
                    self.loadModelOnWorker(worker, model, infer_level=model_index,
                                           batch_size=batch_size, is_lightweight=0,
                                           router_thres=0, conf_thres=0)
                else:
                    self.loadModelOnWorker(worker, 'sink', infer_level=len(model_order),
                                           batch_size=1, is_lightweight=0,
                                           router_thres=0, conf_thres=0)
            except Exception as e:
                logging.exception(f'[Clipper] Failed to load model on worker {hostID}: {e}')

        logging.info(f'[Clipper] AllocatedModels after ReAlloc: {self.allocatedModels}')
        return


    def checkLBHeartbeat(self):
        try:
            connection = self.lbConnection
            stub = load_balancer_pb2_grpc.LoadBalancerStub(connection)
            message = 'Still alive?'
            request = load_balancer_pb2.LBHeartbeat(message=message)
            response = stub.LBAlive(request)
            logging.info(f'Heartbeat from load balancer received')
        except Exception as e:
            logging.warning('No heartbeat from load balancer')

    
    def checkWorkerHeartbeat(self, hostID: str, worker: WorkerEntry):
        try:
            connection = worker.connection
            stub = worker_pb2_grpc.WorkerDaemonStub(connection)
            message = 'Still alive?'
            request = worker_pb2.HeartbeatRequest(message=message)
            response = stub.Heartbeat(request)
            worker.currentLoad = response.queriesSinceHeartbeat
            worker.queueSize = response.queueSize
            worker.onCUDA = response.onCUDA
            branchingSinceHeartbeat = pickle.loads(response.branchingSinceHeartbeat)
            queriesTimestampSinceHearbeat = pickle.loads(response.queriesTimestampSinceHearbeat)
            self.coming_query_per_task[worker.model] += worker.currentLoad
            logging.info(f'Heartbeat from worker {hostID} received, model variant: '
                         f'{worker.model}, currentLoad: {worker.currentLoad}, total '
                         f'queries received: {response.totalQueries}, queue size: '
                         f'{worker.queueSize}, cuda available: {worker.onCUDA}, '
                         f'branching since heartbeat: {len(branchingSinceHeartbeat)}, '
                         f'queries timestamp since heartbeat: {len(queriesTimestampSinceHearbeat)}')
            # if lightweight model worker (infer_level=0), then save new reuqests processing time
            # if sink worker (infer_level=2), then update the requests processing time
            if worker.infer_level >= 0 and worker.infer_level < len(get_model_order()):
                for requestID in queriesTimestampSinceHearbeat:
                    if requestID in self.queriesEndTime:
                        endTime = self.queriesEndTime.pop(requestID)
                        processingTime = endTime - queriesTimestampSinceHearbeat[requestID]
                        if processingTime > self.apps[0].getLatencySLO() / USECONDS_IN_SEC * SLO_FACTOR:
                            self.slo_timeouts['timeout'] += 1
                        else:
                            self.slo_timeouts['succeed'] += 1
                        self.slo_timeouts['total'] += 1
                        self.queriesProcessed.add(requestID)
                        self.query_latency_since_recorded.append(processingTime)
                    else:
                        if requestID in self.queriesStartTime:
                            self.queriesIntermediateTime[requestID] = queriesTimestampSinceHearbeat[requestID]
                        else:
                            self.queriesStartTime[requestID] = queriesTimestampSinceHearbeat[requestID]
                        
            elif worker.infer_level == len(get_model_order()):
                popped = []
                for requestID in queriesTimestampSinceHearbeat:
                    if requestID in self.queriesStartTime:
                        popped.append(requestID)
                        startTime = self.queriesStartTime.pop(requestID)
                        processingTime = queriesTimestampSinceHearbeat[requestID] - startTime
                        if processingTime > self.apps[0].getLatencySLO() / USECONDS_IN_SEC * SLO_FACTOR:
                            self.slo_timeouts['timeout'] += 1
                        else:
                            self.slo_timeouts['succeed'] += 1
                        self.slo_timeouts['total'] += 1
                        self.queriesProcessed.add(requestID)
                        self.query_latency_since_recorded.append(processingTime)
                        # # Second check to avoid double-counted as dropped
                        # if requestID in self.queriesStartTime:
                        #     self.queriesStartTime.pop(requestID)
                        
                        if requestID in self.queriesIntermediateTime:
                            self.queriesIntermediateTime.pop(requestID)
                    else:
                        self.queriesEndTime[requestID] = queriesTimestampSinceHearbeat[requestID]
            
        except Exception as e:
            # TODO: remove worker after certain number of missed heartbeats?
            logging.warning(f'No heartbeat from worker: {hostID}')
            logging.exception(f'Exception while checking heartbeat for worker {hostID}: {e}')


    def loadModelOnWorker(self, worker: WorkerEntry, model: str, infer_level: int, batch_size: int, is_lightweight: int, router_thres=None, conf_thres=None):
        ''' Loads the given model on a worker
        '''
        try:
            previousModel = worker.model
            connection = worker.connection
            stub = worker_pb2_grpc.WorkerDaemonStub(connection)
            # TODO: hard-coded application index
            app = self.apps[0]
            appID = app.appID
            task = app.findTaskFromModelVariant(model)
            childrenTasks = pickle.dumps(app.getChildrenTasks(task))
            labelToChildrenTasks = pickle.dumps(app.getLabelToChildrenTasksDict(task))

            target_conf_thres = conf_thres if conf_thres is not None else self.conf_thres
            target_router_thres = router_thres if router_thres is not None else self.router_thres
            optional_fields = ({} if self.sloAdmissionFactor is None
                               else {'slo_admission_factor': self.sloAdmissionFactor})
            request = worker_pb2.LoadModelRequest(modelName=model,
                                                  schedulerName=worker.scheduler,
                                                  **optional_fields,
                                                  infer_level=infer_level,
                                                  conf_thres=target_conf_thres,
                                                  router_thres=target_router_thres,
                                                  batch_size=batch_size,
                                                  is_lightweight=is_lightweight,
                                                  applicationID=appID,
                                                  task=task,
                                                  childrenTasks=childrenTasks,
                                                  labelToChildrenTasks=labelToChildrenTasks)
            response = stub.LoadModel(request)
            logging.info(f'LOAD_MODEL_RESPONSE from host {worker.hostID}: {response.response}, '
                         f'{response.message}')
            
            # Model loaded without any errors
            if response.response == 0:
                self.allocatedModels[model] += 1
                if previousModel is not None:
                    self.allocatedModels[previousModel] -= 1
            # If there is an error while loading model, raise Exception
            else:
                raise Exception(f'Error occurred while loading model {model} on worker '
                                f'{worker.hostID}: {response.message}')
            
            worker.setModel(model, task, appID)
            worker.setLevel(infer_level)
            worker.setConfThres(self.conf_thres)
            worker.setRouterThres(self.router_thres)
            worker.setBatchSize = batch_size
        except Exception as e:
            raise e


def getargs():
    parser = argparse.ArgumentParser(description='Controller micro-service')
    parser.add_argument('--port', '-p', required=False, dest='port', default='50050',
                        help='Port to start the controller on')
    parser.add_argument('--allocation_policy', '-ap', required=True,
                        dest='allocationPolicy', choices=['0', '1', '2', '3', '4', '5'],
                        help=('System to run. 0: Clipper-Light, 1: Clipper-Heavy, '
                              '2: INFaaS-Acc, 3: Proteus, 4: DiffServe, 5: HADIS'))
    parser.add_argument('--cascade', '-c', required=False,
                       dest='cascadeExec', choices=['multi'], default='multi',
                       help=(f'The cascade pipeline to execute.'))
    parser.add_argument('--profile-driven', action='store_true', dest='profileDriven',
                        default=False,
                        help=('Run the profile-driven experiment (simulated execution, '
                              '10x scaled latencies, 6 s SLO). Omit for the real-execution '
                              'experiment. Every component must use the same setting.'))
    parser.add_argument('--cascade_table', '-t', required=False, dest='cascadeTable',
                        choices=CASCADE_TABLES, default=DEFAULT_CASCADE_TABLE,
                        help=('Cascade configuration table for -ap 5 (HADIS). '
                              'hybrid is HADIS as evaluated in the paper; the others '
                              'are routing ablations. Ignored by the baselines.'))

    return parser.parse_args()


def serve(args):
    port = args.port
    allocationPolicy = AllocationPolicy(int(args.allocationPolicy))
    set_cas_exec(args.cascadeExec)
    config.set_profile_driven(args.profileDriven)
    print(config.mode_banner('controller'))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    controller = Controller(allocationPolicy=allocationPolicy,
                            cascadeTable=args.cascadeTable)
    controller_pb2_grpc.add_ControllerServicer_to_server(controller, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logging.info(f'Controller started, listening on port {port}...')
    logging.info(f'Using resource allocation policy {allocationPolicy}')
    server.wait_for_termination()


if __name__=='__main__':
    logfile_name = f'../../logs/controller_{time.time()}.log'
    logging.basicConfig(filename=logfile_name, level=logging.INFO, 
                        format='%(asctime)s %(levelname)-8s %(message)s')
    serve(getargs())
