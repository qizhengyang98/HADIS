"""DiffServe baseline allocator (-ap 4).

Ported from the DiffServe testbed (``diffserve/src/controller/qaware_cascade_ILP.py``)
so that the baseline runs inside this codebase instead of a second checkout.
The formulation is unchanged; only the plumbing is:

* the model pair is a parameter (default (1, 3) = sd35turbo -> sd35large, the
  pairing the paper fixes DiffServe to) instead of the hard-coded
  ``('sdturbo', 'sdv15')`` names;
* latencies come from ``config.get_controller_runtimes()`` in **seconds**
  keyed ``(model, batch)``, rather than a CSV in milliseconds keyed
  ``(task, variant, batch)``;
* ``f(t)``, the fraction of queries the discriminator escalates at threshold
  ``t``, is the identity.  The original mapped an *absolute* discriminator
  score to an escalation fraction through an empirically measured CDF
  (``traces/apps/f_t_sdturbo_sdv15.csv``).  In this codebase ``conf_thres`` is a
  *percentile* of the score distribution -- ``model.py:executeBatch`` picks the
  threshold as ``sorted(scores)[int(len * conf_thres) - 1]`` -- so the escalated
  fraction is ``t`` by construction, and the CDF is the identity.

The ILP jointly picks the number of workers for each stage, a batch size for
each stage, and the discriminator threshold, maximising the escalation
threshold (its proxy for quality) subject to throughput and latency:

    max  t
    s.t. x_light + x_heavy <= S - 1                (one worker reserved for the sink)
         x_light * p(b_light) >= w * D
         x_heavy * p(b_heavy) >= w * D * f(t)
         L(b_light) + Q_light + L(b_heavy) + Q_heavy <= SLO
"""
import sys
sys.path.append('..')
import math
import time
from collections import defaultdict

import gurobipy as gp

import config


class DiffServeILPAllocator:
    def __init__(self, profiled_runtimes, model_pair=None, total_servers=1):
        """profiled_runtimes: {(model, batch): seconds}."""
        self.iteration = 0

        light_idx, heavy_idx = model_pair or config.DIFFSERVE_MODEL_PAIR
        model_order = config.get_model_order()
        self.light = model_order[light_idx]
        self.heavy = model_order[heavy_idx]
        self.light_level = light_idx
        self.heavy_level = heavy_idx

        # Workers per stage, batch size per stage, discriminator threshold
        self.x1 = self.x2 = None
        self.b1 = self.b2 = None
        self.thr = None
        self.measured_demand = 0
        self.max_f = -1

        self.profiled_throughputs = {}
        self.execution_latencies = {}
        self.profiled_runtimes = profiled_runtimes
        for (model, batch_size), runtime in profiled_runtimes.items():
            self.profiled_throughputs[(model, batch_size)] = batch_size / runtime
            self.execution_latencies[(model, batch_size)] = (100000.0 if runtime == math.inf
                                                             else runtime)

        self.total_servers = total_servers

        # Threshold grid, and f(t) = t (see module docstring).
        self.t_values = [round(0.005 * i, 4) for i in range(201)]
        self.f_t_values = list(self.t_values)

    def update_num_servers(self, num_servers):
        self.total_servers = num_servers

    def initialize(self):
        """Starting point before any demand has been observed."""
        self.b1 = 8
        self.b2 = 8
        self.x1 = math.ceil(self.total_servers / 2)
        # one worker reserved for the sink; clamp so a not-yet-populated cluster
        # does not produce a negative worker count
        self.x2 = max(self.total_servers - self.x1 - 1, 0)
        self.thr = 1
        self.max_f = -1
        return self.prepare_data_structures(self.x1, self.x2, self.b1, self.b2) + (self.thr,)

    def iterate(self, sysDemand, latencySLOInSec, demand_per_model, queue_length_per_model,
                demand_weight=1.0):
        if sysDemand == 0:
            return self.initialize()
        return self.solve_ilp(sysDemand, latencySLOInSec, demand_per_model,
                              queue_length_per_model, demand_weight=demand_weight)

    def solve_ilp(self, sysDemand, latencySLOInSec, demand_per_model, queue_length_per_model,
                  demand_weight=1.0):
        """Joint ILP over (workers, batch sizes, discriminator threshold).

        demand_weight < 1 under-provisions (trades SLO compliance for quality);
        demand_weight > 1 over-provisions as a buffer against demand spikes.
        """
        self.thr = None

        m = gp.Model('DiffServe query-aware cascade ILP')
        m.setParam('LogToConsole', 0)
        m.setParam('Threads', 12)

        total_servers = self.total_servers
        total_demand = sysDemand
        allowed_batch_sizes = config.get_allowed_batch_sizes()
        slo = latencySLOInSec

        x1 = m.addVar(vtype=gp.GRB.INTEGER, name='x1')
        x2 = m.addVar(vtype=gp.GRB.INTEGER, name='x2')
        b1 = m.addVars(allowed_batch_sizes, vtype=gp.GRB.BINARY, name='b1')
        b2 = m.addVars(allowed_batch_sizes, vtype=gp.GRB.BINARY, name='b2')
        thr_ind = m.addVars(range(len(self.f_t_values)), vtype=gp.GRB.BINARY,
                            name='threshold_indicator')
        threshold = m.addVar(vtype=gp.GRB.CONTINUOUS, name='threshold')

        m.addConstr(x1 >= 0)
        m.addConstr(x2 >= 0)

        # Exactly one batch size per stage, one threshold
        m.addConstr(sum(b1[i] for i in allowed_batch_sizes) <= 1)
        m.addConstr(sum(b2[j] for j in allowed_batch_sizes) <= 1)
        m.addConstr(sum(thr_ind[k] for k in range(len(self.f_t_values))) <= 1)

        # Reserve one worker for the sink
        m.addConstr(x1 + x2 <= total_servers - 1)

        # Real mode caps the batch size of the heavy models (see config)
        for i in allowed_batch_sizes:
            if i > config.get_max_batch_size(self.light):
                m.addConstr(b1[i] == 0)
            if i > config.get_max_batch_size(self.heavy):
                m.addConstr(b2[i] == 0)

        # x1 * p(b1) >= w * D
        m.addConstr(sum(b1[i] * self.profiled_throughputs[self.light, i]
                        for i in allowed_batch_sizes) * x1 >= demand_weight * total_demand)

        # x2 * p(b2) >= w * D * f(t)
        m.addConstr(sum(b2[j] * self.profiled_throughputs[self.heavy, j]
                        for j in allowed_batch_sizes) * x2 >=
                    demand_weight * total_demand * sum(thr_ind[k] * self.f_t_values[k]
                                                       for k in range(len(self.f_t_values))))

        m.addConstr(threshold == sum(thr_ind[k] * self.t_values[k]
                                     for k in range(len(self.f_t_values))))

        exec_latency_b1 = sum(b1[i] * self.execution_latencies[self.light, i]
                              for i in allowed_batch_sizes)
        exec_latency_b2 = sum(b2[j] * self.execution_latencies[self.heavy, j]
                              for j in allowed_batch_sizes)

        # Queuing delay from Little's law on the measured EWMAs
        queue_safety_factor = 0.0 if demand_weight < 1 else 1.2
        queuing_delay = defaultdict(float)
        for model in queue_length_per_model:
            queue_length = queue_length_per_model[model]
            arrival_rate = demand_per_model.get(model, 0)
            queuing_delay[model] = (0 if not arrival_rate
                                    else queue_safety_factor * queue_length / arrival_rate)

        m.addConstr(exec_latency_b1 + queuing_delay[self.light] +
                    exec_latency_b2 + queuing_delay[self.heavy] <= slo)

        m.setObjective(threshold, gp.GRB.MAXIMIZE)

        start_time = time.time()
        m.optimize()
        print(f'Time to solve DiffServe ILP: {time.time() - start_time:.4f} seconds')

        if m.status != gp.GRB.OPTIMAL:
            # Infeasible: shed load onto the lightweight stage with no escalation
            print(f'DiffServe ILP failed with status {m.status}; falling back to all-light')
            required, batch_sizes = self.prepare_data_structures(
                x1=total_servers - 1, x2=0, b1=16, b2=1)
            return required, batch_sizes, 0.0

        for i in allowed_batch_sizes:
            if b1[i].X > 0:
                self.b1 = i
            if b2[i].X > 0:
                self.b2 = i
        self.thr = threshold.X
        self.x1 = int(x1.X)
        self.x2 = int(x2.X)
        print(f'total demand: {total_demand}, x1: {self.x1}, b1: {self.b1}, '
              f'x2: {self.x2}, b2: {self.b2}, threshold: {self.thr}')

        required, batch_sizes = self.prepare_data_structures(self.x1, self.x2, self.b1, self.b2)
        return required, batch_sizes, self.thr

    def prepare_data_structures(self, x1, x2, b1, b2):
        """{model: workers}, {model: batch size} for the two active stages."""
        required_workers = {self.light: x1, self.heavy: x2}
        batch_sizes = {self.light: b1, self.heavy: b2}
        return required_workers, batch_sizes

    def get_stats(self):
        return (self.iteration, self.x1, self.x2, self.b1, self.b2, self.thr,
                self.max_f, self.measured_demand)
