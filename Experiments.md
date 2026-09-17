# HADIS Artifact: Running the Experiments

This document is the step-by-step guide to reproducing the two experiments. For
what the artifact is, how the repository is laid out and how to install its
dependencies, see [README.md](README.md).

---

## 1. What you can run

**E1, profile-driven end-to-end (paper Figure 7).** 16 workers, simulated model
execution on profiled latencies, a ~350 s replay of the Microsoft Azure Functions
trace peaking at 64 QPS with a 6 s SLO. Run once per system, six methods in total.

| `-ap` | System | Log name to collect under |
|---|---|---|
| 0 | Clipper-Light | `clipper_light` |
| 1 | Clipper-Heavy | `clipper_heavy` |
| 2 | INFaaS-Acc | `infaas` |
| 3 | Proteus | `proteus` |
| 4 | DiffServe | `diffserve` |
| 5 | HADIS | `hadis` |

**E2, functional test on real GPUs.** 4 workers, real checkpoints, real router and real discriminator, a ~3500 s replay peaking at 1.6 QPS
with a 60 s SLO. This is not a reproduction of a paper figure: four GPUs cannot carry the paper's load. It
demonstrates that the system serves end-to-end on real models and that the cascade
routes between them. See section 7.

Sections 2–6 below are E1 and use `--profile-driven`. **Every component must be given the same mode flag**. A mismatch pits a 6 s SLO against un-scaled 27 s latencies without any error being raised.

---

## 2. Cluster layout

The walkthrough below uses three nodes, which is one convenient split rather than
a requirement. E1 needs 16 workers with 2 cores each, so any arrangement giving
32 worker cores in total works: two nodes of 16 cores as shown here, four of 8,
or one 32-core node. The head node needs 8 cores. Only E2 fixes the
layout, at one worker per node.

```
  head node                          worker node A          worker node B
  ---------                          -------------          -------------
  controller      (port 50050)       8 workers              8 workers
  load balancer   (port 50049)       --node 1               --node 2
                                     ports 50051-50058      ports 50101-50108
  sink worker     (port 50048)
  client          (port 60050)
```

All three must be able to reach each other on the head-node ports 50048 (sink),
50049 (load balancer), 50050 (controller) and 60050 (client), plus each worker
node's band (50051-50100 for node 1, 50101-50150 for node 2, ...).

---

## 3. Running one method

**Activate the environment first, on every node you launch anything from:**

```bash
conda activate hadis
```

The launch scripts use whichever `python` is on `PATH`, so they pick up the
environment only while it is active. They check this before starting and refuse
with a clear message otherwise. 

Repeat this whole section once per `-ap` value.

### Step 0: Run the sample test first (Skip if this has been verified)

It takes three minutes on a single node and confirms that the environment, the Gurobi licence, the repository's data files and all five components work together, so a failure here is much cheaper to diagnose than one eight minutes into a real run:

```bash
experiments/sample_test.sh
```

See README.md section 4 for what it checks.

### Step 1: head node

One terminal is enough: each script launches its component into a window of the
tmux session `hadis-head` and returns once that component is accepting
connections, so you can type the next command straight away.

```bash
cd <artifact>/hadis

./start_controller.sh --profile-driven -ap 5
#   system : HADIS  (-ap 5)
#   ---> workers should use:  -cip <HEAD_IP>      <- copy this IP
#   >>> CONTROLLER LAUNCHED and ready  (tmux hadis-head:contr, port 50050, 2s)

./start_load_balancer.sh --profile-driven -cip <HEAD_IP>
#   >>> LOAD BALANCER LAUNCHED and ready  (tmux hadis-head:loadb, port 50049, 1s)

./start_worker_sink.sh --profile-driven -cip <HEAD_IP>
#   >>> SINK LAUNCHED and ready  (tmux hadis-head:sink, port 50048, 9s)
```

Order matters: the load balancer must be registered with the controller before any
worker starts, because workers learn its address from the controller when they
register. 

Once a component (e.g., controller, worker, load balancer, etc) is launched, the logs will be under `hadis/logs`.


### Step 2: worker nodes

On **worker node A** (`-n` means the number of workers launched on this node):

```bash
cd <artifact>/hadis
./start_worker.sh --profile-driven -cip <HEAD_IP> --node 1 -n 8
#   ...
#   All 8 workers are up (14s).
```

On **worker node B**:

```bash
cd <artifact>/hadis
./start_worker.sh --profile-driven -cip <HEAD_IP> --node 2 -n 8
```

**Give every node a different `--node`.** Each creates a tmux session `hadis-workers` with one window per worker, and blocks
until all of its workers are up.

### Step 3: client (head node)

Only once **ALL** worker nodes have reported `All 8 workers are up`:

```bash
./start_client.sh --profile-driven -lbip <HEAD_IP>
#   >>> CLIENT LAUNCHED  (tmux hadis-head:client)
#
#   Replaying the trace (timeout 1200s). This terminal will return when it ends.
#     ... 30s elapsed, 21 seconds of results recorded
#     ... 60s elapsed, 51 seconds of results recorded
#     ...
#   >>> TRACE ENDED after 365s  (358 seconds of results in logs/)
```

Note `-lbip` (load balancer), not `-cip`. 

This command **blocks until the trace has been fully replayed**, printing progress
every 30 s, then returns. The client sleeps 10 s, waits for a routing table, then replays `trace_64qps.txt` (chosen automatically in profile-driven mode); the whole run takes about 6 minutes.

If the client exits without finishing, or the trace does not end within the timeout (1200 s profile-driven, 5400 s real; override with `TRACE_TIMEOUT=`), the
script says so and exits non-zero instead of leaving you waiting. 

### Step 4: stop and collect

Run `./stop_all.sh` on **all nodes** (it kills the `hadis-head` /
`hadis-workers` sessions and every component process, including the spawned model
executors). Then, on the head node:

```bash
cd <artifact>
experiments/collect_logs.sh hadis          # use the name from the table in section 1
```

`collect_logs.sh` moves the run's logs into `results/logs/<method>/` and
leaves `hadis/logs/` empty for the next run. **Run it after every method.** The
controller appends to its CSVs, so skipping it merges two runs into one file.

---

## 4. Plotting

```bash
python analysis/plot_e1_end2end.py
```

Reads `results/logs/<method>/` and writes `results/figures/fig7_end2end.png` and
`results/figures/summary.csv`, which reports the average FID and SLO violation
ratio measured for each system.

**You can plot after any run.** The figure is drawn from the collected logs. Methods you have not run yet are left blank. 

---

## 5. Quick reference

```bash
# head node, per method
./start_controller.sh    --profile-driven -ap <0..5>
./start_load_balancer.sh --profile-driven -cip <HEAD_IP>
./start_worker_sink.sh   --profile-driven -cip <HEAD_IP>
#   ... start workers on both worker nodes ...
./start_client.sh        --profile-driven -lbip <HEAD_IP>   # blocks until TRACE ENDED
./stop_all.sh                                     # on every node
experiments/collect_logs.sh <method>                # on the head node

# worker nodes, per method
./start_worker.sh --profile-driven -cip <HEAD_IP> --node 1 -n 8   # node A
./start_worker.sh --profile-driven -cip <HEAD_IP> --node 2 -n 8   # node B
```

---

## 6. Troubleshooting

| Symptom | Cause |
|---|---|
| Workers never appear in the controller log | Load balancer started after the workers, or registered over loopback; restart with the head node's real IP |
| Remote workers cannot reach the load balancer or sink | `-cip localhost` was used on the head node |
| One worker node's readiness check returns instantly | Both nodes used the same `--node` |
| A run's CSVs have far more rows than expected | `collect_logs.sh` was not run after the previous system |
| `ERROR: the client exited without reporting 'Trace ended'` | The client crashed; attach to `hadis-head:client` for the traceback |
| `WARNING: no 'Trace ended' after <N>s` | The trace is slower than the timeout, or the client is stuck waiting for a routing table (no workers registered). Raise `TRACE_TIMEOUT=` or check the workers |
| Run ends early with short CSVs; controller terminal shows `ValueError: sleep length must be non-negative` | The controller's 1 s event loop overran, which kills its thread while gRPC keeps serving. Reduce the workers per node or give the head node more cores |
| `gurobipy.GurobiError: HostID mismatch` | The controller is not on the node the Gurobi licence is bound to. Only the controller needs the licence |

---

## 7. E2: functional test on real GPUs

Four nodes with one GPU each, plus the head node. 

As in section 3, run
`conda activate hadis` on every node before launching anything.

### Step 0: profile this GPU (once)

The below script will download diffusion models that it does not find in the given folder. That is ~78 GiB and the SD3.5 repositories are gated, so accept the licence on each model page and `huggingface-cli login` first.

On one of the worker nodes (`--cache-dir` is the folder to save the downloaded diffusion models, needs >100GB space.):

```bash
cd hadis_artifact
python experiments/profile_gpu.py --cache-dir /path/to/hf/cache
```

It loads every diffusion model variant, page-locks it, measures the swapping time and batch latency, and writes `hadis/profiles/<gpu>.json`. The controller and workers read it automatically when exactly one file is present. Takes about an hour.

The process settles at ~80 GB of memory, peaking near 90 GB while loading, so **the node needs ~100 GB of RAM**. Page-locked pages cannot be swapped out: a node that cannot hold them is OOM-killed, not slowed down.

If the profile is missing the planner falls back to the reference latencies from
the paper's hardware, which is fine for a smoke test and wrong for measurement.

### Step 1: head node

Same as E1 but without the mode flag:

```bash
./start_controller.sh -ap 5
./start_load_balancer.sh -cip <HEAD_IP>
./start_worker_sink.sh -cip <HEAD_IP>
```

### Step 2: worker nodes (one worker per node)

One per node, each with its own `--node` number. `--cache` is the folder where diffusion models are saved:

```bash
./start_worker_re.sh -cip <HEAD_IP> --node 1 -n 1 --cache /path/to/hf/cache  # node A
./start_worker_re.sh -cip <HEAD_IP> --node 2 -n 1 --cache /path/to/hf/cache  # node B
./start_worker_re.sh -cip <HEAD_IP> --node 3 -n 1 --cache /path/to/hf/cache  # node C
./start_worker_re.sh -cip <HEAD_IP> --node 4 -n 1 --cache /path/to/hf/cache  # node D
```

**Give every node a different `--node`.** `start_worker_re.sh` runs `worker_re.py`, the real-execution worker. Run **one worker per node**: each holds all four variants in page-locked host memory.

**How rerouting decision is made.** The discriminator runs on every generated image, but
by default the rerouting decision uses the corresponding confidence score precomputed offline for each query using the same discriminator.
This keeps results consistent and planning stable on a four-GPU testbed. To decide
on the live discriminator output instead, add `--live-discriminator` to every
`start_worker_re.sh` command. 

That is ~78 GiB and the SD3.5 repositories are gated, so accept the licence on each model page and `huggingface-cli login` first.


### Step 3: client

```bash
./start_client.sh -lbip <HEAD_IP>
```

In real mode the client defaults to the ~3500 s trace and a 5400 s timeout. The terminal returns when the trace ends, after about an hour.

### Step 4: plot

```bash
./stop_all.sh
python analysis/plot_e2_functional.py --log-dir hadis/logs
```

Four panels: offered vs completed throughput, SLO violation over time, queries per
second by model, and every model swap with its cost. **Run `experiments/remove_logs.sh` to clean all logs under `hadis/logs` before the next run(either for E1 or E2).**

---

## 8. E1 with one script: `run_E1.sh`

`run_E1.sh` runs the whole E1 experiment and generates its results with a single command. It sets up every component on every node, runs each chosen system, collects the logs and plots Figure 7 automatically. It performs the steps of sections 3 and 4 for you, so there is no need to start, stop or collect anything by hand.

### How it works

The script runs in two roles:

- **Head node**: drives the experiment. It starts the controller, load balancer, sink and client, tells the worker nodes when to start and stop their workers, collects the logs and plots.
- **Worker nodes**: each runs an agent that connects to the head node and starts, checks and stops the workers on that node when the head node asks.

For each chosen system, the head node:

1. starts the controller, load balancer and sink, and waits until each one has launched and registered with the controller;
2. starts the workers on every node through the agents, and waits until the controller has registered every worker and placed a model on each, and every worker has loaded its model and received a routing table;
3. starts the client, which replays the trace (about 6 minutes);
4. runs `stop_all.sh` on every node and collects the logs into `results/logs/<system>/`.

After the last system it runs `analysis/plot_e1_end2end.py` and writes `results/figures/fig7_end2end.png`. Each system takes about 8 minutes, so all six take about 48 minutes.

### Step 0: Before you start

- `conda activate hadis` on the head node and on every worker node.
- `hadis/logs/` must be empty. File an earlier run with `experiments/collect_logs.sh <name>` or discard it with `experiments/remove_logs.sh`.


### Step 1: start an agent on each worker node

On **each worker node**, run one agent. Give every node a different `--node` (1, 2, 3, ...) and set `-n` to the number of workers for that node, about one worker per 2 vCPUs. The workers across all nodes should add up to 16. For example, with one 16-vCPU node and two 8-vCPU nodes:

```bash
cd <artifact>
./run_E1.sh --agent --head-ip <HEAD_IP> --node 1 -n 8    # worker node A, 16 vCPUs
./run_E1.sh --agent --head-ip <HEAD_IP> --node 2 -n 4    # worker node B, 8 vCPUs
./run_E1.sh --agent --head-ip <HEAD_IP> --node 3 -n 4    # worker node C, 8 vCPUs
#   ...
#   [16:51:10] Waiting for the head node ...
```

`<HEAD_IP>` is the head node's address as the worker nodes see it (`hostname -I` on the head node). Leave the agents running: they wait for the head node, and can be started before or after it.

### Step 2: dry run on the head node

`--nodes` is the number of worker nodes, i.e. the number of agents to wait for:

```bash
cd <artifact>
./run_E1.sh --nodes 3 --dry-run
```

A dry run launches nothing. It:

- checks the head node: the Python environment, tmux, the Gurobi licence and that `hadis/logs/` is empty;
- waits for every agent to connect, which confirms that each worker node can reach the head node;
- checks each worker node through its agent: its worker count and port range, tmux, that CUDA is visible, and whether its `hadis/logs/` is shared with the head node or holds old worker logs;
- prints every command it would run, in order, on the head node and on each worker node, for every chosen system.

```
#   [ ok ] node 1 (nodeA): agent responded        8 workers on 16 vCPUs, ports 50051-50058
#   [ ok ] node 1 (nodeA): tmux available
#   [ ok ] node 1 (nodeA): CUDA visible
#   ...
#   total workers: 16
#   ...
#   All checks passed. The agents keep running: rerun this command without
#   --dry-run to start (about 48 minutes).
```

Fix anything marked `[FAIL]` and dry-run again. The agents keep running after a dry run, so there is no need to restart them.

### Step 3: run E1 on the head node

Run the same command without `--dry-run`:

```bash
./run_E1.sh --nodes 3                  # all six systems
```

**Choosing systems.** `--systems` takes a comma separated list of the `-ap` numbers from section 1 and runs only those, in the given order. Without it all six run.

```bash
./run_E1.sh --nodes 3 --systems 5      # HADIS only, about 8 minutes: a quick end-to-end test
./run_E1.sh --nodes 3 --systems 4,5    # DiffServe, then HADIS
```

Running one system first (`--systems 5`) is a good way to check the whole setup before the full run.

The head node prints each step as it completes, and each agent prints the tasks it runs. The head node's output is also saved to `results/logs/run_E1_<time>.log`. At the end it prints a summary:

```
#   E1 summary  (48 min)
#     Clipper-Light  ok  ... seconds of results in results/logs/clipper_light/
#     ...
#     HADIS          ok  ... seconds of results in results/logs/hadis/
#     figure         results/figures/fig7_end2end.png
```

When a run (not a dry run) finishes, the agents exit. Start them again (Step 1) before the next run.

### If something goes wrong

- **A system fails** at any step: it is stopped on every node, its logs are filed under `results/logs/failed_<system>_<time>/`, and the run moves on to the next system.
- **Ctrl-C on the head node** stops the workers on every node. The interrupted run's logs stay in `hadis/logs/`; clear them with `experiments/remove_logs.sh` before starting again.
- **An agent stops responding** for 30 s: the current system fails with a message naming the node.
- **Workers take longer than 300 s to settle** on a slow node: raise the limit with `READY_TIMEOUT=600 ./run_E1.sh --nodes 3`.
- **Port 50047 is in use** on the head node: pass another port with `--agent-port <port>`, to the head node and to every agent.
- **A result directory already exists**: `collect_logs.sh` moves the old `results/logs/<system>/` aside to `<system>_<time>` rather than overwriting it.
