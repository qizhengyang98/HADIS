# HADIS: A Hybrid Architecture for Query-Aware Diffusion Model Serving

This repository is the artifact for the paper **"HADIS: A Hybrid Architecture for
Query-Aware Diffusion Model Serving"**, conditionally accepted at **EuroSys 2027**.

## 1. What the paper does

Query-aware model serving routes queries through cascades of increasing cost to
balance quality against throughput. Existing cascade systems force *every* query
through the lightweight stage, wasting work on hard queries, and fix the
light/heavy model pair regardless of how the workload shifts. In text-to-image
diffusion serving this waste is structurally severe: quality can only be assessed
*after* generation completes, and a rejected lightweight output cannot be reused
by the heavyweight model.

HADIS is a diffusion serving system built on a **hybrid cascade architecture**
that combines two complementary mechanisms:

- a **pre-generation router**, which sends predicted-hard queries straight to the
  heavy model, skipping the lightweight stage entirely;
- a **post-generation discriminator**, which catches false-easy cases that the
  router let through, and escalates them.

The hybrid design couples three decisions that must adapt together as load
changes: routing thresholds, model-pair selection, and GPU allocation. HADIS
solves them jointly with a MILP that re-plans periodically at runtime.

The artifact corresponds to the paper's end-to-end comparison against five baselines
and provides a functional test that runs the real models on real GPUs.

---

## 2. Repository structure

```
hadis_artifact/
├── hadis/                        the serving system
│   ├── src/
│   │   ├── config.py             ALL mode-dependent constants (see note below)
│   │   ├── controller/           resource allocation, the scheduler
│   │   ├── load_balancer/        routing table, dispatch to workers
│   │   ├── worker/               model execution and CPU<->GPU model swapping
│   │   ├── client/               trace replay, latency/SLO accounting
│   │   ├── tables/               offline cascade configuration tables
│   │   ├── common/               app graph, query objects, host helpers
│   │   ├── protos/               generated gRPC stubs (codegen.py regenerates)
│   │   └── codegen.py            regenerate protos/ from *.proto
│   ├── router/                   pre-generation router inputs
│   ├── discriminator/            post-generation discriminator
│   ├── traces/                   workload traces, prompts, SLO definitions
│   ├── profiles/                 measured hardware profile (see §4)
│   ├── gurobi/                   put your gurobi.lic here (§3.3)
│   ├── logs/                     run output (CSVs and per-component logs)
│   └── start_*.sh, stop_all.sh   launch scripts, one per component
├── experiments/                  sample test, hardware profiler, log collection
├── analysis/                     log parsing and figure generation
├── results/                      collected logs and figures
├── requirements.txt              pinned dependencies
└── Experiments.md                how to run the experiments
```

### Where each HADIS component lives

| Component | Path | Notes |
|---|---|---|
| **Controller** (scheduler) | `hadis/src/controller/controller.py` | Event loop, worker registry, re-planning. Selects the allocation policy via `-ap` |
| **HADIS MILP** | `hadis/src/controller/qaware_multi_cascade_ILP.py` | `solve_milp_loop()` jointly optimises router threshold, discriminator threshold, model pair and GPU counts |
| **Load balancer** | `hadis/src/load_balancer/load_balancer.py` | Holds the routing table, forwards queries to workers |
| **Worker (simulated)** | `hadis/src/worker/worker.py`, `model.py` | Used by E1; sleeps on profiled latencies instead of generating |
| **Worker (real models)** | `hadis/src/worker/worker_re.py`, `model_re.py` | Used by E2; runs the real checkpoints |
| **Router** | `isHardByRouter()` in `model.py` / `model_re.py`, features in `router/prompt_features.csv` | Percentile threshold over precomputed prompt features |
| **Discriminator** | `discriminator/CLIP_discriminator_head.pt`, `model.py`, `confidence_scores/` | A 2-layer head on a frozen CLIP ViT-B/32 encoder. Only the head is shipped (0.5 MB); the encoder is reconstructed by `clip.load('ViT-B/32')` at startup |
| **Client** | `hadis/src/client/client.py` | Replays the trace, records per-query latency |
| **Hardware profiler** | `experiments/profile_gpu.py` | Measures latency, swap cost and memory on *your* GPU (section 3.5) |
| **Sample test** | `experiments/sample_test.sh` | One-command check that E1 can run here (section 4) |
| **Analysis** | `analysis/parse_logs.py`, `plot_e1_end2end.py`, `plot_e2_functional.py` | Turns logs into the figures |

> **One note on `config.py`.** Every mode-dependent constant, e.g., SLO, trace, model latencies, batch caps, re-planning interval, lives there, because the two experiments run on different time scales (E1 compresses time 10×). Components print their mode in a startup banner; a mismatch between them is silent and produces meaningless results.

---

## 3. Installation

### 3.1 Environment

```bash
conda create -n hadis python=3.8
conda activate hadis
pip install -r requirements.txt
```

`requirements.txt` is pinned to the versions the reported results were produced
with. It installs one package straight from GitHub (OpenAI CLIP, which is not on
PyPI), so **`git` must be available and github.com reachable**. Otherwise the
whole install aborts on that line.

### 3.2 GPU and host requirements

The head node runs the controller, load balancer, sink and client. It needs
**8 CPU cores** and no GPU of its own.

| Experiment | Workers | Worker requirements |
|---|---|---|
| E1 (profile-driven) | 16 in total | **2 CPU cores per worker (32 in total)**, any CUDA GPU. No model is loaded |
| E2 (real models) | 4, one per node | **8 CPU cores each**, one GPU with at least 40 GB, and about 100 GB of host RAM |

**E1 does not constrain how the 16 workers are distributed.** Any number of nodes
will do as long as they provide 32 cores in total: 2 nodes of 16 cores, 4 of 8,
or a single 32-core node. Workers only need CUDA to be visible so they can report themselves as GPU workers; they never load a model, so several can share one card.

**E2 does constrain it: one worker per node.** Each worker holds all four model
variants in **page-locked** host memory (measured 79.5 GB resident, 89 GB peak
while loading), so the ~100 GB of RAM is per worker and not negotiable.
Page-locked pages cannot be swapped out, so a node that cannot hold them is
OOM-killed rather than slowed down. Each worker also needs its own GPU.


### 3.3 Gurobi

HADIS's resource allocation is a MILP solved with Gurobi, so **a Gurobi licence
is required** to run the controller. (Only the controller needs it; workers do
not.)

1. Follow the instructions on the [official website](https://www.gurobi.com/solutions/licensing/)
   to obtain a commercial or a free academic licence.
2. Gurobi provides a `gurobi.lic` file. Place it at:

```
hadis/gurobi/gurobi.lic
```

An academic named-user licence is free and sufficient. Note that Gurobi binds the
licence to the machine it was generated on, so it must be activated on (or moved
to) the node that runs the controller. `gurobipy.GurobiError: HostID mismatch`
means the controller is running somewhere else.

### 3.4 The CLIP encoder

The discriminator is a 2-layer head on a frozen CLIP ViT-B/32 encoder. Only the
head ships with the artifact (0.5 MB); the encoder is fetched once by
`clip.load('ViT-B/32')` on the first run that touches the discriminator, whether
that is `experiments/profile_gpu.py` or a worker starting up.
It lands in **`~/.cache/clip/ViT-B-32.pt`** (338 MB), and each worker node needs to access to it.

### 3.5 Model checkpoints (E2 only)

E2 needs four diffusion variants (~78 GiB) in **one** HuggingFace cache
directory.

**Authenticate first.** The three SD3.5 repositories are gated, so accept the
licence on each model page and then:

```bash
huggingface-cli login
```

**Then let the profiler fetch them.** `experiments/profile_gpu.py` checks the
cache offline and downloads anything missing into it, so a normal first run
populates the cache by itself:

```bash
python experiments/profile_gpu.py --cache-dir /path/to/hf/cache
```

Pass `--no-download` to forbid this, which turns a missing checkpoint into an
immediate error naming what is absent instead of an unexpected 78 GiB transfer.

---

## 4. Sample test: is the installation ready?

Before committing to a full experiment, run the sample test. It takes about
three minutes on **one node with a CUDA GPU and 16 CPU cores**, and it is a
single command:

```bash
conda activate hadis
experiments/sample_test.sh
```

**Run `experiments/remove_logs.sh` to clean all logs under `hadis/logs` before the sample test.** It checks everything E1 depends on and then proves it by running E1 in miniature. First the preconditions: every package from `requirements.txt`
imports, the Gurobi licence solves an actual LP, the repository is complete, and the machine has CUDA, tmux and enough cores. Then it launches the controller, load
balancer, sink, four workers and the client on a 39 s trace, and confirms the
MILP produced plans, the workers executed, and queries completed end to end.

The output is a checklist ending in PASS or FAIL:

```
-- 6. Results --------------------------------------------------
  [ ok ] client replayed the trace                       58s
  [ ok ] controller wrote slo_timeouts_per_second.csv    47 rows
  [ ok ] workers logged execution                        5 of 5 (workers + sink)
  [ ok ] controller solved the MILP                      9 plans
         queries: offered 281, completed 279, timeout 2, dropped 0
  [ ok ] queries completed end to end

 PASS: 18 checks passed in 58s. This machine can run E1.
```

It deliberately uses `-ap 5` (HADIS) so the MILP runs and the Gurobi licence is
exercised rather than assumed. On failure it leaves `hadis/logs/` intact so you
can read what went wrong.

**Scope.** The sample test covers E1 only. It loads no model, because E1 is
profile-driven: the workers sleep on profiled latencies and need CUDA merely to
be visible. E2 additionally needs the diffusion checkpoints and a hardware
profile, which `experiments/profile_gpu.py` downloads and measures (sections 3.4
and 3.5). 

---

## 5. Experiments

Two experiments, described step by step in **[Experiments.md](Experiments.md)**:

- **E1, profile-driven end-to-end** (paper Figure 7). 16 workers, simulated model
  execution driven by profiled latencies, a ~350 s replay peaking at 64 QPS. Run once
  per system for six systems: Clipper-Light, Clipper-Heavy, INFaaS-Acc, Proteus,
  DiffServe and HADIS, selected with `-ap 0` through `-ap 5`.
- **E2, functional test on real GPUs.** 4 workers, real checkpoints, real router and discriminator, a ~3500 s replay peaking at 1.6 QPS
  with a 60 s SLO. By default rerouting uses precomputed per-prompt scores;
  `--live-discriminator` switches it to the live output (Experiments.md section 7). This is not
  a figure reproduction, because four GPUs cannot carry the paper's load. It
  demonstrates that the system serves end-to-end on real models and traces. The experiments
  in the paper are conducted on a testbed cluster that consists of 16 workers, each
  with 128GB hosting memory, 8 CPU cores, and an NVIDIA L40S GPU.

---

## 6. Citation

```bibtex
@inproceedings{hadis2027,
  title     = {{HADIS}: A Hybrid Architecture for Query-Aware Diffusion Model Serving},
  author    = {Qizheng Yang and Tung-I Chen and Siyu Zhao and Ramesh K. Sitaraman and Hui Guan},
  booktitle = {Proceedings of the Twenty-Second European Conference on Computer Systems},
  year      = {2027},
  location  = {Rabat, Morocco}
}
```
