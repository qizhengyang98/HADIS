#!/bin/bash
# Start N real-execution workers on THIS node, one tmux window each (E2).
#
# Usage:
#   ./start_worker_re.sh -cip CONTROLLER_IP --node N [-n N] [--cache DIR]
#
#   --node N      which node this is: 1, 2, 3, ... Sets the port band so you
#                 never compute a base port: node 1 gets 50051-50100, node 2
#                 gets 50101-50150, node 3 gets 50151-50200, and so on.
#   -n N          workers to launch on this node (default 1, max 50)
#   --base-port   explicit first port, instead of --node
#   --cache DIR   the HuggingFace cache holding ALL FOUR checkpoints: the
#                 directory containing the models--<org>--<name> folders.
#                 Defaults to $HADIS_MODEL_CACHE, else HuggingFace's own cache.
#   --live-discriminator
#                 escalate on the live discriminator output instead of the
#                 score precomputed for each prompt (the default)
#   --dry-run     print the ports and commands, then exit without launching
#   --session     tmux session name (default hadis-workers-re)
#
# The band matters: /work is shared and worker logs, plus
# logs/model_swaps_<port>.csv, are named by port. Two nodes on the same port
# overwrite each other's evidence and each node's readiness check sees the
# other's log.
#
# This is the real-model counterpart of start_worker.sh and runs worker_re.py.
# There is no --profile-driven flag: a simulated run belongs in start_worker.sh.
#
# E2: four nodes, one worker each --
#     ./start_worker_re.sh -cip <ip> --node 1 -n 1     # then --node 2, 3, 4
#
# Each worker holds all four variants in page-locked host memory (measured
# 79.5 GB resident, 89 GB peak while loading) and keeps one on its GPU, so run
# ONE worker per node unless the node has >=100 GB of RAM per worker.
#
# Watch:  tmux attach -t hadis-workers-re   (Ctrl-b w to pick, Ctrl-b d to detach)
# Stop:   ./stop_all.sh
set -e
source "$(dirname "$0")/_common.sh"

CIP=""

# Worker ports are banded by node so nobody has to compute them: node 1 gets
# 50051-50100, node 2 gets 50101-50150, node 3 gets 50151-50200, and so on.
# Pass --node N and the base port follows; -n must stay <= 50 so the bands
# cannot overlap. --base-port still overrides if you want an explicit port.
FIRST_PORT=50051
PORTS_PER_NODE=50
NUM_WORKERS=1
BASE_PORT=$FIRST_PORT
NODE=""
DRY_RUN=0
LIVE_FLAG=""
SESSION="hadis-workers-re"
CACHE="${HADIS_MODEL_CACHE:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        -cip|--controller_ip) CIP="$2"; shift 2 ;;
        -n|--num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --node) NODE="$2"; shift 2 ;;
        --base-port) BASE_PORT="$2"; NODE=""; shift 2 ;;
        --cache|--cache-dir) CACHE="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
        --live-discriminator) LIVE_FLAG="--live-discriminator"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -n "$NODE" ]; then
    if ! [ "$NODE" -ge 1 ] 2>/dev/null; then
        echo "ERROR: --node must be 1, 2, 3, ... (got '$NODE')" >&2
        exit 1
    fi
    BASE_PORT=$((FIRST_PORT + (NODE - 1) * PORTS_PER_NODE))
fi
if [ "$NUM_WORKERS" -gt "$PORTS_PER_NODE" ]; then
    echo "ERROR: -n $NUM_WORKERS exceeds the ${PORTS_PER_NODE}-port band per node;" >&2
    echo "       a node may run at most ${PORTS_PER_NODE} workers." >&2
    exit 1
fi

if [ -z "$CIP" ]; then
    echo "ERROR: -cip <controller ip> is required (printed by start_controller.sh)" >&2
    exit 1
fi
if ! command -v tmux >/dev/null; then
    echo "ERROR: tmux not found" >&2
    exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists. Run ./stop_all.sh first." >&2
    exit 1
fi
if [ ! -f "$ROOT/discriminator/CLIP_discriminator_head.pt" ] \
   && [ ! -f "$ROOT/discriminator/CLIP_discriminator.pt" ]; then
    echo "ERROR: no discriminator weights in $ROOT/discriminator." >&2
    exit 1
fi
if [ ! -d "$ROOT/profiles" ] || [ -z "$(ls -A "$ROOT/profiles" 2>/dev/null)" ]; then
    echo "WARNING: no profile in $ROOT/profiles, so the planner will fall back to the"
    echo "         reference latencies. Run experiments/profile_gpu.py on this GPU first."
fi

# Validate the cache this worker will actually read, whether it came from
# --cache, $HADIS_MODEL_CACHE, or HuggingFace's default location. Checking only
# the explicit case would let the commonest mistake, forgetting to export
# HADIS_MODEL_CACHE on a worker node, fail minutes into a 78 GiB load instead
# of immediately.
if [ -n "$CACHE" ]; then
    CHECK_DIR="$CACHE"
    CHECK_WHAT="--cache '$CACHE'"
else
    CHECK_DIR="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:+$HF_HOME/hub}}}"
    CHECK_DIR="${CHECK_DIR:-$HOME/.cache/huggingface/hub}"
    CHECK_WHAT="the default HuggingFace cache '$CHECK_DIR' (no --cache given, \$HADIS_MODEL_CACHE unset)"
fi

if [ ! -d "$CHECK_DIR" ]; then
    echo "ERROR: $CHECK_WHAT is not a directory" >&2
    exit 1
fi
missing=""
for repo in models--ByteDance--SDXL-Lightning \
            models--stabilityai--stable-diffusion-xl-base-1.0 \
            models--stabilityai--stable-diffusion-3.5-large-turbo \
            models--stabilityai--stable-diffusion-3.5-medium \
            models--stabilityai--stable-diffusion-3.5-large; do
    [ -d "$CHECK_DIR/$repo" ] || missing="$missing    $repo\n"
done
if [ -n "$missing" ]; then
    echo "ERROR: $CHECK_WHAT is missing:" >&2
    printf "%b" "$missing" >&2
    echo "  All four variants must be in ONE cache directory. Either point at it:" >&2
    echo "    ./start_worker_re.sh ... --cache /path/to/hf/cache" >&2
    echo "    (or export HADIS_MODEL_CACHE=/path/to/hf/cache on THIS node)" >&2
    echo "  or populate it with:" >&2
    echo "    python experiments/profile_gpu.py --cache-dir /path/to/hf/cache" >&2
    exit 1
fi

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NUM_GPUS" -eq 0 ] && NUM_GPUS=1

# Which physical GPU(s) this job owns.
#
# These workers run inside tmux, whose server lives in a systemd user scope --
# OUTSIDE the SLURM job's cgroup. Inside the cgroup SLURM remaps the allocated
# card to index 0; outside it, index 0 is the machine's first physical GPU,
# which may belong to somebody else's job. Hardcoding 0 sent workers onto a
# neighbour's card and they died with CUDA OOM against 36 GiB they did not own.
# SLURM_JOB_GPUS carries the physical indices, so prefer it.
GPU_LIST="${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}"
if [ -z "$GPU_LIST" ]; then
    GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))
fi
IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"
NUM_GPUS=${#GPU_ARR[@]}
echo "   gpus      : ${NUM_GPUS} usable [${GPU_LIST}]${SLURM_JOB_GPUS:+  (from SLURM_JOB_GPUS)}"

require_py
MODE_NAME="real"
banner "workers (real execution)"
echo "   controller: ${CIP}"
echo "   workers   : ${NUM_WORKERS} on ports ${BASE_PORT}-$((BASE_PORT + NUM_WORKERS - 1))"
echo "   cache     : ${CACHE:-<HuggingFace default>}"
echo "   escalation: $([ -n "$LIVE_FLAG" ] && echo "live discriminator output" || echo "precomputed per-prompt scores (default)")"
echo "   session   : ${SESSION}"
echo
echo "   First start loads ~78 GiB of weights per worker; allow several minutes"
echo "   before the worker registers."
echo

if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN, nothing launched:"
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        echo "  worker $((i + 1)): port $((BASE_PORT + i))  gpu ${GPU_ARR[$((i % NUM_GPUS))]}  window w$((BASE_PORT + i))"
    done
    exit 0
fi

START_TS=$(date +%s)

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    port=$((BASE_PORT + i))
    # One model per GPU: every worker gets its own device.
    env_vars="CUDA_VISIBLE_DEVICES=${GPU_ARR[$((i % NUM_GPUS))]}"
    [ -n "$CACHE" ] && env_vars="$env_vars HADIS_MODEL_CACHE='$CACHE'"
    # Tee to a file: the model runs in a spawned child whose stderr goes to this
    # pane, and a pane dies with its tmux session, so a traceback or a "Killed"
    # would be lost exactly when it is most needed.
    console="$ROOT/logs/console_w${port}.txt"
    cmd="cd $ROOT/src/worker && $env_vars PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python $PY worker_re.py -cip $CIP -p $port $LIVE_FLAG 2>&1 | tee '$console'"

    if [ "$i" -eq 0 ]; then
        tmux new-session -d -s "$SESSION" -n "w$port"
    else
        tmux new-window -t "$SESSION" -n "w$port"
    fi
    tmux send-keys -t "$SESSION:w$port" "$cmd" C-m
    echo "  worker $((i + 1))/${NUM_WORKERS} -> port $port  (window w$port, console logs/console_w${port}.txt)"
    sleep 1
done

echo
echo "Waiting for workers to load their models (logs/model_<port>.log)..."
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
waited=0
while true; do
    ready=0
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        log="$ROOT/logs/model_$((BASE_PORT + i)).log"
        # mtime guard: a log left over from an earlier run would match at once
        [ -f "$log" ] && [ "$(stat -c %Y "$log" 2>/dev/null || echo 0)" -ge "$START_TS" ] \
            && grep -q "variants ready" "$log" 2>/dev/null && ready=$((ready + 1))
    done
    [ "$ready" -eq "$NUM_WORKERS" ] && { echo "All ${NUM_WORKERS} workers have their models (${waited}s)."; break; }
    [ "$waited" -ge "$WAIT_TIMEOUT" ] && { echo "WARNING: $((NUM_WORKERS - ready)) worker(s) not ready after ${WAIT_TIMEOUT}s."; break; }
    sleep 5
    waited=$((waited + 5))
done

echo "Attach with: tmux attach -t $SESSION"
