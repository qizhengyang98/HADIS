#!/bin/bash
# Start N workers on THIS node, one tmux window each. Run after the sink.
#
# Usage:
#   ./start_worker.sh [--profile-driven] -cip CONTROLLER_IP --node N [-n N]
#
#   --node N      which node this is: 1, 2, 3, ... Sets the port band so you
#                 never compute a base port: node 1 gets 50051-50100, node 2
#                 gets 50101-50150, node 3 gets 50151-50200, and so on.
#   -n N          workers to launch on this node (default 1, max 50)
#   --base-port   explicit first port, instead of --node
#   --dry-run     print the ports and commands, then exit without launching
#   --session     tmux session name (default hadis-workers)
#
# /work is shared and worker logs are named by port, so each node needs its own
# band, and that is what --node gives you.
#
# E1 (profile-driven): 8 workers on each of two nodes,
#     node 1:  ./start_worker.sh --profile-driven -cip <ip> --node 1 -n 8   # 50051-50058
#     node 2:  ./start_worker.sh --profile-driven -cip <ip> --node 2 -n 8   # 50101-50108
#
# In real mode each worker is pinned to its own GPU via CUDA_VISIBLE_DEVICES.
#
# Watch:  tmux attach -t hadis-workers   (Ctrl-b w to pick a window, Ctrl-b d to detach)
# Stop:   ./stop_all.sh
set -e
source "$(dirname "$0")/_common.sh"
parse_mode_flag "$@"

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
SESSION="hadis-workers"
set -- "${REST_ARGS[@]}"
while [ $# -gt 0 ]; do
    case "$1" in
        -cip|--controller_ip) CIP="$2"; shift 2 ;;
        -n|--num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --node) NODE="$2"; shift 2 ;;
        --base-port) BASE_PORT="$2"; NODE=""; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
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
banner "workers"
echo "   controller: ${CIP}"
echo "   workers   : ${NUM_WORKERS} on ports ${BASE_PORT}-$((BASE_PORT + NUM_WORKERS - 1))"
echo "   session   : ${SESSION}"
echo

if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN, nothing launched:"
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        echo "  worker $((i + 1)): port $((BASE_PORT + i))  gpu ${GPU_ARR[$((i % NUM_GPUS))]}  window w$((BASE_PORT + i))"
    done
    exit 0
fi

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    port=$((BASE_PORT + i))
    if [ -z "$MODE_FLAG" ]; then
        # Real mode: one model per GPU, so give each worker its own device.
        gpu_env="CUDA_VISIBLE_DEVICES=${GPU_ARR[$((i % NUM_GPUS))]}"
    else
        # Simulated mode: no model is loaded, workers only need CUDA to report
        # onCUDA=1 to the controller, so they can share a device.
        gpu_env=""
    fi
    cmd="cd $ROOT/src/worker && $gpu_env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python $PY worker.py -cip $CIP -p $port $MODE_FLAG"

    if [ "$i" -eq 0 ]; then
        tmux new-session -d -s "$SESSION" -n "w$port"
    else
        tmux new-window -t "$SESSION" -n "w$port"
    fi
    tmux send-keys -t "$SESSION:w$port" "$cmd" C-m
    echo "  worker $((i + 1))/${NUM_WORKERS} -> port $port  (window w$port)"
    sleep 1
done

echo
echo "Waiting for workers to come up (logs/model_<port>.log)..."
WAIT_TIMEOUT="${WAIT_TIMEOUT:-600}"
waited=0
while true; do
    missing=0
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        [ -f "$ROOT/logs/model_$((BASE_PORT + i)).log" ] || missing=$((missing + 1))
    done
    [ "$missing" -eq 0 ] && { echo "All ${NUM_WORKERS} workers are up (${waited}s)."; break; }
    [ "$waited" -ge "$WAIT_TIMEOUT" ] && { echo "WARNING: ${missing} worker(s) still down after ${WAIT_TIMEOUT}s."; break; }
    sleep 2
    waited=$((waited + 2))
done

echo "Attach with: tmux attach -t $SESSION"
