#!/bin/bash
# Shared setup for the HADIS launch scripts. Sourced, not executed.
#
# Every component takes --profile-driven; all of them must agree. Each script
# prints a banner saying which mode it started in -- check that they match.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Interpreter: the active conda environment (see README section 3.1), or set
# PY=/path/to/python to override.
PY="${PY:-$(command -v python3 || command -v python)}"
if [ -z "$PY" ]; then
    echo "ERROR: no python found. Activate the environment (conda activate hadis)" >&2
    echo "       or run with PY=/path/to/python" >&2
    exit 1
fi

# Gurobi (controller only). The academic licence is node-locked; see INSTALL.md.
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$ROOT/gurobi/gurobi.lic}"

# diffusers loads SD3.5's T5 tokenizer through protobuf, which crashes on the
# C++ implementation shipped in this environment.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

MODE_FLAG=""          # set by parse_mode_flag
MODE_NAME="real"
USE_TMUX=1            # head-node components run in tmux unless --foreground
HEAD_SESSION="${HEAD_SESSION:-hadis-head}"

# Consume the flags every script shares (--profile-driven, --foreground);
# leaves the rest in REST_ARGS.
parse_mode_flag() {
    REST_ARGS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --profile-driven) MODE_FLAG="--profile-driven"; MODE_NAME="profile-driven"; shift ;;
            --foreground|--no-tmux) USE_TMUX=0; shift ;;
            *) REST_ARGS+=("$1"); shift ;;
        esac
    done
}

# --------------------------------------------------------------------------- #
# Component launching and readiness
# --------------------------------------------------------------------------- #

port_open() { (exec 3<>"/dev/tcp/$1/$2") >/dev/null 2>&1; }

# wait_for_port <host> <port> [timeout] [label] -- returns 1 on timeout.
wait_for_port() {
    local host=$1 port=$2 timeout=${3:-60} label=${4:-"$1:$2"} waited=0
    while ! port_open "$host" "$port"; do
        sleep 1
        waited=$((waited + 1))
        if [ "$waited" -ge "$timeout" ]; then
            echo "  WARNING: ${label} not reachable at ${host}:${port} after ${timeout}s" >&2
            return 1
        fi
    done
    READY_SECS=$waited
    return 0
}

# Launch a head-node component. Default: a window of the shared tmux session,
# and the script returns so the same terminal can start the next component.
# With --foreground the component replaces this shell instead.
#
# The window is driven with send-keys into a shell rather than being given the
# command directly, so that a component which dies immediately leaves its error
# on screen instead of closing the window.
run_component() {
    local window="$1" cmd="$2"

    if [ "$USE_TMUX" != "1" ]; then
        exec bash -c "$cmd"
    fi

    if ! command -v tmux >/dev/null; then
        echo "ERROR: tmux not found; rerun with --foreground" >&2
        exit 1
    fi

    if tmux has-session -t "$HEAD_SESSION" 2>/dev/null; then
        if tmux list-windows -t "$HEAD_SESSION" -F '#W' 2>/dev/null | grep -qx "$window"; then
            echo "ERROR: window '${window}' already exists in tmux session '${HEAD_SESSION}'." >&2
            echo "       Run ./stop_all.sh first, or attach: tmux attach -t ${HEAD_SESSION}" >&2
            exit 1
        fi
        tmux new-window -t "$HEAD_SESSION" -n "$window"
    else
        tmux new-session -d -s "$HEAD_SESSION" -n "$window"
    fi
    tmux send-keys -t "${HEAD_SESSION}:${window}" "$cmd" C-m
    echo "  running in tmux ${HEAD_SESSION}:${window}"
}

# True if a "<python> <script> ..." process is running for this user. Matches on
# argv[0]/argv[1] rather than the command line text, so a shell that merely
# mentions the script name is not mistaken for the component itself.
python_proc_running() {
    local script=$1 pid argv0 argv1
    for pid in $(ps -u "$USER" -o pid= 2>/dev/null); do
        [ -r "/proc/$pid/cmdline" ] || continue
        argv0=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | sed -n 1p)
        argv1=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | sed -n 2p)
        case "${argv0##*/}" in python*) ;; *) continue ;; esac
        [ "${argv1##*/}" = "$script" ] && return 0
    done
    return 1
}

# Printed once a component is accepting connections: the go-ahead to start the
# next one in the same terminal.
component_launched() {   # label window port
    echo
    echo "  >>> ${1} LAUNCHED and ready  (tmux ${HEAD_SESSION}:${2}, port ${3}, ${READY_SECS:-0}s)"
}

attach_hint() {
    if [ "$USE_TMUX" = "1" ]; then
        echo "  attach: tmux attach -t ${HEAD_SESSION}   (Ctrl-b w switch, Ctrl-b d detach)"
    fi
    return 0
}

banner() {
    echo "=============================================================="
    echo " HADIS ${1}"
    echo "   mode   : ${MODE_NAME}$([ -z "$MODE_FLAG" ] && echo '  (real models, 60 s SLO, ~1 h trace)' || echo '  (simulated, 10x scaled, 6 s SLO, ~6 min trace)')"
    echo "   host   : $(hostname)  ip: $(hostname -I 2>/dev/null | awk '{print $1}')"
    echo "   python : ${PY}"
    echo "   logs   : ${ROOT}/logs"
    echo "=============================================================="
    echo "  ALL components must run in the same mode."
}

require_py() {
    if [ ! -x "$PY" ]; then
        echo "ERROR: python not found at $PY (set PY=/path/to/python)" >&2
        exit 1
    fi
    # $PY comes from PATH, so it is the environment's interpreter only while
    # that environment is active. Check for a dependency rather than let the
    # component fail later with a bare ImportError in a tmux pane.
    if ! "$PY" -c "import grpc" >/dev/null 2>&1; then
        echo "ERROR: $PY cannot import grpc, so it is not the artifact environment." >&2
        echo "       Activate it first:  conda activate hadis" >&2
        echo "       (or point at it directly:  PY=/path/to/env/bin/python $0 ...)" >&2
        exit 1
    fi
    mkdir -p "$ROOT/logs"
}
