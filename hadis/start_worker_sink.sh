#!/bin/bash
# Start the sink worker. Run this THIRD, after the load balancer.
#
# The sink is the terminal stage of the cascade: it accepts finished queries and
# reports their completion timestamps to the controller. It runs on the CPU (the
# controller identifies it by onCUDA=0), so it can share the head node.
#
# Usage:
#   ./start_worker_sink.sh [--profile-driven] [-cip CONTROLLER_IP] [-p PORT]
#
# Runs in a window of the tmux session "hadis-head" and returns; --foreground
# runs it inline. Give -cip the head node's real IP on a multi-node run: the
# sink's routing entry is built from the address it registers from, and remote
# workers forward finished queries to it.
set -e
source "$(dirname "$0")/_common.sh"
parse_mode_flag "$@"

CIP=localhost
PORT=50048
set -- "${REST_ARGS[@]}"
while [ $# -gt 0 ]; do
    case "$1" in
        -cip|--controller_ip) CIP="$2"; shift 2 ;;
        -p|--port) PORT="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

require_py
banner "sink worker"
echo "   controller: ${CIP}   port: ${PORT}"
echo

wait_for_port "$CIP" "${CPORT:-50050}" 90 "controller" || {
    echo "ERROR: controller not reachable at ${CIP}:${CPORT:-50050}; start it first." >&2
    exit 1
}

# The sink never runs a model, so it is always simulated regardless of mode.
run_component sink "cd '$ROOT/src/worker' && '$PY' worker.py -cip $CIP -p $PORT --is_sink $MODE_FLAG"
if [ "$USE_TMUX" = "1" ]; then
    wait_for_port localhost "$PORT" 120 "sink"
    component_launched "SINK" sink "$PORT"
    attach_hint
fi
