#!/bin/bash
# Start the load balancer. Run this SECOND, after the controller.
#
# The load balancer must be registered with the controller before any worker
# starts, because workers learn the load balancer's address from the controller
# when they register.
#
# Usage:
#   ./start_load_balancer.sh [--profile-driven] [-cip CONTROLLER_IP] [-p PORT]
#
# Runs in a window of the tmux session "hadis-head" and returns; --foreground
# runs it inline. On a multi-node run give -cip the head node's real IP, not
# localhost: the controller records whatever address the load balancer connects
# from and hands it to the workers, so a loopback registration sends remote
# workers to their own 127.0.0.1.
set -e
source "$(dirname "$0")/_common.sh"
parse_mode_flag "$@"

CIP=localhost
PORT=50049
set -- "${REST_ARGS[@]}"
while [ $# -gt 0 ]; do
    case "$1" in
        -cip|--controller_ip) CIP="$2"; shift 2 ;;
        -p|--port) PORT="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

require_py
banner "load balancer"
echo "   controller: ${CIP}"
echo

# load_balancer.py re-raises if the controller is not reachable, so wait first.
wait_for_port "$CIP" "${CPORT:-50050}" 90 "controller" || {
    echo "ERROR: controller not reachable at ${CIP}:${CPORT:-50050}; start it first." >&2
    exit 1
}

run_component loadb "cd '$ROOT/src/load_balancer' && '$PY' load_balancer.py -cip $CIP -p $PORT $MODE_FLAG"
if [ "$USE_TMUX" = "1" ]; then
    wait_for_port localhost "$PORT" 90 "load balancer"
    component_launched "LOAD BALANCER" loadb "$PORT"
    attach_hint
fi
