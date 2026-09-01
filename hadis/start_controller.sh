#!/bin/bash
# Start the controller. Run this FIRST, on the head node.
#
# Usage:
#   ./start_controller.sh [--profile-driven] [-ap N] [-t TABLE] [-p PORT]
#
#   -ap N   system to run (default 5):
#             0 Clipper-Light  1 Clipper-Heavy  2 INFaaS-Acc
#             3 Proteus        4 DiffServe      5 HADIS
#   -t      cascade table for HADIS: hybrid (default), fixed13, disc_only, router_only
#
# Runs in a window of the tmux session "hadis-head" and returns, so the same
# terminal can start the load balancer next. Pass --foreground to run inline.
#
# Copy the IP printed below; the workers need it as -cip.
set -e
source "$(dirname "$0")/_common.sh"
parse_mode_flag "$@"

AP=5
TABLE=hybrid
PORT=50050
set -- "${REST_ARGS[@]}"
while [ $# -gt 0 ]; do
    case "$1" in
        -ap|--allocation_policy) AP="$2"; shift 2 ;;
        -t|--cascade_table) TABLE="$2"; shift 2 ;;
        -p|--port) PORT="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "$AP" in
    0) SYSTEM="Clipper-Light" ;;  1) SYSTEM="Clipper-Heavy" ;;
    2) SYSTEM="INFaaS-Acc" ;;     3) SYSTEM="Proteus" ;;
    4) SYSTEM="DiffServe" ;;      5) SYSTEM="HADIS" ;;
    *) echo "invalid -ap $AP (expected 0-5)" >&2; exit 1 ;;
esac

require_py
banner "controller"
echo "   system : ${SYSTEM}  (-ap ${AP})"
[ "$AP" = "5" ] && echo "   table  : ${TABLE}"
echo "   ---> workers should use:  -cip $(hostname -I | awk '{print $1}')"
echo

run_component contr "cd '$ROOT/src/controller' && '$PY' controller.py -ap $AP -p $PORT -t $TABLE $MODE_FLAG"
if [ "$USE_TMUX" = "1" ]; then
    wait_for_port localhost "$PORT" 90 "controller"
    component_launched "CONTROLLER" contr "$PORT"
    attach_hint
fi
