#!/bin/bash
# Start the client and replay the workload trace. Run this LAST, once every
# worker is up (start_worker.sh waits for that and says so).
#
# Usage:
#   ./start_client.sh [--profile-driven] [-lbip LOAD_BALANCER_IP] [-trace NAME]
#
#   -trace   a path, or a short name resolved as
#            traces/maf/multi_dynamic_testbed/trace_<name>qps.txt.
#            Defaults to the trace for the active mode:
#              profile-driven  trace_64qps.txt    peaks at 64 QPS over ~347 s
#              real            trace_1_6qps.txt   peaks at 1.6 QPS over ~3470 s
#
# Runs in a window of the tmux session "hadis-head" and BLOCKS until the trace
# has been fully replayed, then returns to the terminal -- so you do not have to
# attach to tmux to find out when the run is done. Progress is printed while it
# waits. --foreground runs the client inline instead (it will not return, since
# the client keeps serving after the trace ends). --no-wait launches and returns
# immediately.
#
# Results are the CSVs in logs/; collect them with experiments/collect_logs.sh
# before the next run.
set -e
source "$(dirname "$0")/_common.sh"
parse_mode_flag "$@"

LBIP=localhost
TRACE=""
WAIT_FOR_TRACE=1
set -- "${REST_ARGS[@]}"
while [ $# -gt 0 ]; do
    case "$1" in
        -lbip|--lb_ip) LBIP="$2"; shift 2 ;;
        -trace|--trace_file) TRACE="$2"; shift 2 ;;
        --no-wait) WAIT_FOR_TRACE=0; shift ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

require_py
banner "client"
echo "   load balancer: ${LBIP}"
echo

wait_for_port "$LBIP" "${LBPORT:-50049}" 90 "load balancer" || {
    echo "ERROR: load balancer not reachable at ${LBIP}:${LBPORT:-50049}; start it first." >&2
    exit 1
}

CLIENT_CMD="cd '$ROOT/src/client' && '$PY' client.py -lbip $LBIP $MODE_FLAG"
[ -n "$TRACE" ] && CLIENT_CMD="cd '$ROOT/src/client' && '$PY' client.py -lbip $LBIP -trace $TRACE $MODE_FLAG"

# Marker so this run's client log can be told apart from any left over.
MARKER="$ROOT/logs/.client_start_marker"
: > "$MARKER"

run_component client "$CLIENT_CMD"

if [ "$USE_TMUX" != "1" ]; then
    exit 0          # unreachable: run_component exec'd the client
fi

echo
echo "  >>> CLIENT LAUNCHED  (tmux ${HEAD_SESSION}:client)"
attach_hint

if [ "$WAIT_FOR_TRACE" != "1" ]; then
    echo "  --no-wait: returning immediately. Watch for 'Trace ended' yourself."
    exit 0
fi

# The client sleeps 10 s, waits for a routing table, then replays the trace:
# ~347 s in profile-driven mode, ~3470 s in real mode.
if [ -n "$MODE_FLAG" ]; then
    TIMEOUT="${TRACE_TIMEOUT:-1200}"
else
    TIMEOUT="${TRACE_TIMEOUT:-5400}"
fi

client_log() {
    find "$ROOT/logs" -maxdepth 1 -name 'client_*.log' -newer "$MARKER" 2>/dev/null | head -1
}
served_rows() {
    local f="$ROOT/logs/slo_timeouts_per_second.csv"
    [ -f "$f" ] && echo $(( $(wc -l < "$f") - 1 )) || echo 0
}

echo
echo "  Replaying the trace (timeout ${TIMEOUT}s). This terminal will return when it ends."
waited=0
status=timeout
while [ "$waited" -lt "$TIMEOUT" ]; do
    log=$(client_log)
    if [ -n "$log" ] && grep -q "Trace ended" "$log" 2>/dev/null; then
        status=ended
        break
    fi
    if [ "$waited" -ge 30 ] && ! python_proc_running client.py; then
        status=died
        break
    fi
    sleep 5
    waited=$((waited + 5))
    if [ $((waited % 30)) -eq 0 ]; then
        echo "    ... ${waited}s elapsed, $(served_rows) seconds of results recorded"
    fi
done

rm -f "$MARKER"

echo
case "$status" in
    ended)
        echo "  >>> TRACE ENDED after ${waited}s  ($(served_rows) seconds of results in logs/)"
        echo
        echo "  Next:  ./stop_all.sh                    (on this node AND each worker node)"
        echo "         ../experiments/collect_logs.sh <name>"
        ;;
    died)
        echo "  >>> ERROR: the client exited without reporting 'Trace ended'." >&2
        echo "      Check: tmux attach -t ${HEAD_SESSION}   (window: client)" >&2
        exit 1
        ;;
    timeout)
        echo "  >>> WARNING: no 'Trace ended' after ${TIMEOUT}s; the client is still running." >&2
        echo "      Check: tmux attach -t ${HEAD_SESSION}   (window: client)" >&2
        echo "      Raise the limit with TRACE_TIMEOUT=<seconds>." >&2
        exit 1
        ;;
esac
