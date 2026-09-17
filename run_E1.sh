#!/bin/bash
# Run E1 (paper Figure 7) end to end: all six systems, one after another, on a
# head node plus worker nodes, then plot the figure.
#
# The head node drives the sweep. Each worker node runs a small agent that
# connects to the head node and runs what the head node asks for (start the
# workers, check them, stop them). The agents only need to reach the head node
# over the network, which the workers need anyway, so no SSH or shared file
# system is required.
#
# 1. On the head node (environment active: conda activate hadis):
#
#      ./run_E1.sh --nodes 3                    # expect 3 worker nodes
#      ./run_E1.sh --nodes 3 --dry-run          # checks and plan, launches nothing
#      ./run_E1.sh --nodes 2 --systems 4,5      # only DiffServe and HADIS
#
# 2. On each worker node, in its own terminal (environment active), once:
#
#      ./run_E1.sh --agent --head-ip <HEAD_IP> --node 1 -n 8
#      ./run_E1.sh --agent --head-ip <HEAD_IP> --node 2 -n 4
#      ./run_E1.sh --agent --head-ip <HEAD_IP> --node 3 -n 4
#
#    The head node prints these commands with its IP filled in. Steps 1 and 2
#    can be started in either order. Give every node a different --node, and
#    about one worker per 2 vCPUs of that node. Figure 7 uses 16 workers in
#    total. An agent keeps running after a dry run, so the real run can follow
#    without restarting it, and exits when a real sweep has finished.
#
# Head node options:
#   --nodes K         number of worker nodes (agents) to wait for (required)
#   --systems LIST    comma separated -ap values to run (default 0,1,2,3,4,5)
#   --head-ip IP      this node's address as the workers see it
#                     (default: first address from `hostname -I`)
#   --agent-port P    port the agents connect to (default 50047)
#   --dry-run         wait for the agents, run every check, print the plan
#
# Agent options:
#   --agent           run as the agent of a worker node
#   --head-ip IP      the head node's address (required)
#   --node I          this node's number, 1 to K (sets its port band)
#   -n N              workers to launch on this node
#   --agent-port P    as on the head node
#
# For each system the head node does the steps from Experiments.md, and waits
# for each step to report success before moving on:
#   1. controller, load balancer, sink on the head node
#   2. workers on every node through the agents, then waits until the
#      controller has registered every worker and placed a model on each, and
#      every worker has loaded its model and received a routing table
#   3. client, which blocks until the trace has been replayed
#   4. stop_all.sh on every node, then experiments/collect_logs.sh <system>
# After the last system it runs analysis/plot_e1_end2end.py.
#
# A system that fails at any step is stopped on every node, its logs are filed
# under results/logs/failed_<system>_<time>/ (the plot ignores those), and the
# sweep moves on to the next system. Ctrl-C on the head node stops every node.
# Each system takes about 7 to 8 minutes.
#
# Run the head node from a plain terminal, a `tmux new -s e1run` session or
# with nohup. Do not run it inside a tmux session named hadis-head,
# hadis-workers, hadis-workers-re or hadis-e1: stop_all.sh kills those sessions.

set -u

ART="$(cd "$(dirname "$0")" && pwd)"
HADIS="$ART/hadis"
LOGS="$HADIS/logs"
PY="${PY:-$(command -v python3 || command -v python)}"
export PY
MODE="--profile-driven"

AGENT=0
NUM_NODES=""
SYSTEMS="0,1,2,3,4,5"
HEAD_IP=""
AGENT_PORT=50047
DRY_RUN=0
NODE=""
COUNT=""

READY_TIMEOUT="${READY_TIMEOUT:-300}"   # seconds for the workers to settle
AGENT_TIMEOUT=30                        # an agent silent this long has stopped
FIRST_PORT=50051
PORTS_PER_NODE=50
SINK_PORT=50048

usage() { sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --nodes) NUM_NODES="$2"; shift 2 ;;
        --systems) SYSTEMS="$2"; shift 2 ;;
        --head-ip) HEAD_IP="$2"; shift 2 ;;
        --agent-port) AGENT_PORT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --agent) AGENT=1; shift ;;
        --node) NODE="$2"; shift 2 ;;
        -n) COUNT="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1  (see ./run_E1.sh --help)" >&2; exit 1 ;;
    esac
done

say()  { echo "[$(date +%H:%M:%S)] $*"; }
fail() { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; }
is_int() { [ "$1" -ge "$2" ] 2>/dev/null; }   # is_int <value> <minimum>

node_base_port() { echo $((FIRST_PORT + ($1 - 1) * PORTS_PER_NODE)); }

# HTTP between the head node and the agents, with Python's standard library so
# that nothing else has to be installed.  http <GET|POST> <url> [body file]
HTTP_PY='
import sys, urllib.request, urllib.error
method, url = sys.argv[1], sys.argv[2]
data = open(sys.argv[3], "rb").read() if len(sys.argv) > 3 else None
try:
    req = urllib.request.Request(url, data=data, method=method)
    with urllib.request.urlopen(req, timeout=10) as r:
        sys.stdout.write(r.read().decode())
except urllib.error.HTTPError:
    sys.exit(2)
except Exception:
    sys.exit(1)
'
http() { "$PY" -c "$HTTP_PY" "$@"; }

# =========================================================================== #
# Agent (worker node)
# =========================================================================== #

if [ "$AGENT" = "1" ]; then
    if [ -z "$HEAD_IP" ] || ! is_int "${NODE:-x}" 1 || ! is_int "${COUNT:-x}" 1; then
        echo "usage: ./run_E1.sh --agent --head-ip <HEAD_IP> --node <I> -n <N>" >&2
        exit 1
    fi
    if [ "$COUNT" -gt "$PORTS_PER_NODE" ]; then
        echo "ERROR: -n must be at most $PORTS_PER_NODE" >&2
        exit 1
    fi
    if ! "$PY" -c "import grpc" >/dev/null 2>&1; then
        echo "ERROR: $PY cannot import grpc; activate the environment first (conda activate hadis)" >&2
        exit 1
    fi

    URL="http://$HEAD_IP:$AGENT_PORT"
    AGENT_ID="$(hostname)-$$"
    BASE=$(node_base_port "$NODE")
    MY_PORTS=$(seq "$BASE" $((BASE + COUNT - 1)))
    WORK=$(mktemp -d)

    echo "=============================================================="
    echo " HADIS E1 agent"
    echo "   this node : $(hostname), $(nproc) vCPUs"
    echo "   node      : --node $NODE, $COUNT workers on ports $BASE-$((BASE + COUNT - 1))"
    echo "   head node : $URL"
    echo "=============================================================="
    [ $((COUNT * 2)) -gt "$(nproc)" ] && \
        echo "  NOTE: $COUNT workers on $(nproc) vCPUs; about one worker per 2 vCPUs is advised."

    # Keep telling the head node this agent is alive, also while a task runs.
    ( while true; do http GET "$URL/alive/$NODE/$AGENT_ID" >/dev/null 2>&1; sleep 5; done ) &
    HEARTBEAT=$!
    agent_exit() {
        kill "$HEARTBEAT" 2>/dev/null
        "$HADIS/stop_all.sh" >/dev/null 2>&1
        rm -rf "$WORK"
        exit "$1"
    }
    trap 'echo; say "interrupted; stopped the workers on this node"; agent_exit 130' INT TERM

    # How many of this node's workers have loaded their model and received a
    # routing table from the controller.
    ready_count() {
        local p ok=0
        for p in $MY_PORTS; do
            grep -q 'Check LOAD_MODEL_RESPONSE' "$LOGS/worker_$p.log" 2>/dev/null && \
                grep -q 'Setting routing table at worker' "$LOGS/worker_$p.log" && ok=$((ok + 1))
        done
        echo "$ok"
    }

    run_task() {   # cmd arg
        case "$1" in
            env)
                echo "host=$(hostname)"
                echo "cpus=$(nproc)"
                echo "workers=$COUNT"
                command -v tmux >/dev/null && echo "tmux=ok" || echo "tmux=missing"
                "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1 \
                    && echo "cuda=ok" || echo "cuda=missing"
                [ -f "$LOGS/$2" ] && echo "shared=yes" || echo "shared=no"
                local p n=0
                for p in $MY_PORTS; do
                    [ -f "$LOGS/worker_$p.log" ] && n=$((n + 1))
                    [ -f "$LOGS/model_$p.log" ] && n=$((n + 1))
                done
                echo "stale=$n"
                ;;
            start)
                "$HADIS/stop_all.sh" >/dev/null 2>&1
                "$HADIS/start_worker.sh" $MODE -cip "$HEAD_IP" --node "$NODE" -n "$COUNT"
                ;;
            check)
                ready_count
                ;;
            stop)
                "$HADIS/stop_all.sh"
                ;;
            collect)
                # Only matters when hadis/logs is not shared with the head node:
                # file this node's worker logs so the next system starts clean.
                local p f moved=0
                for p in $MY_PORTS; do
                    for f in "$LOGS/worker_$p.log" "$LOGS/model_$p.log"; do
                        [ -f "$f" ] || continue
                        mkdir -p "$ART/results/logs/$2"
                        mv "$f" "$ART/results/logs/$2/" && moved=$((moved + 1))
                    done
                done
                echo "moved $moved log file(s) to results/logs/$2/"
                ;;
            *)
                echo "unknown task: $1"
                return 1
                ;;
        esac
    }

    say "Waiting for the head node ..."
    last=""
    connected=0
    while true; do
        if ! reply=$(http GET "$URL/task/$NODE/$AGENT_ID" 2>/dev/null); then
            if [ "$connected" = "1" ]; then
                say "Lost the head node; waiting for it to come back ..."
                connected=0
            fi
            sleep 3
            continue
        fi
        read -r session id cmd arg <<< "$reply"
        case "$cmd" in
            wait) sleep 1; continue ;;
            hello)
                echo "$AGENT_ID $COUNT $(hostname) $(nproc)" > "$WORK/hello"
                if http POST "$URL/hello/$NODE" "$WORK/hello" >/dev/null 2>&1; then
                    [ "$connected" = "0" ] && say "Connected to the head node as --node $NODE"
                    connected=1
                fi
                sleep 1
                continue
                ;;
            conflict)
                fail "--node $NODE is already taken by the agent on $arg; give this node a different --node"
                agent_exit 1
                ;;
        esac
        [ "$session:$id" = "$last" ] && { sleep 1; continue; }
        last="$session:$id"

        if [ "$cmd" = "exit" ]; then
            say "The head node has finished the sweep; stopping the workers and exiting."
            echo "0" > "$WORK/result"
            http POST "$URL/result/$NODE/$id" "$WORK/result" >/dev/null 2>&1
            agent_exit 0
        fi

        say "task: $cmd ${arg:-}"
        run_task "$cmd" "${arg:-}" > "$WORK/out" 2>&1
        rc=$?
        sed 's/^/      | /' "$WORK/out"
        { echo "$rc"; cat "$WORK/out"; } > "$WORK/result"
        http POST "$URL/result/$NODE/$id" "$WORK/result" >/dev/null 2>&1 || \
            say "could not send the result to the head node"
    done
fi

# =========================================================================== #
# Head node
# =========================================================================== #

if ! is_int "${NUM_NODES:-x}" 1; then
    echo "ERROR: --nodes <number of worker nodes> is required (or --agent on a worker node)" >&2
    echo "       see ./run_E1.sh --help" >&2
    exit 1
fi

system_name() {   # -ap value -> log name (Experiments.md section 1)
    case "$1" in
        0) echo clipper_light ;; 1) echo clipper_heavy ;; 2) echo infaas ;;
        3) echo proteus ;;       4) echo diffserve ;;     5) echo hadis ;;
        *) return 1 ;;
    esac
}
system_label() {
    case "$1" in
        0) echo Clipper-Light ;; 1) echo Clipper-Heavy ;; 2) echo INFaaS-Acc ;;
        3) echo Proteus ;;       4) echo DiffServe ;;     5) echo HADIS ;;
    esac
}

IFS=',' read -r -a AP_ARR <<< "$SYSTEMS"
for ap in "${AP_ARR[@]}"; do
    system_name "$ap" >/dev/null || { echo "ERROR: invalid system '$ap' in --systems (expected 0-5)" >&2; exit 1; }
done

[ -z "$HEAD_IP" ] && HEAD_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "$HEAD_IP" ]; then
    echo "ERROR: could not find this node's address; pass --head-ip" >&2
    exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
SESSION="$STAMP-$$"
STATE=$(mktemp -d)
RUN_LOG="$ART/results/logs/run_E1_$STAMP.log"

# --------------------------------------------------------------------------- #
# Coordinator: a small HTTP server the agents poll. It keeps its state as files
# in $STATE, which the rest of this script reads and writes:
#   task_<node>          the node's current task, "<session> <id> <cmd> <arg>"
#   result_<node>_<id>   the exit status on the first line, then the output
#   hello_<node>         "<agent id> <workers> <host> <vCPUs>" of the agent
#   seen_<node>          touched whenever that agent calls in
# --------------------------------------------------------------------------- #

SERVER_PY='
import http.server, os, sys, time
state, port, stale = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])

def path(name):
    return os.path.join(state, name)

def write(name, data):
    tmp = path("." + name)
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path(name))

def owner(node):
    try:
        with open(path("hello_" + node)) as f:
            return f.read().split()
    except FileNotFoundError:
        return None

def alive(node):
    try:
        return time.time() - os.path.getmtime(path("seen_" + node)) < stale
    except FileNotFoundError:
        return False

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body):
        body = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.strip("/").split("/")
        if len(p) != 3 or p[0] not in ("task", "alive") or not p[1].isdigit():
            return self.reply(404, "")
        node, agent = p[1], p[2]
        o = owner(node)
        if o is None or (o[0] != agent and not alive(node)):
            return self.reply(200, "0 0 hello")
        if o[0] != agent:
            return self.reply(200, "0 0 conflict " + (o[2] if len(o) > 2 else "another node"))
        write("seen_" + node, b"")
        if p[0] == "alive":
            return self.reply(200, "ok")
        try:
            with open(path("task_" + node), "rb") as f:
                return self.reply(200, f.read())
        except FileNotFoundError:
            return self.reply(200, "0 0 wait")

    def do_POST(self):
        p = self.path.strip("/").split("/")
        data = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if len(p) == 2 and p[0] == "hello" and p[1].isdigit():
            write("hello_" + p[1], data)
            write("seen_" + p[1], b"")
            return self.reply(200, "ok")
        if len(p) == 3 and p[0] == "result" and p[1].isdigit() and p[2].isdigit():
            write("result_%s_%s" % (p[1], p[2]), data)
            return self.reply(200, "ok")
        self.reply(404, "")

http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
'

SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    rm -rf "$STATE"
}
trap cleanup EXIT

start_coordinator() {
    "$PY" -c "$SERVER_PY" "$STATE" "$AGENT_PORT" 15 > "$STATE/server.log" 2>&1 &
    SERVER_PID=$!
    local waited=0
    until http GET "http://127.0.0.1:$AGENT_PORT/task/0/probe" >/dev/null 2>&1; do
        sleep 1; waited=$((waited + 1))
        if ! kill -0 "$SERVER_PID" 2>/dev/null || [ "$waited" -ge 10 ]; then
            fail "could not listen on port $AGENT_PORT (in use?); pick another with --agent-port"
            sed 's/^/      | /' "$STATE/server.log" | tail -3
            SERVER_PID=""
            return 1
        fi
    done
}

NODE_HOST=()   # index i-1: host name, workers, vCPUs of node i
NODE_COUNT=()
NODE_CPUS=()
TOTAL_WORKERS=0

wait_for_agents() {
    local i waited=0 missing
    while true; do
        missing=""
        for i in $(seq 1 "$NUM_NODES"); do
            [ -f "$STATE/hello_$i" ] || missing="$missing $i"
        done
        [ -z "$missing" ] && break
        # Running agents reconnect within a few seconds, so only then ask.
        if [ "$waited" -ge 6 ] && [ $(( (waited - 6) % 60 )) -eq 0 ]; then
            say "Waiting for agents on node(s):$missing. On each of those worker nodes run:"
            for i in $missing; do
                echo "      ./run_E1.sh --agent --head-ip $HEAD_IP --node $i -n <workers>$([ "$AGENT_PORT" != 50047 ] && echo " --agent-port $AGENT_PORT")"
            done
        fi
        sleep 2; waited=$((waited + 2))
    done
    for i in $(seq 1 "$NUM_NODES"); do
        read -r _ count host cpus < "$STATE/hello_$i"
        NODE_COUNT+=("$count"); NODE_HOST+=("$host"); NODE_CPUS+=("$cpus")
        TOTAL_WORKERS=$((TOTAL_WORKERS + count))
    done
    for f in "$STATE"/hello_*; do
        i=${f##*_}
        [ "$i" -gt "$NUM_NODES" ] && say "NOTE: an agent registered as --node $i, but --nodes is $NUM_NODES; it is ignored."
    done
    return 0
}

# Tasks for the agents.
TASK_ID=0
declare -A TASK_OF
send_task() {   # node cmd [arg]
    TASK_ID=$((TASK_ID + 1))
    echo "$SESSION $TASK_ID $2 ${3:-}" > "$STATE/.task_$1"
    mv "$STATE/.task_$1" "$STATE/task_$1"
    TASK_OF[$1]=$TASK_ID
}

# await <node> <timeout>: wait for the node's current task to finish. The
# output is left in $STATE/result_<node>_<id>; returns the task's exit status,
# or 124 if the agent went silent or the timeout passed.
await() {
    local node=$1 timeout=$2 waited=0 f seen
    f="$STATE/result_${node}_${TASK_OF[$node]}"
    while [ ! -f "$f" ]; do
        sleep 1; waited=$((waited + 1))
        seen=$(stat -c %Y "$STATE/seen_$node" 2>/dev/null || echo 0)
        if [ $(( $(date +%s) - seen )) -gt "$AGENT_TIMEOUT" ]; then
            fail "the agent on node $node (${NODE_HOST[$((node - 1))]:-?}) stopped responding"
            return 124
        fi
        if [ "$waited" -ge "$timeout" ]; then
            fail "node $node did not finish its task within ${timeout}s"
            return 124
        fi
    done
    return "$(head -1 "$f")"
}
result_text() { tail -n +2 "$STATE/result_${1}_${TASK_OF[$1]}" 2>/dev/null; }

# on_all_nodes <cmd> <arg> <timeout> [expected text]: run a task on every node
# at once, print each node's output, succeed only if all succeed.
on_all_nodes() {
    local cmd=$1 arg=$2 timeout=$3 expect=${4:-} i rc ok=0
    for i in $(seq 1 "$NUM_NODES"); do send_task "$i" "$cmd" "$arg"; done
    for i in $(seq 1 "$NUM_NODES"); do
        await "$i" "$timeout"; rc=$?
        result_text "$i" | sed "s/^/      | [node $i] /"
        if [ "$rc" -ne 0 ]; then
            [ "$rc" -ne 124 ] && fail "node $i: '$cmd' exited with status $rc"
            ok=1
        elif [ -n "$expect" ] && ! result_text "$i" | grep -q -- "$expect"; then
            fail "node $i: did not see '$expect' in the output"
            ok=1
        fi
    done
    return $ok
}

# Run a head-node step, indent its output, and succeed only if it exits 0 and
# printed the expected line. step <expected text> <command...>
step() {
    local expect="$1" out rc; shift
    out=$(mktemp)
    "$@" 2>&1 | sed -u 's/^/      | /' | tee "$out"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        fail "command exited with status $rc"
    elif ! grep -q -- "$expect" "$out"; then
        fail "did not see '$expect' in the output"
        rc=1
    fi
    rm -f "$out"
    return "$rc"
}

controller_log() { ls -t "$LOGS"/controller_*.log 2>/dev/null | head -1; }

# wait_until <timeout> <label> <command...>: retry the command every 2 s.
wait_until() {
    local timeout=$1 label=$2 waited=0; shift 2
    while ! "$@"; do
        sleep 2; waited=$((waited + 2))
        if [ "$waited" -ge "$timeout" ]; then
            fail "$label: not reached after ${timeout}s"
            return 1
        fi
    done
    say "  ok: $label (${waited}s)"
}

log_has() {   # pattern: the current controller log contains it
    local f; f=$(controller_log)
    [ -n "$f" ] && grep -Eq -- "$1" "$f"
}

# "<models placed on GPU workers> <sink>" from the controller's latest allocation.
# Every system logs its allocation each planning interval, under one of two names.
allocation() {
    local f; f=$(controller_log)
    [ -n "$f" ] || { echo "0 0"; return; }
    "$PY" - "$f" <<'PY'
import ast, re, sys
last = None
with open(sys.argv[1], errors='ignore') as fh:
    for line in fh:
        m = re.search(r'AllocatedModels (?:after ReAlloc|AFTER loading): (\{.*\})', line)
        if m:
            last = m.group(1)
if last is None:
    print('0 0')
else:
    d = ast.literal_eval(last)
    print(sum(v for k, v in d.items() if k != 'sink'), d.get('sink', 0))
PY
}
sink_placed()     { local a; read -r -a a <<< "$(allocation)"; [ "${a[1]}" -ge 1 ]; }
workers_placed()  { local a; read -r -a a <<< "$(allocation)"; [ "${a[0]}" -ge "$TOTAL_WORKERS" ] && [ "${a[1]}" -ge 1 ]; }
workers_registered() {
    local f n; f=$(controller_log)
    [ -n "$f" ] || return 1
    n=$(grep -c 'Established GRPC connection with worker' "$f")
    [ "$n" -ge $((TOTAL_WORKERS + 1)) ]      # the workers plus the sink
}
workers_routed() {
    local i n
    for i in $(seq 1 "$NUM_NODES"); do send_task "$i" check; done
    for i in $(seq 1 "$NUM_NODES"); do
        await "$i" 60 || return 1
        n=$(result_text "$i" | tail -1)
        [ "${n:-0}" -ge "${NODE_COUNT[$((i - 1))]}" ] 2>/dev/null || return 1
    done
}

stop_everywhere() {
    on_all_nodes stop "" 120 || true
    "$HADIS/stop_all.sh" 2>&1 | sed "s/^/      | [head] /"
}

# --------------------------------------------------------------------------- #
# Checks (also run by --dry-run)
# --------------------------------------------------------------------------- #

echo "=============================================================="
echo " HADIS E1: all systems, multi-node"
echo "   head node : $(hostname)  ($HEAD_IP), agents connect to port $AGENT_PORT"
echo "   nodes     : $NUM_NODES worker node(s)"
echo "   systems   : $(for ap in "${AP_ARR[@]}"; do printf '%s ' "$(system_label "$ap")"; done)"
echo "   python    : $PY"
echo "=============================================================="
echo

nfail=0
check() {   # check <label> <0|1> [detail]
    if [ "$2" = "0" ]; then printf "  [ ok ] %-44s %s\n" "$1" "${3:-}"
    else printf "  [FAIL] %-44s %s\n" "$1" "${3:-}"; nfail=$((nfail + 1)); fi
}

say "Checking the head node"
if [ -n "${TMUX:-}" ]; then
    sess=$(tmux display-message -p '#S' 2>/dev/null)
    case "$sess" in
        hadis-head|hadis-workers|hadis-workers-re|hadis-e1)
            check "not inside tmux session '$sess'" 1 "stop_all.sh would kill this script" ;;
    esac
fi
if "$PY" -c "import grpc" >/dev/null 2>&1; then
    check "python environment" 0 "$PY"
else
    check "python environment" 1 "$PY cannot import grpc; conda activate hadis"
fi
if command -v tmux >/dev/null; then check "tmux available" 0; else check "tmux available" 1; fi

need_gurobi=0
for ap in "${AP_ARR[@]}"; do [ "$ap" -ge 2 ] && need_gurobi=1; done
if [ "$need_gurobi" = "1" ]; then
    if GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HADIS/gurobi/gurobi.lic}" "$PY" - >/dev/null 2>&1 <<'PY'
import gurobipy as gp
m = gp.Model(); m.setParam('OutputFlag', 0)
x = m.addVar(ub=1); m.setObjective(x, gp.GRB.MAXIMIZE); m.optimize()
raise SystemExit(0 if m.status == gp.GRB.OPTIMAL else 1)
PY
    then check "Gurobi licence valid" 0 "needed by systems 2 to 5"
    else check "Gurobi licence valid" 1 "needed by systems 2 to 5"
    fi
fi

leftover=$(find "$LOGS" -maxdepth 1 -type f ! -name '.*' 2>/dev/null | wc -l)
if [ "$leftover" -eq 0 ]; then
    check "hadis/logs is empty" 0
else
    check "hadis/logs is empty" 1 "$leftover file(s); collect_logs.sh <name> or remove_logs.sh first"
fi

if [ "$nfail" -gt 0 ]; then
    echo
    fail "$nfail check(s) failed on the head node; fix them first."
    exit 1
fi

echo
start_coordinator || exit 1
say "Listening for agents on port $AGENT_PORT"
wait_for_agents

echo
say "Checking the worker nodes"
MARKER=".run_E1_$SESSION"
: > "$LOGS/$MARKER"
on_all_nodes env "$MARKER" 120 >/dev/null
rm -f "$LOGS/$MARKER"
SHARED=()
for i in $(seq 1 "$NUM_NODES"); do
    out=$(result_text "$i")
    val() { echo "$out" | sed -n "s/^$1=//p"; }
    base=$(node_base_port "$i")
    label="node $i ($(val host))"
    if [ -z "$(val host)" ]; then
        check "node $i: agent responded" 1 "no answer to the checks"
        SHARED+=(0)
        continue
    fi
    check "$label: agent responded" 0 "$(val workers) workers on $(val cpus) vCPUs, ports $base-$((base + NODE_COUNT[i - 1] - 1))"
    if [ "$(val tmux)" = ok ]; then check "$label: tmux available" 0; else check "$label: tmux available" 1; fi
    if [ "$(val cuda)" = ok ]; then check "$label: CUDA visible" 0; else check "$label: CUDA visible" 1 "workers need a GPU to register"; fi
    if [ "$(val shared)" = yes ]; then
        SHARED+=(1)
        check "$label: hadis/logs" 0 "shared with the head node"
    else
        SHARED+=(0)
        stale=$(val stale)
        if [ "${stale:-0}" -eq 0 ]; then
            check "$label: hadis/logs" 0 "own directory, no old worker logs"
        else
            check "$label: hadis/logs" 1 "$stale old worker log(s) for this node's ports; remove them"
        fi
    fi
done
echo "  total workers: $TOTAL_WORKERS$([ "$TOTAL_WORKERS" -ne 16 ] && echo "   (NOTE: Figure 7 uses 16)")"
echo

# --------------------------------------------------------------------------- #
# Dry run: print the plan
# --------------------------------------------------------------------------- #

if [ "$DRY_RUN" = "1" ]; then
    say "DRY RUN: nothing is launched. For each system, in this order:"
    for ap in "${AP_ARR[@]}"; do
        name=$(system_name "$ap")
        echo
        echo "  == $(system_label "$ap")  (-ap $ap, collected as results/logs/$name/)"
        echo "    [head]    hadis/start_controller.sh $MODE -ap $ap"
        echo "              wait for 'CONTROLLER LAUNCHED' and the controller's planning loop"
        echo "    [head]    hadis/start_load_balancer.sh $MODE -cip $HEAD_IP"
        echo "              wait for 'LOAD BALANCER LAUNCHED' and the controller connecting to it"
        echo "    [head]    hadis/start_worker_sink.sh $MODE -cip $HEAD_IP"
        echo "              wait for 'SINK LAUNCHED' and the controller placing the sink"
        for i in $(seq 1 "$NUM_NODES"); do
            echo "    [node $i]  hadis/start_worker.sh $MODE -cip $HEAD_IP --node $i -n ${NODE_COUNT[$((i - 1))]}   (${NODE_HOST[$((i - 1))]})"
        done
        echo "              wait for 'All N workers are up' on each node, then until the"
        echo "              controller has registered $TOTAL_WORKERS workers and placed a model on each,"
        echo "              and every worker has loaded it and received a routing table"
        echo "    [head]    hadis/start_client.sh $MODE -lbip $HEAD_IP"
        echo "              blocks until 'TRACE ENDED'"
        echo "    [node *]  hadis/stop_all.sh"
        echo "    [head]    hadis/stop_all.sh"
        echo "    [head]    experiments/collect_logs.sh $name"
    done
    echo
    echo "  == Finally"
    echo "    [head]    analysis/plot_e1_end2end.py   -> results/figures/fig7_end2end.png"
    echo
    if [ "$nfail" -gt 0 ]; then
        say "$nfail check(s) failed above; fix them before a real run."
        exit 1
    fi
    say "All checks passed. The agents keep running: rerun this command without"
    say "--dry-run to start (about $(( ${#AP_ARR[@]} * 8 )) minutes)."
    exit 0
fi

if [ "$nfail" -gt 0 ]; then
    fail "$nfail check(s) failed; nothing launched."
    exit 1
fi

# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #

mkdir -p "$ART/results/logs"
exec > >(tee -a "$RUN_LOG") 2>&1
say "Output is also saved to results/logs/$(basename "$RUN_LOG")"

on_interrupt() {
    echo
    fail "interrupted; stopping every node. The current run's logs stay in hadis/logs/."
    stop_everywhere
    exit 130
}
trap on_interrupt INT TERM

run_system() {   # ap
    local ap=$1

    say "[1/6] controller"
    step "CONTROLLER LAUNCHED" "$HADIS/start_controller.sh" $MODE -ap "$ap" || return 1
    wait_until 60 "controller planning loop running" log_has 'AllocatedModels' || return 1

    say "[2/6] load balancer"
    step "LOAD BALANCER LAUNCHED" "$HADIS/start_load_balancer.sh" $MODE -cip "$HEAD_IP" || return 1
    wait_until 60 "controller connected to the load balancer" \
        log_has 'Established GRPC connection with load balancer' || return 1

    say "[3/6] sink"
    step "SINK LAUNCHED" "$HADIS/start_worker_sink.sh" $MODE -cip "$HEAD_IP" || return 1
    wait_until 60 "controller registered the sink" \
        log_has "Established GRPC connection with worker .*port: $SINK_PORT" || return 1
    wait_until 60 "controller placed the sink" sink_placed || return 1

    say "[4/6] workers on $NUM_NODES node(s)"
    on_all_nodes start "" 700 "workers are up" || return 1
    wait_until "$READY_TIMEOUT" "controller registered all $TOTAL_WORKERS workers" workers_registered || return 1
    wait_until "$READY_TIMEOUT" "controller placed a model on all $TOTAL_WORKERS workers" workers_placed || return 1
    wait_until "$READY_TIMEOUT" "every worker loaded its model and has a routing table" workers_routed || return 1
    say "  everything is settled: $(allocation | awk '{print $1}') workers serving"

    say "[5/6] client (blocks until the trace ends, about 6 minutes)"
    step "TRACE ENDED" "$HADIS/start_client.sh" $MODE -lbip "$HEAD_IP" || return 1
    return 0
}

declare -A RESULT
sweep_start=$(date +%s)
for ap in "${AP_ARR[@]}"; do
    name=$(system_name "$ap")
    label=$(system_label "$ap")
    echo
    echo "=============================================================="
    say "$label  (-ap $ap)"
    echo "=============================================================="
    t0=$(date +%s)

    # Start from nothing running anywhere.
    stop_everywhere >/dev/null 2>&1

    if run_system "$ap"; then
        status=ok
    else
        status=failed
    fi

    say "[6/6] stop every node and collect"
    stop_everywhere
    if [ "$status" = "ok" ]; then
        "$ART/experiments/collect_logs.sh" "$name" 2>&1 | sed 's/^/      | /'
        rows=0
        csv="$ART/results/logs/$name/slo_timeouts_per_second.csv"
        [ -f "$csv" ] && rows=$(($(wc -l < "$csv") - 1))
        [ "$rows" -gt 0 ] || status=failed
        RESULT[$ap]="$status  $rows seconds of results in results/logs/$name/"
    else
        name="failed_${name}_$(date +%Y%m%d_%H%M%S)"
        if "$ART/experiments/collect_logs.sh" "$name" >/dev/null 2>&1; then
            RESULT[$ap]="failed, logs in results/logs/$name/"
        else
            RESULT[$ap]="failed, no logs were written"
        fi
    fi
    # Nodes with their own hadis/logs file their worker logs on their side.
    for i in $(seq 1 "$NUM_NODES"); do
        [ "${SHARED[$((i - 1))]}" = "1" ] || send_task "$i" collect "$name"
    done
    for i in $(seq 1 "$NUM_NODES"); do
        [ "${SHARED[$((i - 1))]}" = "1" ] || await "$i" 60 >/dev/null
    done
    say "$label: ${RESULT[$ap]}  ($(( ($(date +%s) - t0) / 60 )) min)"
done

trap - INT TERM
for i in $(seq 1 "$NUM_NODES"); do send_task "$i" exit; done
for i in $(seq 1 "$NUM_NODES"); do await "$i" 30 >/dev/null 2>&1; done

echo
echo "=============================================================="
say "Plotting Figure 7"
echo "=============================================================="
"$PY" "$ART/analysis/plot_e1_end2end.py" 2>&1 | sed 's/^/      | /'
plot_rc=${PIPESTATUS[0]}

echo
echo "=============================================================="
echo " E1 summary  ($(( ($(date +%s) - sweep_start) / 60 )) min)"
for ap in "${AP_ARR[@]}"; do
    printf "   %-14s %s\n" "$(system_label "$ap")" "${RESULT[$ap]}"
done
if [ "$plot_rc" -eq 0 ]; then
    echo "   figure         results/figures/fig7_end2end.png"
else
    echo "   figure         plotting failed, see above"
fi
echo "=============================================================="

for ap in "${AP_ARR[@]}"; do
    case "${RESULT[$ap]}" in ok*) ;; *) exit 1 ;; esac
done
[ "$plot_rc" -eq 0 ]
