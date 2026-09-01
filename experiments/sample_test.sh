#!/bin/bash
# Sample test: kick the tyres on E1 before committing to the full sweep.
#
# Runs the whole stack on ONE node with a short trace and reports whether every
# piece an E1 run depends on is in place: the Python environment from
# requirements.txt, the Gurobi licence, the repository's data files, and the
# five components talking to each other (controller, load balancer, sink,
# workers, client).
#
# Usage:
#   experiments/sample_test.sh              # 4 workers, ~3 minutes
#   NUM_WORKERS=8 experiments/sample_test.sh
#
# Needs one node with a CUDA GPU and 16 CPU cores. No model is loaded: E1 is
# profile-driven, so the workers sleep on profiled latencies and only need CUDA
# to be visible. That also means this test says nothing about E2's checkpoints;
# for those, run experiments/profile_gpu.py, which downloads and profiles them.
#
# It uses -ap 5 (HADIS) on purpose, so the MILP runs and the Gurobi licence is
# exercised rather than assumed.
set -u

ART="$(cd "$(dirname "$0")/.." && pwd)"
HADIS="$ART/hadis"
NUM_WORKERS="${NUM_WORKERS:-4}"
PY="${PY:-$(command -v python3 || command -v python)}"
MODE="--profile-driven"
AP=5
TRACE="$HADIS/traces/maf/smoke/trace_smokeqps.txt"

pass=0; fail=0
check() {  # check <name> <0|1> [detail]
    if [ "$2" = "0" ]; then printf "  [ ok ] %-42s %s\n" "$1" "${3:-}"; pass=$((pass+1))
    else printf "  [FAIL] %-42s %s\n" "$1" "${3:-}"; fail=$((fail+1)); fi
}

echo "==============================================================="
echo " HADIS sample test: is this machine ready to run E1?"
echo "==============================================================="
echo
echo "-- 1. Python environment ---------------------------------------"
[ -x "$PY" ]; check "interpreter found" $? "$($PY -V 2>&1)"
$PY - <<'PY' 2>/dev/null
import sys
mods = ['grpc', 'grpc_tools', 'google.protobuf', 'gurobipy', 'torch', 'torchvision',
        'diffusers', 'transformers', 'safetensors', 'huggingface_hub', 'sentencepiece',
        'numpy', 'pandas', 'scipy', 'PIL', 'matplotlib', 'brokenaxes']
missing = []
for m in mods:
    try: __import__(m)
    except Exception: missing.append(m)
print(' '.join(missing))
sys.exit(1 if missing else 0)
PY
check "requirements.txt packages importable" $?

echo
echo "-- 2. Gurobi ---------------------------------------------------"
[ -f "$HADIS/gurobi/gurobi.lic" ]; check "licence file at hadis/gurobi/gurobi.lic" $?
GRB_LICENSE_FILE="$HADIS/gurobi/gurobi.lic" $PY - >/dev/null 2>&1 <<'PY'
import gurobipy as gp
m = gp.Model(); m.setParam('OutputFlag', 0)
x = m.addVar(ub=1); m.setObjective(x, gp.GRB.MAXIMIZE); m.optimize()
raise SystemExit(0 if m.status == gp.GRB.OPTIMAL else 1)
PY
check "licence valid (solves a 1-variable LP)" $?

echo
echo "-- 3. Repository data ------------------------------------------"
for f in traces/apps/multi_models.json \
         traces/maf/multi_dynamic_testbed/trace_64qps.txt \
         traces/text_imagenet1k_hr_5k.txt \
         router/prompt_features.csv \
         discriminator/confidence_scores/scores_model_0.txt \
         src/tables/hybrid.py; do
    [ -f "$HADIS/$f" ]; check "$f" $?
done

echo
echo "-- 4. Machine --------------------------------------------------"
command -v tmux >/dev/null; check "tmux available" $?
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
check "CUDA visible to torch" $? "$(nvidia-smi -L 2>/dev/null | head -1 | cut -c1-40)"
cores=$(nproc 2>/dev/null || echo 0)
[ "$cores" -ge 16 ]; check "at least 16 CPU cores" $? "found $cores"

if [ "$fail" -gt 0 ]; then
    echo
    echo "  $fail precondition(s) failed; not launching. Fix the above first."
    exit 1
fi

echo
echo "-- 5. End-to-end run (-ap 5 HADIS, $NUM_WORKERS workers, 39 s trace) ----"
"$HADIS/stop_all.sh" >/dev/null 2>&1 || true

# hadis/logs is where a real run writes, and this test needs it empty to judge
# its own output. Refuse rather than delete: an uncollected E1 run left here is
# somebody's results, and collect_logs.sh is how it gets filed.
existing=$(ls "$HADIS"/logs/*.log "$HADIS"/logs/*.csv "$HADIS"/logs/*.txt 2>/dev/null | wc -l)
if [ "$existing" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
    echo
    echo "  hadis/logs/ already holds $existing file(s) from an earlier run."
    echo "  File them first so they are not lost:"
    echo "      experiments/collect_logs.sh <name>     # moves them to results/logs/<name>/"
    echo "  or discard them and rerun with:"
    echo "      FORCE=1 experiments/sample_test.sh"
    exit 1
fi
rm -f "$HADIS"/logs/*.log "$HADIS"/logs/*.csv "$HADIS"/logs/*.txt 2>/dev/null || true

cleanup() { "$HADIS/stop_all.sh" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# Launch exactly the way a reviewer does. Each start_*.sh puts its component
# into the hadis-head tmux session itself and returns once that component is
# accepting connections, so no extra tmux wrapping and no sleeps are needed --
# and this way the test exercises the real launch path rather than a
# substitute.
start=$(date +%s)
"$HADIS/start_controller.sh"    $MODE -ap $AP                        >/dev/null || exit 1
"$HADIS/start_load_balancer.sh" $MODE -cip localhost                 >/dev/null || exit 1
"$HADIS/start_worker_sink.sh"   $MODE -cip localhost                 >/dev/null || exit 1
"$HADIS/start_worker.sh"        $MODE -cip localhost --node 1 -n "$NUM_WORKERS" >/dev/null || exit 1
"$HADIS/start_client.sh"        $MODE -lbip localhost -trace "$TRACE" --no-wait >/dev/null || exit 1

echo "  running the trace (up to 150 s)..."
waited=0
while [ "$waited" -lt 150 ]; do
    grep -qi "trace ended" "$HADIS"/logs/client_*.log 2>/dev/null && break
    sleep 5; waited=$((waited + 5))
done
sleep 4
cleanup; trap - EXIT
elapsed=$(( $(date +%s) - start ))

echo
echo "-- 6. Results --------------------------------------------------"
grep -qi "trace ended" "$HADIS"/logs/client_*.log 2>/dev/null
check "client replayed the trace" $? "${elapsed}s"
for f in slo_timeouts_per_second.csv query_num_per_second.csv cascade_config_per_second.csv; do
    rows=""
    if [ -f "$HADIS/logs/$f" ]; then
        n=$(wc -l < "$HADIS/logs/$f")
        rows="$((n - 1)) rows"
    fi
    [ -s "$HADIS/logs/$f" ]
    check "controller wrote $f" $? "$rows"
done

nmodel=$(ls "$HADIS"/logs/model_*.log 2>/dev/null | wc -l)
expected=$((NUM_WORKERS + 1))
ls "$HADIS"/logs/model_*.log >/dev/null 2>&1
check "workers logged execution" $? "$nmodel of $expected, workers plus sink"

nplans=$(grep -h -c 'Model plan' "$HADIS"/logs/controller_*.log 2>/dev/null | head -1)
[ -n "$nplans" ] && [ "$nplans" -gt 0 ] 2>/dev/null
check "controller solved the MILP" $? "${nplans:-0} plans"

$PY - "$HADIS/logs" <<'PY'
import csv, os, sys
p = os.path.join(sys.argv[1], 'slo_timeouts_per_second.csv')
if not os.path.isfile(p):
    sys.exit(1)
rows = list(csv.DictReader(open(p)))
t = {k: sum(int(r[k]) for r in rows) for k in ('succeed', 'timeout', 'drop', 'total')}
served = t['succeed'] + t['timeout'] + t['drop']
print(f"         queries: offered {t['total']}, completed {t['succeed']}, "
      f"timeout {t['timeout']}, dropped {t['drop']}")
sys.exit(0 if t['succeed'] > 0 else 1)
PY
check "queries completed end to end" $?

echo
echo "==============================================================="
if [ "$fail" -eq 0 ]; then
    echo " PASS: $pass checks passed in ${elapsed}s. This machine can run E1."
    echo " Next: Experiments.md section 3 for the full sweep."
    echo "==============================================================="
    exit 0
else
    echo " FAIL: $fail of $((pass + fail)) checks failed. See above."
    echo " The run's logs are left in hadis/logs/ so you can read them."
    echo "==============================================================="
    exit 1
fi
