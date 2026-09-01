#!/bin/bash
# Stop every HADIS process on THIS node. Run it on each node that ran workers.
#
# Matching on the command line text (`pkill -f worker.py`) is unsafe here for
# two reasons, so this script does neither of the obvious things:
#
#   1. -f matches the whole command line, so any shell that merely *mentions* a
#      pattern -- a grep, an editor, the terminal that launched a worker -- is
#      killed too. Defence: a process is a target only if it really is
#      "<python> <script>.py ...", checked against argv[0] and argv[1] in
#      /proc/<pid>/cmdline. Text in a shell's arguments can never match that.
#      This script and its ancestors are also excluded, belt and braces.
#
#   2. Each worker runs its model in a torch.multiprocessing child whose command
#      line is "python -c from multiprocessing.spawn import spawn_main ...", so
#      it contains no script name and no pattern can ever match it. That child
#      can outlive its parent and hold GPU memory. Defence: every matched
#      process is killed together with its descendants.

SESSIONS=("${SESSION:-hadis-workers}" "hadis-workers-re" "${HEAD_SESSION:-hadis-head}" "hadis-e1")
SCRIPTS=(controller.py load_balancer.py worker.py worker_re.py client.py)

# This script, the shell that ran it, that shell's parent, ...
protected=""
pid=$$
while [ -n "$pid" ] && [ "$pid" -gt 1 ] 2>/dev/null; do
    protected="$protected $pid"
    pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
done

is_protected() {
    case " $protected " in *" $1 "*) return 0 ;; esac
    return 1
}

# True only for a real "<python> <one of SCRIPTS> ..." process.
is_hadis_process() {
    local pid=$1 argv0 argv1 script
    [ -r "/proc/$pid/cmdline" ] || return 1
    argv0=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | sed -n 1p)
    argv1=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | sed -n 2p)
    [ -n "$argv1" ] || return 1
    # ${x##*/} rather than basename: argv[1] is "-c" for the spawned model
    # executor, which basename would take as an option.
    case "${argv0##*/}" in python*) ;; *) return 1 ;; esac
    for script in "${SCRIPTS[@]}"; do
        [ "${argv1##*/}" = "$script" ] && return 0
    done
    return 1
}

# Every descendant of a pid, depth first (catches the spawned model executor).
descendants() {
    local child
    for child in $(pgrep -P "$1" 2>/dev/null); do
        echo "$child"
        descendants "$child"
    done
}

targets() {
    local pid
    for pid in $(ps -u "$USER" -o pid= 2>/dev/null); do
        is_protected "$pid" && continue
        if is_hadis_process "$pid"; then
            echo "$pid"
            descendants "$pid"
        fi
    done | sort -un
}

for s in "${SESSIONS[@]}"; do
    tmux kill-session -t "$s" 2>/dev/null && echo "Killed tmux session '$s'"
done
sleep 1

pids=$(targets)
if [ -n "$pids" ]; then
    echo "Stopping: $(echo $pids | tr '\n' ' ')"
    kill $pids 2>/dev/null
    sleep 2
    remaining=$(targets)
    if [ -n "$remaining" ]; then
        kill -9 $remaining 2>/dev/null
        sleep 1
    fi
fi

echo "Remaining HADIS processes on $(hostname) (should be empty):"
left=$(targets)
if [ -n "$left" ]; then
    ps -o pid=,args= -p "$(echo $left | tr ' ' ',')" 2>/dev/null | cut -c1-120
else
    echo "  none"
fi
