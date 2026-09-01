#!/bin/bash
# Move a finished run's logs out of hadis/logs into results/logs/<name>/ so the
# next run starts from a clean directory.
#
# Usage:  experiments/collect_logs.sh <name>      e.g.  collect_logs.sh hadis
set -e

NAME="$1"
if [ -z "$NAME" ]; then
    echo "usage: $0 <name>   (e.g. hadis, proteus, diffserve, e2_functional)" >&2
    exit 1
fi

ART="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ART/hadis/logs"
DST="$ART/results/logs/$NAME"

# Ignore dotfiles: .gitkeep, and the marker start_client.sh uses to find its log.
if [ ! -d "$SRC" ] || [ -z "$(ls "$SRC" 2>/dev/null)" ]; then
    echo "ERROR: no logs in $SRC" >&2
    exit 1
fi

if [ -d "$DST" ]; then
    STAMP=$(date +%Y%m%d_%H%M%S)
    echo "$DST exists; moving it aside to ${DST}_$STAMP"
    mv "$DST" "${DST}_$STAMP"
fi

mkdir -p "$DST"
find "$SRC" -maxdepth 1 -type f ! -name '.*' -exec mv {} "$DST/" \;

echo "Collected into $DST:"
for f in slo_timeouts_per_second.csv query_num_per_second.csv cascade_config_per_second.csv; do
    if [ -f "$DST/$f" ]; then
        echo "  $f  ($(($(wc -l < "$DST/$f") - 1)) rows)"
    else
        echo "  $f  MISSING -- the run did not produce results"
    fi
done
