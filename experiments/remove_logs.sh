#!/bin/bash
# Empty hadis/logs/, keeping only .gitkeep.
#
# Usage:
#   experiments/remove_logs.sh          # asks first, listing what will go
#   experiments/remove_logs.sh -y       # no prompt
#
# Manual housekeeping only. Nothing else in the artifact calls this, and nothing
# should: a finished run's logs are results, and experiments/collect_logs.sh is
# how they are filed into results/logs/<name>/. Use this when you want to start
# from a clean slate and are certain the current logs do not matter.
set -u

ART="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ART/hadis/logs"
ASSUME_YES=0
[ "${1:-}" = "-y" ] || [ "${1:-}" = "--yes" ] && ASSUME_YES=1

if [ ! -d "$LOGS" ]; then
    echo "No such directory: $LOGS" >&2
    exit 1
fi

# Everything except .gitkeep, dotfiles included (start_client.sh leaves a marker).
mapfile -t victims < <(find "$LOGS" -mindepth 1 -not -name '.gitkeep' | sort)

if [ "${#victims[@]}" -eq 0 ]; then
    echo "hadis/logs/ is already empty."
    exit 0
fi

echo "This will delete ${#victims[@]} item(s) from hadis/logs/:"
printf '  %s\n' "${victims[@]##*/}" | head -20
[ "${#victims[@]}" -gt 20 ] && echo "  ... and $(( ${#victims[@]} - 20 )) more"

# A finished run that has not been collected is somebody's results.
if ls "$LOGS"/slo_timeouts_per_second.csv >/dev/null 2>&1; then
    echo
    echo "  NOTE: slo_timeouts_per_second.csv is present, so this looks like a"
    echo "        finished run. To keep it instead:"
    echo "            experiments/collect_logs.sh <name>"
fi

if [ "$ASSUME_YES" != "1" ]; then
    echo
    printf "Delete them? [y/N] "
    read -r reply
    case "$reply" in
        y|Y|yes|YES) ;;
        *) echo "Cancelled; nothing was deleted."; exit 0 ;;
    esac
fi

find "$LOGS" -mindepth 1 -not -name '.gitkeep' -delete
remaining=$(find "$LOGS" -mindepth 1 | wc -l)
echo "Removed. hadis/logs/ now holds $remaining item(s) (.gitkeep)."
