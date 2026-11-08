#!/bin/bash
# Replace the closed ORCD partition (old name: sched_mit_ + mvogelsb) by orcd_ug_pi_mki_mvogelsb_all (ORCD, 2026-11)
# in every text file under the given directories (default: .), skipping .git and logs/. Dry run by default: lists
# file:line.
#   ./switch_partition.sh                      # list matches in this repo
#   ./switch_partition.sh ~ /orcd/data/mvogelsb/004/chcho     # list matches elsewhere too
#   ./switch_partition.sh --apply DIR ...      # replace in place (each changed file is listed)
# Only the exact old name is changed (whole word: a longer name that starts with it is left alone). The old name is
# split in this file so the script does not match itself.

OLD="sched_mit_""mvogelsb"
NEW=orcd_ug_pi_mki_mvogelsb_all
APPLY=0
if [ "$1" = "--apply" ]; then APPLY=1; shift; fi
[ $# -eq 0 ] && set -- .

for d in "$@"; do
    grep -rIlw --exclude-dir=.git --exclude-dir=logs "$OLD" "$d" 2>/dev/null
done | sort -u | while read -r f; do
    if [ $APPLY -eq 1 ]; then
        perl -pi -e "s/\b$OLD\b/$NEW/g" "$f" && echo "changed: $f"
    else
        grep -nw "$OLD" "$f" | sed "s|^|$f:|"
    fi
done
if [ $APPLY -eq 0 ]; then echo "(dry run; add --apply to replace)"; fi
