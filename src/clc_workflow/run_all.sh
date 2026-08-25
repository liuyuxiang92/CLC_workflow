#!/usr/bin/env bash
#
# Run the whole pipeline, stages 1-9, unattended.
#
#     clc run config.yaml
#     nohup clc run config.yaml > run_all.out 2>&1 &          # survives logout
#
# Invoked by `clc run`, which sets $CLC to the clc entry point.  Runs in YOUR working
# directory: logs and any relative output land where you started it, not next to the
# installed package.
#
# Nothing here polls for jobs.  dpdispatcher's run_submission() BLOCKS until every job
# of that stage has finished and its backward_files have been downloaded, so stages 3,
# 5 and 7 return only when their results are already on disk.  "Wait for the cluster"
# is therefore just "wait for the command to exit".
#
# Every stage selects its work as "has my input, lacks my output", so this script is
# safe to re-run at any time: finished structures are skipped, only the gaps are redone.
# If it dies overnight, run the same command again.
#
# Options
#   --from N --to N   run only stages N..M   (1 2 3 4 5 6 6.5 7 8 9)
#   --dry-run         resolve and print stages 1/3/5/7 without submitting anything
#   --retries N       attempts per remote stage (default 3); see "retrying" below
#   --retry-wait S    pause before retrying a failed submission (default 60s)
#   --min-frac F      abort if a remote stage ends below this completion (default 0.5)
#   --bob             also run stage 6.5 (structural check); off by default
#   --delta-xlsx F    also run the optional `clc delta` step; repeat the flag to
#                     stack several spreadsheets into one dataset
#   --logdir DIR      per-stage logs (default ./pipeline_logs, in your cwd)
#   --notify CMD      run CMD at the end; the summary is on its stdin
#
# Retrying tells two failures apart, because they want opposite answers:
#   * the SUBMISSION errored (non-zero exit: credentials, network, a full queue).
#     Often transient -> wait --retry-wait and try again, up to --retries.
#   * the submission was fine but structures are still outstanding.  Those jobs ran and
#     those structures failed; if none of them moved on an attempt, another queue wait
#     buys nothing -> stop retrying and continue with what worked.  Stages 8 and 9 are
#     built for an incomplete tree (they average over the sets that succeeded).
# --min-frac is the backstop: below it, something systematic is wrong and the script
# stops rather than spending hours analysing nothing.
#
set -uo pipefail

# `clc run` exports CLC; the fallback lets the script still be run by path.
CLC="${CLC:-clc}"
CONFIG=""; FROM="1"; TO="9"; DRY=""; RETRIES=3; MIN_FRAC=0.5
RUN_BOB=0; BOB_PLOT=""; DELTA_XLSX=(); LOGDIR=""; NOTIFY=""; RETRY_WAIT=60
PY="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)       FROM="$2"; shift 2 ;;
    --to)         TO="$2"; shift 2 ;;
    --dry-run)    DRY="--dry-run"; shift ;;
    --retries)    RETRIES="$2"; shift 2 ;;
    --retry-wait) RETRY_WAIT="$2"; shift 2 ;;
    --min-frac)   MIN_FRAC="$2"; shift 2 ;;
    --bob)        RUN_BOB=1; shift ;;
    --bob-plot)   RUN_BOB=1; BOB_PLOT="--plot"; shift ;;
    --delta-xlsx) DELTA_XLSX+=("$2"); shift 2 ;;
    --logdir)     LOGDIR="$2"; shift 2 ;;
    --notify)     NOTIFY="$2"; shift 2 ;;
    # print the leading comment block, however long it grows
    -h|--help)    awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' \
                      "${BASH_SOURCE[0]}"; exit 0 ;;
    -*)           echo "unknown option: $1" >&2; exit 2 ;;
    *)            CONFIG="$1"; shift ;;
  esac
done
CONFIG="${CONFIG:-config.yaml}"
[[ -f "$CONFIG" ]] || { echo "[ERROR] no such config: $CONFIG" >&2; exit 2; }
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
LOGDIR="${LOGDIR:-$PWD/pipeline_logs}"
mkdir -p "$LOGDIR"

STARTED_AT="$(date +%Y%m%d_%H%M%S)"
SUMMARY="$LOGDIR/summary_${STARTED_AT}.txt"

# output_root is resolved by the same loader the drivers use, so a relative path in the
# config means here exactly what it means to every stage -- relative to config.yaml, not
# to this script and not to your cwd.
ROOT="$("$PY" - "$CONFIG" <<'EOF'
import sys
from clc_workflow.clc_config import load_config
print(load_config(sys.argv[1])["output_root"])
EOF
)"
[[ -n "$ROOT" ]] || { echo "[ERROR] could not read output_root from $CONFIG" >&2; exit 2; }

say() { printf '%s\n' "$*" | tee -a "$SUMMARY"; }
rule() { say "==============================================================================="; }

# stage ordering, including the fractional one
stage_num() { case "$1" in 6.5) echo 65 ;; *) echo "$(( ${1%%.*} * 10 ))" ;; esac; }
in_range() {
  local n; n=$(stage_num "$1")
  [[ $n -ge $(stage_num "$FROM") && $n -le $(stage_num "$TO") ]]
}

# structures still needing a stage, per status.py's own accounting
pending() {
  "$CLC" status "$CONFIG" --pending "$1" 2>/dev/null | grep -c . || true
}
total_rows() {
  [[ -f "$ROOT/manifest.csv" ]] && echo $(( $(grep -c . "$ROOT/manifest.csv") - 1 )) || echo 0
}

FAILED_STAGE=""

# run_local <label> <logfile-tag> <command...>
run_local() {
  local label="$1" tag="$2"; shift 2
  local log="$LOGDIR/${tag}_${STARTED_AT}.log"
  say ""
  rule
  say "[stage $label] $(date '+%F %T')"
  say "  \$ $*"
  say "  log: $log"
  "$@" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    say "  [FAILED] exit $rc -- see $log"
    FAILED_STAGE="$label"
    return 1
  fi
  say "  [ok]"
  return 0
}

# run_remote <label> <submit-stage> <status-label>
# Blocks in dpdispatcher until the jobs finish and their files come back, then retries
# while the outstanding count is still falling.
run_remote() {
  local label="$1" sub="$2" st="$3"
  local log="$LOGDIR/stage${label}_${sub}_${STARTED_AT}.log"
  local total before after attempt=1
  total=$(total_rows)
  before=$(pending "$st")

  say ""
  rule
  say "[stage $label] submit.py $sub   $(date '+%F %T')"
  say "  outstanding at start: $before / $total"
  say "  log: $log"
  if [[ -n "$DRY" ]]; then
    "$CLC" submit "$sub" "$CONFIG" --dry-run 2>&1 | tee -a "$log"
    say "  [dry-run] nothing submitted"
    return 0
  fi
  if [[ "$before" -eq 0 ]]; then
    say "  [ok] nothing outstanding, skipping"
    return 0
  fi

  while :; do
    say "  attempt $attempt/$RETRIES -- submitting; this blocks until the jobs finish"
    "$CLC" submit "$sub" "$CONFIG" 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    after=$(pending "$st")
    say "  attempt $attempt done (exit $rc): outstanding $before -> $after / $total"

    [[ "$after" -eq 0 ]] && { say "  [ok] every structure finished"; return 0; }
    if [[ $attempt -ge $RETRIES ]]; then
      say "  [warn] out of attempts, $after structure(s) never finished"
      break
    fi
    # Two different failures, two different answers.  A non-zero exit means the
    # SUBMISSION broke -- credentials, network, a full queue -- and that is often
    # transient, so it is worth another go after a pause.  A clean exit with
    # structures still outstanding means those jobs ran and their structures failed;
    # if none of them moved this attempt, another queue wait buys nothing.
    if [[ $rc -eq 0 && "$after" -ge "$before" ]]; then
      say "  [warn] submission succeeded but no structure moved -- these $after are"
      say "         failing on their own merits, not transiently; not retrying"
      break
    fi
    if [[ $rc -ne 0 ]]; then
      say "  [warn] submission itself failed (exit $rc); retrying in ${RETRY_WAIT}s"
      sleep "$RETRY_WAIT"
    fi
    before="$after"; attempt=$(( attempt + 1 ))
  done

  local done_n=$(( total - after ))
  local ok
  ok=$("$PY" -c "print(1 if $total and $done_n/$total >= $MIN_FRAC else 0)")
  if [[ "$ok" != "1" ]]; then
    say "  [FAILED] only $done_n/$total finished, below --min-frac $MIN_FRAC."
    say "           Something systematic is wrong (image, model path, GPU, quota) --"
    say "           stopping rather than building an analysis on this. See $log."
    FAILED_STAGE="$label"
    return 1
  fi
  say "  [ok] continuing with $done_n/$total"
  return 0
}

rule
say "CLC pipeline -- stages $FROM..$TO${DRY:+  (DRY RUN)}"
say "  config      : $CONFIG"
say "  output_root : $ROOT"
say "  started     : $(date '+%F %T')"
say "  logs        : $LOGDIR"
rule

# A function, not a { } group or a ( ) subshell: `return` here must stop the pipeline
# without killing the script (the summary below still has to run), and FAILED_STAGE must
# survive to be printed -- a subshell would lose it.
run_pipeline() {
  in_range 1   && { run_local 1 stage1_gen_sqs      "$CLC" sqs "$CONFIG" $DRY || return 1; }
  if [[ -n "$DRY" ]] && in_range 1; then
    say ""
    say "[dry-run] stage 1 wrote nothing, so there is no manifest for the later stages"
    say "          to read.  Drop --dry-run, or use --from 2 on an existing tree."
    return 0
  fi
  in_range 2   && { run_local 2 stage2_gen_md       "$CLC" md "$CONFIG" || return 1; }
  in_range 3   && { run_remote 3 md md              || return 1; }
  in_range 4   && { run_local 4 stage4_collect_md   "$CLC" collect-md "$CONFIG" || return 1; }
  in_range 5   && { run_remote 5 opt opt            || return 1; }
  in_range 6   && { run_local 6 stage6_collect_opt  "$CLC" collect-opt "$CONFIG" || return 1; }

  # 6.5 is a report, not a gate: a bad B-O-B distribution is worth seeing but is not a
  # reason to stop, so its exit status is deliberately ignored.
  if [[ $RUN_BOB -eq 1 ]] && in_range 6.5; then
    run_local 6.5 stage6h_check_bob "$CLC" check-bob "$CONFIG" $BOB_PLOT || true
  fi
  if [[ ${#DELTA_XLSX[@]} -gt 0 ]]; then
    run_local delta stage_delta_dataset \
      "$CLC" delta "$CONFIG" --xlsx "${DELTA_XLSX[@]}" || true
  fi

  in_range 7   && { run_remote 7 phonon phonon      || return 1; }
  in_range 8   && { run_local 8 stage8_collect      "$CLC" collect "$CONFIG" || return 1; }
  in_range 9   && { run_local 9 stage9_plots \
                      "$CLC" plot \
                      --parent "$ROOT" --series-glob 'set_*' || return 1; }
  return 0
}

run_pipeline
PIPE_RC=$?

say ""
rule
if [[ $PIPE_RC -eq 0 ]]; then
  say "PIPELINE COMPLETE   $(date '+%F %T')"
else
  say "PIPELINE STOPPED at stage ${FAILED_STAGE:-?}   $(date '+%F %T')"
  say "  Fix the cause and run the same command again -- finished structures are"
  say "  skipped, so it picks up exactly where it stopped."
fi
rule
say ""
say "Final state:"
"$CLC" status "$CONFIG" 2>&1 | tee -a "$SUMMARY"

if [[ $PIPE_RC -eq 0 && -z "$DRY" ]]; then
  say ""
  say "Results:"
  for f in "$ROOT/results.csv" "$ROOT/results_avg.csv" \
           "$ROOT/_state_bavg_regci/dG0_vacancy_summary.csv"; do
    [[ -f "$f" ]] && say "  $f  ($(( $(grep -c . "$f") - 1 )) rows)"
  done
  [[ -d "$ROOT/_state_bavg_regci/line_plots" ]] && \
    say "  $ROOT/_state_bavg_regci/line_plots/  ($(find "$ROOT/_state_bavg_regci/line_plots" -name '*.png' | grep -c .) plots)"
  if [[ -f "$ROOT/_state_bavg_regci/dG0_vacancy_summary.csv" ]]; then
    say ""
    say "dG0_vacancy_summary.csv:"
    column -s, -t "$ROOT/_state_bavg_regci/dG0_vacancy_summary.csv" 2>/dev/null \
      | head -20 | tee -a "$SUMMARY"
  fi
fi

say ""
say "summary: $SUMMARY"
[[ -n "$NOTIFY" ]] && $NOTIFY < "$SUMMARY"
exit $PIPE_RC
