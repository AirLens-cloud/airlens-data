#!/usr/bin/env bash
#
# dispatch-forecast-collect.sh — start the forecast-collect GitHub workflow through the
# workflow_dispatch API. Run by airlens-forecast-dispatch.timer (every 6h at :50 UTC).
#
# Why: GitHub's own cron for forecast-collect arrives hours late, and band-forecast follows
# it through workflow_run. This VM's timer is the on-time trigger. The workflow keeps its
# schedule as a backup, and its first job skips a second collection inside the same 6-hour
# slot (scripts/etl/forecast_slot_guard.py), so a late backup does not collect twice.
#
# The token (fine-grained PAT, this repo only) arrives through systemd LoadCredential. It
# reaches curl only as a config on stdin (`curl -q -K -`) — never in argv, the environment or
# the log. `-q` has to be the first argument: it keeps curl from reading a ~/.curlrc, which could
# add a trace file or a proxy. Never add `set -x`: it would print the token.
#
# Exit 0 = the dispatch was accepted. Exit 1 = anything else (the unit retries once).
# An answer of 200 carries the new run's ID in its body (204 has no body); the log keeps that
# number as `run <id>` and nothing else from the response.
#
# A refused dispatch makes the script read the workflow's own state (GET). A workflow GitHub switched off after 60 days
# of inactivity (state `disabled_inactivity`) is switched back on and the dispatch is tried once more. Any other state,
# a state it cannot read or a failed lookup leaves the workflow as it is and the run fails saying what the script saw.
# The status and wording of the refusal decide nothing: GitHub's documentation does not describe the answer to a
# dispatch aimed at a disabled workflow.
set -euo pipefail
# The token check below is an ASCII range and the log filter must not depend on the VM's locale.
export LC_ALL=C

REPO="AirLens-cloud/airlens-data"
WORKFLOW="forecast-collect.yml"
REF="main"
API="https://api.github.com/repos/$REPO/actions/workflows/$WORKFLOW"

log() { echo "dispatch-forecast-collect: $*"; }
die() { echo "dispatch-forecast-collect: $*" >&2; exit 1; }

# The credential is checked before anything else touches the network.
[ -n "${CREDENTIALS_DIRECTORY:-}" ] || die "CREDENTIALS_DIRECTORY is not set (run through the systemd unit)"
[ -r "$CREDENTIALS_DIRECTORY/gh_dispatch_pat" ] || die "credential gh_dispatch_pat is not readable"
TOKEN="$(cat "$CREDENTIALS_DIRECTORY/gh_dispatch_pat")"
# Trim surrounding whitespace (a trailing newline is the usual one), then allow [A-Za-z0-9_] only.
# The error never prints the value.
TOKEN="${TOKEN#"${TOKEN%%[![:space:]]*}"}"
TOKEN="${TOKEN%"${TOKEN##*[![:space:]]}"}"
case "$TOKEN" in
  ''|*[!A-Za-z0-9_]*) die "credential gh_dispatch_pat is empty or has characters outside [A-Za-z0-9_]" ;;
esac

BODY="$(mktemp)"
trap 'rm -f "$BODY"' EXIT

# request METHOD URL [JSON] — HTTP status into HTTP_CODE, response body into $BODY.
request() {
  local method="$1" url="$2" data="${3-}" rc=0
  local args=(-sS --max-time 30 -o "$BODY" -w '%{http_code}' -X "$method"
    -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2022-11-28' -A airlens-e2-dispatch)
  if [ -n "$data" ]; then
    args+=(--data "$data")
  elif [ "$method" = PUT ]; then
    args+=(-H 'Content-Length: 0')  # a body-less PUT gets an explicit zero length (some GitHub endpoints answer 411 without it)
  fi
  : > "$BODY"  # an empty response body must not leave the previous request's body behind
  # Only the config on stdin carries the token. printf is a builtin, so it is in no process's argv either.
  HTTP_CODE="$(printf 'header = "Authorization: Bearer %s"\n' "$TOKEN" | curl -q -K - "${args[@]}" "$url")" || rc=$?
  [ "$rc" -eq 0 ] || die "curl failed (exit $rc) for $method $url"
}

dispatch() { request POST "$API/dispatches" "{\"ref\":\"$REF\"}"; }
accepted() { [ "$HTTP_CODE" = 204 ] || [ "$HTTP_CODE" = 200 ]; }

# The run ID of a dispatch that answered 200 (that body carries workflow_run_id; the older 204 has none). Only the digits
# of an integer value reach the log, at most 20 of them. Empty when there is no such number.
run_id() {
  [ "$HTTP_CODE" = 200 ] || return 0
  grep -a -o -E '"workflow_run_id"[[:space:]]*:[[:space:]]*[0-9]{1,20}[[:space:]]*([,}]|$)' "$BODY" | head -n 1 \
    | tr -cd '0-9' || true
}

# dispatched [SUFFIX] — the success line: the HTTP status, and the run ID when the answer had one.
dispatched() {
  local id
  id="$(run_id)"
  log "dispatched $WORKFLOW on $REF$1 (HTTP $HTTP_CODE${id:+, run $id})"
}

# The `state` of the workflow object in $BODY (active, disabled_inactivity, disabled_manually, ...). Only [a-z_] reaches the
# caller, at most 40 characters. Empty when the body has no such value.
workflow_state() {
  grep -a -o -E '"state"[[:space:]]*:[[:space:]]*"[a-z_]{1,40}"' "$BODY" | head -n 1 \
    | sed -E 's/^"state"[[:space:]]*:[[:space:]]*"//; s/"$//' || true
}

# The first "message" of a GitHub error body, for the log: printable ASCII only (so it stays one line), the token masked
# (filtering and masking come before the length cut, so a cut cannot leave half a token), at most 200 characters.
api_message() {
  local msg
  msg="$(grep -a -o -E '"message"[[:space:]]*:[[:space:]]*"([^"\\]|\\.)*"' "$BODY" | head -n 1 \
    | sed -E 's/^"message"[[:space:]]*:[[:space:]]*"//; s/"$//' | tr -cd '[:print:]')" || msg=""
  msg="${msg//"$TOKEN"/***}"
  printf '%s\n' "${msg:0:200}"
}

# fail_http WHAT — log the HTTP status and the API's message, exit 1.
fail_http() {
  local msg
  msg="$(api_message)"
  die "$1 failed: HTTP $HTTP_CODE${msg:+ (message: $msg)}"
}

dispatch
if accepted; then
  dispatched ""
  exit 0
fi

# The dispatch was refused. GitHub's documentation does not say which status and wording a dispatch to a switched-off workflow
# gets, so neither is taken as the sign of one: whatever the refusal was, the workflow's own state decides.
# GitHub turns a workflow off after 60 days without repository activity (state disabled_inactivity), and the workflow's
# schedule backup stops with it: that one is switched back on here and the dispatch is tried once more. A workflow somebody
# disabled on purpose (disabled_manually, for an incident or a rollback) is respected, and so is any other state, an active
# one included (a refusal that raced with a re-enable): the run fails and the unit's retry dispatches again 90 s later.
# The refusal is kept before the lookup because the lookup overwrites $BODY.
refusal_message="$(api_message)"
refused="HTTP $HTTP_CODE${refusal_message:+ (message: $refusal_message)}"
request GET "$API"
if [ "$HTTP_CODE" = 200 ]; then
  state="$(workflow_state)"
  seen="workflow state ${state:-not readable}"
else
  state=""
  seen="state lookup HTTP $HTTP_CODE"
fi
[ "$state" = disabled_inactivity ] || die "dispatch failed: $refused; $seen — not enabling it (only an inactivity disable is undone here)"
log "dispatch refused: $refused; $WORKFLOW is $state — enabling it and retrying the dispatch once"
request PUT "$API/enable"
[ "$HTTP_CODE" = 204 ] || fail_http "enable"
dispatch
if accepted; then
  dispatched " after enabling it"
  exit 0
fi
fail_http "dispatch"
