#!/usr/bin/env bash
#
# deploy.sh — push this directory's sidecar onto the airlens-e2 VM. Idempotent.
#
# The sidecar ran un-source-controlled on the VM until 2026-09-02; this repo
# copy is now the single source of truth and the VM is a deploy target, not an
# editing surface. Edit here, deploy with this script — never the reverse.
# (Recovered from the live VM with per-file sha256 parity; see README.md.)
#
# Secrets are NOT deployed: /etc/airlens/{hf_token,openaq_api_key,cf_r2_chatlog,gh_dispatch_pat}
# live only on the VM (systemd LoadCredential reads them). This script never
# touches them.
#
# Usage: E2_HOST=user@host E2_SSH_KEY=~/.ssh/<key> ./deploy.sh   (or pass user@host as $1)
# Host and key stay out of this public repo — see the private org runbook.
set -euo pipefail

HOST="${1:-${E2_HOST:?set E2_HOST=user@host (Tailscale) or pass it as \$1}}"
KEY="${E2_SSH_KEY:?set E2_SSH_KEY to the VM ssh private key path}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
SSH=(ssh -i "$KEY" "$HOST")
SCP=(scp -q -i "$KEY")

STAGE="/tmp/e2-sidecar-deploy.$$"
"${SSH[@]}" "mkdir -p $STAGE"
trap '"${SSH[@]}" "rm -rf $STAGE" || true' EXIT

"${SCP[@]}" -r "$HERE/bin" "$HERE/sidecar" "$HERE/systemd" "$HOST:$STAGE/"
# The publisher and its contracts ship from this repo too — the VM no longer
# needs a clone of any repo. hf_publish.py finds contracts/ at parents[2], so
# the lib/ tree mirrors the repo layout (lib/scripts/etl, lib/contracts).
"${SSH[@]}" "mkdir -p $STAGE/lib/scripts/etl $STAGE/lib/contracts"
"${SCP[@]}" "$REPO_ROOT/scripts/etl/hf_publish.py" "$HOST:$STAGE/lib/scripts/etl/"
"${SCP[@]}" "$REPO_ROOT"/contracts/validate.py "$REPO_ROOT"/contracts/*.schema.json "$HOST:$STAGE/lib/contracts/"

"${SSH[@]}" "set -euo pipefail
  # lib/ first: the new bin/ scripts call it, and a timer may fire mid-deploy.
  sudo install -d -o ubuntu -g ubuntu /opt/airlens/lib/scripts/etl /opt/airlens/lib/contracts
  sudo install -m 0644 -o ubuntu -g ubuntu $STAGE/lib/scripts/etl/hf_publish.py /opt/airlens/lib/scripts/etl/
  sudo install -m 0644 -o ubuntu -g ubuntu $STAGE/lib/contracts/* /opt/airlens/lib/contracts/
  sudo install -m 0755 -o root -g root $STAGE/bin/*.sh /usr/local/bin/
  sudo install -d -o ubuntu -g ubuntu /opt/airlens/sidecar
  sudo install -m 0644 -o ubuntu -g ubuntu $STAGE/sidecar/* /opt/airlens/sidecar/
  sudo install -m 0644 -o root -g root $STAGE/systemd/* /etc/systemd/system/
  sudo install -d -o ubuntu -g ubuntu -m 0700 /var/lib/airlens/chatlog
  sudo systemctl daemon-reload
  sudo systemctl enable --now airlens-openaq-shadow.timer airlens-sc-shadow.timer
  # The chat-log timer only starts once its credential exists — enabling it
  # without one would just fail hourly and bury the real signal in noise.
  if sudo test -r /etc/airlens/cf_r2_chatlog; then
    sudo systemctl enable --now airlens-chatlog-pull.timer
  else
    echo 'NOTE: /etc/airlens/cf_r2_chatlog absent — chatlog-pull timer not enabled.'
  fi
  # Same for the forecast-collect dispatch timer: without its token every run would just fail.
  if sudo test -r /etc/airlens/gh_dispatch_pat; then
    sudo systemctl enable --now airlens-forecast-dispatch.timer
  else
    echo 'NOTE: /etc/airlens/gh_dispatch_pat absent — forecast-dispatch timer not enabled.'
  fi
  systemctl list-timers airlens-openaq-shadow.timer airlens-sc-shadow.timer airlens-chatlog-pull.timer airlens-forecast-dispatch.timer --no-pager
"

echo "deployed. next slots: sc :05 / openaq :15 / chatlog :35 hourly, forecast dispatch :50 every 6h (UTC 00/06/12/18) — e2-shadow-freshness.yml keeps watching the collectors."
