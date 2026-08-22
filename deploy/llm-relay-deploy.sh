#!/usr/bin/env bash
# llm-relay-deploy — atomic release deploy for llm-relay.
#
# Install as /usr/local/sbin/llm-relay-deploy (root:root 0755).
# Usage:  llm-relay-deploy [ref]        deploy a git ref (default: main)
#         llm-relay-deploy --rollback   revert to the previous release
#         llm-relay-deploy --list       show releases, newest first
#         llm-relay-deploy --status     current sha, unit state, restarts, health
#
# Requires a bare mirror, owned by the service user, holding the only copy of
# any GitHub token:
#   sudo -u llm git clone --mirror https://<TOKEN>@github.com/<org>/llm-relay.git \
#        /srv/llm/repos/llm-relay.git
# Release dirs are local clones of that mirror and carry no credentials.
#
# The interpreter is a SHARED venv (/srv/llm/venvs/llm-relay) that lives OUTSIDE
# the release: the unit's WorkingDirectory is what decides which release's code
# gets imported. So this script never builds a per-release venv — but it does
# import-check the new tree against that shared venv before the symlink swap,
# because a release that adds a third-party import would otherwise fail only at
# exec time, costing a restart-plus-rollback cycle.
#
# Mirrors llm-portal-deploy. Gateway (llm-gateway-01) only — the cairn-02
# checkout is frozen and is not deployed from.

set -euo pipefail

REPO=llm-relay
OWNER=llm
MIRROR=/srv/llm/repos/${REPO}.git
RELEASES=/srv/llm/releases
CURRENT=/srv/llm/current/${REPO}
SERVICE=llm-relay.service
VENV=/srv/llm/venvs/${REPO}/bin/python
# /health, not /healthz — /healthz 404s on the relay. /health is auth-exempt on
# the trusted loopback listener (18090), so no service key is needed here.
HEALTH_URL=http://127.0.0.1:18090/health
KEEP=5
HEALTH_TIMEOUT=45  # the relay polls backends on boot; allow longer than the portal

as_owner() { sudo -u "$OWNER" "$@"; }
die() { echo "deploy: $*" >&2; exit 1; }

# One deploy at a time. Defined after die() so the guard can actually report.
exec 9>/run/llm-relay-deploy.lock
flock -n 9 || die "deploy already running"

cd /  # find(1) restores initial cwd on exit; avoid inaccessible dirs under sudo

# Atomic symlink swap: ln -sfn is NOT atomic when the target exists, mv -T is.
point_at() {
    as_owner ln -sfn "$1" "${CURRENT}.new"
    as_owner mv -Tf "${CURRENT}.new" "$CURRENT"
}

wait_healthy() {
    local i
    for ((i = 0; i < HEALTH_TIMEOUT; i++)); do
        if curl -fsS -m 2 "$HEALTH_URL" | grep -q '"ok"'; then
            return 0
        fi
        sleep 1
    done
    return 1
}

# Healthy means: unit active, health endpoint answering, no restart since we
# began, and the live process running out of the release we just pointed at.
# The last two matter: a crash loop can answer /health between respawns.
verify() {
    systemctl is-active --quiet "$SERVICE" || return 1
    wait_healthy || return 1
    [[ "$(systemctl show -p NRestarts --value "$SERVICE")" == "0" ]] || return 1
    local pid cwd
    pid=$(systemctl show -p MainPID --value "$SERVICE")
    cwd=$(readlink "/proc/${pid}/cwd" 2>/dev/null || true)
    [[ "$cwd" == "$(readlink -f "$CURRENT")" ]] || {
        echo "deploy: live process cwd ($cwd) is not the current release" >&2
        return 1
    }
}

prune() {
    local keep_current keep_prev
    keep_current=$(readlink -f "$CURRENT")
    keep_prev=${1:-}
    as_owner find "$RELEASES" -maxdepth 1 -name "${REPO}-*" -printf '%T@ %p\n' \
        | sort -rn | cut -d' ' -f2- | tail -n "+$((KEEP + 1))" \
        | grep -vx -e "$keep_current" -e "${keep_prev:-__none__}" \
        | while read -r old; do
            echo "deploy: pruning $old"
            as_owner rm -rf "$old"
        done || true
}

case "${1:-main}" in
--list)
    ls -1dt "${RELEASES}/${REPO}-"* | sed "s|\$| |;s|^$(readlink -f "$CURRENT") |&<- current|"
    exit 0
    ;;
--status)
    SHA=$(as_owner git -C "$(readlink -f "$CURRENT")" rev-parse --short HEAD 2>/dev/null || echo "?")
    echo "current: ${SHA}"
    echo "unit: $(systemctl is-active "$SERVICE")"
    echo "restarts: $(systemctl show -p NRestarts --value "$SERVICE")"
    echo "health: $(curl -fsS -m 2 "$HEALTH_URL" 2>/dev/null || echo unreachable)"
    exit 0
    ;;
--rollback)
    PREV=$(cat /srv/llm/current/.${REPO}.prev 2>/dev/null) || die "no previous release recorded"
    [[ -d "$PREV" ]] || die "previous release $PREV is gone"
    echo "deploy: rolling back to $PREV"
    point_at "$PREV"
    systemctl reset-failed "$SERVICE" || true
    systemctl restart "$SERVICE"
    verify || die "rollback target is also unhealthy — manual intervention required"
    echo "deploy: rolled back to $(basename "$PREV")"
    exit 0
    ;;
esac

REF=${1:-main}
[[ -d "$MIRROR" ]] || die "mirror $MIRROR missing — see header for setup"
[[ -x "$VENV" ]] || die "shared venv interpreter $VENV missing"

# The relay's first release directory on the gateway was placed by hand, so
# $CURRENT may still be a real directory rather than a symlink. mv -T cannot
# replace a directory with a symlink: refuse now, before touching anything.
if [[ -e "$CURRENT" && ! -L "$CURRENT" ]]; then
    die "$CURRENT is not a symlink — move the hand-placed tree under $RELEASES and symlink $CURRENT at it first"
fi

as_owner git -C "$MIRROR" remote update --prune
SHA=$(as_owner git -C "$MIRROR" rev-parse "${REF}^{commit}") || die "unknown ref: $REF"
DEST="${RELEASES}/${REPO}-${SHA}"

# readlink -f prints the path even when the final component does not exist, so
# resolving unconditionally would record a bogus .prev on the first-ever deploy
# (when $CURRENT is not a symlink yet). Only resolve what is actually there.
PREV=""
if [[ -L "$CURRENT" ]]; then
    PREV=$(readlink -f "$CURRENT")
fi

if [[ "$PREV" == "$DEST" ]]; then
    echo "deploy: already at ${SHA} — nothing to do"
    exit 0
fi

# Immutable release dir. Local clone from the mirror hardlinks objects: fast,
# offline, and no token lands in the release.
if [[ ! -d "$DEST" ]]; then
    echo "deploy: creating release ${SHA}"
    install -d -o "$OWNER" -g "$OWNER" -m755 "$DEST"
    as_owner git clone --quiet --no-checkout "$MIRROR" "$DEST"
    as_owner git -C "$DEST" checkout --quiet --detach "$SHA"
fi

# Pre-flight: import the app out of the NEW tree using the shared venv. The
# venv is outside the release, so a release introducing a new third-party
# import would otherwise blow up at exec time. Failing here costs an aborted
# deploy instead of a restart-plus-rollback. cwd is what puts the release on
# sys.path, exactly as the unit's WorkingDirectory does at runtime.
IMPORT_ERR=$(mktemp -t llm-relay-deploy-import.XXXXXX)
trap 'rm -f "$IMPORT_ERR"' EXIT
if ! as_owner env -C "$DEST" "$VENV" -c 'import llm_relay.api.app' 2>"$IMPORT_ERR"; then
    echo "deploy: release ${SHA} does not import under the shared venv" >&2
    cat "$IMPORT_ERR" >&2
    die "install missing dependencies into /srv/llm/venvs/${REPO} first"
fi

# Record what is live so --rollback has a target. Same semantics as the portal
# script: after the auto-rollback below, .prev names the release we rolled back
# TO, so a later explicit --rollback restarts that release rather than stepping
# a second release back.
if [[ -n "$PREV" ]]; then
    echo "$PREV" | as_owner tee "/srv/llm/current/.${REPO}.prev" >/dev/null
fi

point_at "$DEST"
systemctl reset-failed "$SERVICE" || true
systemctl restart "$SERVICE"

if verify; then
    echo "deploy: OK — ${REPO} at ${SHA}"
    prune "$PREV"
    # Self-update: install the deploy script from the release we just deployed.
    if [[ -f "$DEST/deploy/llm-relay-deploy.sh" ]]; then
        if ! cmp -s "$DEST/deploy/llm-relay-deploy.sh" "$0" 2>/dev/null; then
            install -m0755 "$DEST/deploy/llm-relay-deploy.sh" /usr/local/sbin/llm-relay-deploy
            echo "deploy: self-updated deploy script"
        fi
    fi
else
    echo "deploy: FAILED health gate — rolling back" >&2
    journalctl -u "$SERVICE" -n 40 --no-pager >&2 || true
    if [[ -n "$PREV" && -d "$PREV" ]]; then
        point_at "$PREV"
        systemctl reset-failed "$SERVICE" || true
        systemctl restart "$SERVICE"
        verify && echo "deploy: rolled back to $(basename "$PREV")" >&2
    fi
    exit 1
fi
