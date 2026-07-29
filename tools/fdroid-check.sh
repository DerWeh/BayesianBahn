#!/usr/bin/env bash
# Run the checks that fdroiddata's merge-request pipeline runs, against our own
# copy of the metadata file, so problems surface here instead of as a red
# pipeline on the F-Droid MR.
#
# Mirrors these jobs from https://gitlab.com/fdroid/fdroiddata/-/blob/master/.gitlab-ci.yml
#   fdroid rewritemeta  -> the file must already be in canonical form
#   fdroid lint         -> metadata must be valid against fdroiddata's registries
#   checkupdates        -> AutoName/UpdateCheckMode/Binaries must resolve
#
# Usage:
#   tools/fdroid-check.sh [--fix] [--checkupdates] [--refresh]
#
#   --fix           rewrite the metadata into canonical form instead of failing
#   --checkupdates  also run `fdroid checkupdates` (network, clones the app repo)
#   --refresh       ignore the download cache
#
# Set FDROID_SYSTEM_DEPS=1 when fdroidserver's dependencies are already
# installed system-wide (that is what the GitHub workflow does inside the same
# debian:trixie-slim image fdroiddata uses); otherwise a cached venv is built.
set -euo pipefail

APPID=io.github.derweh.bayesianbahn

# fdroiddata's CI runs on debian:trixie-slim, and ruamel.yaml — not fdroidserver
# itself — decides where over-long lines get folded. 0.18.x folds the Binaries
# line, 0.17.x does not, so pip must be pinned to trixie's version or this check
# disagrees with the pipeline it is supposed to predict.
RUAMEL_PIN=0.18.10
CACHE_TTL_HOURS=24

fix=0
run_checkupdates=0
refresh=0
while [ $# -gt 0 ]; do
    case "$1" in
        --fix) fix=1 ;;
        --checkupdates) run_checkupdates=1 ;;
        --refresh) refresh=1 ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# Derived from the script location, not from git: the workflow checks out into
# a container that may not have git available when this runs.
root=$(cd "$(dirname "$0")/.." && pwd)
meta=$root/fdroid/$APPID.yml
[ -f "$meta" ] || { echo "no metadata file at $meta" >&2; exit 1; }

cache=${FDROID_CHECK_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/bayesianbahn-fdroid-check}
mkdir -p "$cache"

# True when $1 is missing, older than the TTL, or --refresh was passed.
stale() {
    [ "$refresh" = 1 ] && return 0
    [ -e "$1" ] || return 0
    [ -n "$(find "$1" -maxdepth 0 -mmin "+$((CACHE_TTL_HOURS * 60))")" ]
}

master=$cache/fdroidserver
if stale "$master"; then
    echo "==> downloading fdroidserver master"
    rm -rf "$master"; mkdir -p "$master"
    curl -fsSL https://gitlab.com/fdroid/fdroidserver/-/archive/master/fdroidserver-master.tar.gz \
        | tar -xz -C "$master" --strip-components=1
    touch "$master"
fi

# lint validates Categories/AntiFeatures against fdroiddata's registries; without
# them every value looks invalid. `?path=config` keeps this to ~2 MB instead of
# cloning the whole of fdroiddata.
cfg=$cache/fdroiddata
if stale "$cfg"; then
    echo "==> downloading fdroiddata config"
    rm -rf "$cfg"; mkdir -p "$cfg"
    curl -fsSL "https://gitlab.com/fdroid/fdroiddata/-/archive/master/fdroiddata-master.tar.gz?path=config" \
        | tar -xz -C "$cfg" --strip-components=1
    curl -fsSL https://gitlab.com/fdroid/fdroiddata/-/raw/master/config.yml -o "$cfg/config.yml"
    touch "$cfg"
fi

if [ "${FDROID_SYSTEM_DEPS:-0}" != 1 ]; then
    venv=$cache/venv
    # The venv only supplies fdroidserver's *dependencies*; the code itself comes
    # from the master checkout via PYTHONPATH, exactly as fdroiddata's CI does it.
    if [ "$(cat "$venv/.pin" 2>/dev/null || true)" != "$RUAMEL_PIN" ]; then
        echo "==> building dependency venv (one-off, takes a few minutes)"
        rm -rf "$venv"
        python3 -m venv "$venv"
        "$venv/bin/pip" -q install fdroidserver "ruamel.yaml==$RUAMEL_PIN"
        printf '%s' "$RUAMEL_PIN" > "$venv/.pin"
    fi
    PATH="$venv/bin:$PATH"
fi
export PATH="$master:$PATH"
export PYTHONPATH="$master:$master/examples"
export PYTHONUNBUFFERED=true
export serverwebroot=/tmp  # fdroiddata's config.yml expands this env var

# A miniature fdroiddata checkout. It has to be a *clean* git repo: rewritemeta
# is judged by `git diff`, and checkupdates refuses to run otherwise — so logs
# and other scratch files live in $work, outside the tree in $stage.
work=$(mktemp -d)
stage=$work/fdroiddata
trap 'rm -rf "$work"' EXIT
mkdir -p "$stage/metadata"
cp "$meta" "$stage/metadata/"
cp "$cfg/config.yml" "$stage/config.yml"
cp -r "$cfg/config" "$stage/config"
chmod 0600 "$stage/config.yml" "$stage"/config/*.yml 2>/dev/null || true
git -C "$stage" init -q
git -C "$stage" add -A
git -C "$stage" -c user.email=none@localhost -c user.name=check commit -qm baseline

status=0
fail() { echo "FAIL: $*"; status=1; }
warn() { echo "warning: $*"; }

# Runs fdroid in the staging dir and returns *its* exit code, while hiding the
# harmless "apksigner not found" warning. Filtering through a pipe instead would
# report grep's status, which is 1 whenever the filter removes everything.
run_fdroid() {
    local rc=0
    ( cd "$stage" && fdroid "$@" ) >"$work/fdroid.log" 2>&1 || rc=$?
    grep -v 'apksigner not found' "$work/fdroid.log" || true
    return "$rc"
}

# lint leaves a `repo/` scratch dir behind and checkupdates clones into `build/`.
# checkupdates refuses to run on a dirty tree, so tidy up between stages.
reset_stage() {
    git -C "$stage" checkout -q -- .
    git -C "$stage" clean -qfd
}

# --- line endings -----------------------------------------------------------
# A single CR makes the pipeline's anchored greps miss and leaves rewritemeta
# with a permanently non-empty diff, with no useful error message.
if grep -q $'\r' "$meta"; then
    fail "$meta contains CR characters; F-Droid metadata must be LF-only"
fi

# --- fdroid rewritemeta -----------------------------------------------------
echo "==> fdroid rewritemeta"
run_fdroid rewritemeta "$APPID" || fail "fdroid rewritemeta errored"
if ! diff -q "$meta" "$stage/metadata/$APPID.yml" >/dev/null; then
    if [ "$fix" = 1 ]; then
        cp "$stage/metadata/$APPID.yml" "$meta"
        echo "fixed: rewrote $meta into canonical form"
    else
        fail "'fdroid rewritemeta' would change the file (trailing whitespace shown in red):"
        git --no-pager diff --no-index --ws-error-highlight=all \
            "$meta" "$stage/metadata/$APPID.yml" || true
        echo "       run '$0 --fix' to apply this"
    fi
fi
reset_stage

# --- fdroid lint ------------------------------------------------------------
echo "==> fdroid lint"
run_fdroid lint "$APPID" || fail "fdroid lint reported problems"

# --- Binaries URL -----------------------------------------------------------
# Reproducible-build verification downloads this exact URL; a typo only shows up
# once F-Droid tries to fetch it. Not fatal here: on a version bump the release
# asset legitimately does not exist yet.

# Reads a scalar whether it sits on the key's line or, once rewritemeta folds an
# over-long line, on the indented line below it.
read_field() {
    awk -v key="$1:" '
        $1 == key {
            if (NF > 1) { print $2; exit }
            getline; sub(/^[[:space:]]+/, ""); print; exit
        }' "$meta"
}
binaries=$(read_field Binaries)
version=$(read_field CurrentVersion)
if [ -n "$binaries" ] && [ -n "$version" ]; then
    url=${binaries//%v/$version}
    echo "==> checking Binaries URL for $version"
    code=$(curl -sSL -o /dev/null -w '%{http_code}' --max-time 60 "$url" || echo 000)
    [ "$code" = 200 ] || warn "Binaries URL returned HTTP $code: $url"
fi

# --- checkupdates -----------------------------------------------------------
if [ "$run_checkupdates" = 1 ]; then
    echo "==> fdroid checkupdates"
    reset_stage
    if ! run_fdroid checkupdates --auto "$APPID"; then
        fail "fdroid checkupdates failed (AutoName / UpdateCheckMode / Repo problem)"
    elif ! git -C "$stage" diff --quiet; then
        # Only a version bump; F-Droid's own bot would do this for us.
        warn "checkupdates found a newer version than the metadata records:"
        git -C "$stage" --no-pager diff
    fi
fi

if [ "$status" = 0 ]; then
    echo "==> all F-Droid metadata checks passed"
fi
exit "$status"
