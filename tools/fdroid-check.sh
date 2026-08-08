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
# Plus checks fdroiddata does *not* do, for drift between the metadata and the
# app it describes (wrong build commit, stale version, stale fork copy).
#
# Usage:
#   tools/fdroid-check.sh [--fix] [--checkupdates] [--fork] [--refresh] [--self-test]
#
#   --fix           rewrite the metadata into canonical form instead of failing
#   --checkupdates  also run `fdroid checkupdates` (network, clones the app repo)
#   --fork          also compare against the copy on the fdroiddata fork branch
#   --refresh       ignore the download cache
#   --self-test     assert this script rejects each way the file has broken before
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
FORK_RAW=${FDROID_FORK_RAW:-https://gitlab.com/DerWeh/fdroiddata/-/raw/$APPID/metadata/$APPID.yml}

fix=0
run_checkupdates=0
check_fork=0
refresh=0
self_test=0
while [ $# -gt 0 ]; do
    case "$1" in
        --fix) fix=1 ;;
        --checkupdates) run_checkupdates=1 ;;
        --fork) check_fork=1 ;;
        --refresh) refresh=1 ;;
        --self-test) self_test=1 ;;
        -h|--help) awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# Derived from the script location, not from git: the workflow checks out into
# a container that may not have git available when this runs. --self-test points
# it at a mutated copy of the tree.
root=${FDROID_CHECK_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
meta=$root/fdroid/$APPID.yml
gradle=$root/app/build.gradle.kts
[ -f "$meta" ] || { echo "no metadata file at $meta" >&2; exit 1; }

cache=${FDROID_CHECK_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/bayesianbahn-fdroid-check}
mkdir -p "$cache"

# --- self-test --------------------------------------------------------------
# Every way this file has actually broken on the F-Droid MR, reproduced as a
# negative test. A checker that silently stops checking is worse than no checker
# at all: an earlier version of this script reported success for everything
# after a `grep` in the middle of a pipeline swallowed fdroid's exit code.
if [ "$self_test" = 1 ]; then
    fails=0
    version=$(python3 -c 'import re,sys;print(re.search(r"^CurrentVersion: *(.+)$",open(sys.argv[1]).read(),re.M).group(1).strip())' "$meta")
    for case in unwrapped-binaries stripped-trailing-space crlf \
                tag-instead-of-commit version-drift bad-category \
                gradle-drift wrong-commit; do
        tmp=$(mktemp -d)
        mkdir -p "$tmp/fdroid" "$tmp/app"
        cp "$meta" "$tmp/fdroid/"
        cp "$gradle" "$tmp/app/"
        # Two cases live outside the metadata file: the app moving on without it,
        # and the release tag pointing somewhere other than the pinned commit.
        if [ "$case" = gradle-drift ]; then
            sed -i 's/versionCode = [0-9]*/versionCode = 99/' "$tmp/app/build.gradle.kts"
        elif [ "$case" = wrong-commit ]; then
            git -C "$tmp" init -q
            git -C "$tmp" -c user.email=none@localhost -c user.name=check \
                commit -q --allow-empty -m baseline
            git -C "$tmp" tag "v$version"
        else
        python3 - "$tmp/fdroid/$APPID.yml" "$case" <<'PY'
import re, sys
path, case = sys.argv[1], sys.argv[2]
original = open(path, encoding="utf-8", newline="").read()
text = original
if case == "unwrapped-binaries":          # what fdroiddata's rewritemeta rejected
    text = text.replace("Binaries: \n  ", "Binaries: ")
elif case == "stripped-trailing-space":   # what a trailing-whitespace hook does
    text = text.replace("Binaries: \n", "Binaries:\n")
elif case == "crlf":                      # what a Windows editor or copy-paste does
    text = text.replace("\n", "\r\n")
elif case == "tag-instead-of-commit":     # what the F-Droid reviewer rejected
    text = re.sub(r"commit: [0-9a-f]{40}", "commit: v0.1.1", text)
elif case == "version-drift":             # metadata left behind by a version bump
    text = re.sub(r"^    versionCode: \d+$", "    versionCode: 99", text, flags=re.M)
elif case == "bad-category":              # value not in fdroiddata's registry
    text = re.sub(r"(?m)^(Categories:\n)  - .+$", r"\1  - Teleportation", text)
else:
    raise SystemExit(f"unknown case {case}")
# A mutation that no longer matches would leave the file valid, and the case
# would "pass" while testing nothing — which is how this very case went stale
# when the reviewer changed our category.
if text == original:
    raise SystemExit(f"self-test case {case!r} no longer applies to the metadata")
open(path, "w", encoding="utf-8", newline="").write(text)
PY
        fi
        rc=0
        FDROID_CHECK_ROOT=$tmp FDROID_CHECK_NO_NET=1 "$0" >"$tmp/out" 2>&1 || rc=$?
        if [ "$rc" = 0 ]; then
            echo "SELF-TEST FAIL: '$case' was not detected"
            sed 's/^/    /' "$tmp/out"
            fails=1
        else
            echo "ok: $case is detected"
        fi
        rm -rf "$tmp"
    done
    [ "$fails" = 0 ] && echo "==> self-test passed"
    exit "$fails"
fi

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

# --- consistency with the app -----------------------------------------------
# The metadata restates facts that live in the app: which commit to build, which
# version that is. fdroiddata's pipeline cannot notice when those drift — it has
# no idea what our repo contains — so F-Droid would cheerfully build the wrong
# commit and publish it under the new version number. Parsed as YAML rather than
# grepped, because whether a value sits on the key's line or on a folded
# continuation line is exactly what keeps changing under us.
echo "==> metadata/app consistency"
if ! python3 - "$meta" "$gradle" "$work/fields.env" <<'PY'
import re, shlex, sys
from ruamel.yaml import YAML

meta_path, gradle_path, out_path = sys.argv[1:4]
meta = YAML(typ="safe").load(open(meta_path, encoding="utf-8"))
gradle = open(gradle_path, encoding="utf-8").read()

def gradle_field(name, value):
    m = re.search(rf"\b{name}\s*=\s*{value}", gradle)
    return m.group(1) if m else None

problems = []
builds = meta.get("Builds") or []
build = builds[-1] if builds else {}
if not builds:
    problems.append("no Builds entry")

commit = str(build.get("commit", ""))
# The F-Droid reviewer rejected a tag here: a tag can be repointed after review,
# a hash cannot.
if not re.fullmatch(r"[0-9a-f]{40}", commit):
    problems.append(f"Builds commit must be a full 40-char hash, got {commit!r}")

name, code = str(build.get("versionName", "")), build.get("versionCode")
cur_name, cur_code = str(meta.get("CurrentVersion", "")), meta.get("CurrentVersionCode")
g_name, g_code = gradle_field("versionName", r'"([^"]+)"'), gradle_field("versionCode", r"(\d+)")

if name != cur_name:
    problems.append(f"Builds versionName {name!r} != CurrentVersion {cur_name!r}")
if code != cur_code:
    problems.append(f"Builds versionCode {code!r} != CurrentVersionCode {cur_code!r}")
if g_name is None or g_code is None:
    problems.append(f"could not read versionName/versionCode from {gradle_path}")
else:
    if name != g_name:
        problems.append(f"Builds versionName {name!r} != build.gradle.kts {g_name!r}")
    if code != int(g_code):
        problems.append(f"Builds versionCode {code!r} != build.gradle.kts {g_code!r}")

for p in problems:
    print(f"  {p}")

with open(out_path, "w", encoding="utf-8") as fh:
    for key, value in (
        ("BINARIES_URL", meta.get("Binaries") or ""),
        ("COMMIT", commit),
        ("VERSION", cur_name),
    ):
        fh.write(f"{key}={shlex.quote(str(value))}\n")

sys.exit(1 if problems else 0)
PY
then
    fail "metadata does not match the app it describes"
fi
# shellcheck source=/dev/null
. "$work/fields.env"

# The build commit must be the one the release tag points at, or F-Droid builds
# something the released APK was never built from and reproducibility fails.
if [ -n "$VERSION" ] && [ -n "$COMMIT" ] && command -v git >/dev/null &&
   git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
    if tagged=$(git -C "$root" rev-parse -q --verify "refs/tags/v$VERSION^{commit}"); then
        [ "$tagged" = "$COMMIT" ] ||
            fail "Builds commit $COMMIT is not what tag v$VERSION points at ($tagged)"
    else
        warn "no tag v$VERSION here — cannot verify the build commit (fetch tags?)"
    fi
fi

# --- Binaries URL -----------------------------------------------------------
# Reproducible-build verification downloads this exact URL; a typo only shows up
# once F-Droid tries to fetch it. Not fatal here: on a version bump the release
# asset legitimately does not exist yet.
if [ "${FDROID_CHECK_NO_NET:-0}" != 1 ] && [ -n "$BINARIES_URL" ] && [ -n "$VERSION" ]; then
    url=${BINARIES_URL//%v/$VERSION}
    echo "==> checking Binaries URL for $VERSION"
    code=$(curl -sSL -o /dev/null -w '%{http_code}' --max-time 60 "$url" || echo 000)
    [ "$code" = 200 ] || warn "Binaries URL returned HTTP $code: $url"
fi

# --- fork copy --------------------------------------------------------------
# The file fdroiddata's pipeline actually reads is the one on the fork branch,
# not this one. A red pipeline after a fix here usually means only that the fix
# was never synced across.
if [ "$check_fork" = 1 ]; then
    echo "==> comparing against the fdroiddata fork branch"
    if curl -fsSL --max-time 60 "$FORK_RAW" -o "$work/fork.yml"; then
        if ! diff -q "$meta" "$work/fork.yml" >/dev/null; then
            fail "the fdroiddata fork has a different file (see fdroid/README.md to sync):"
            git --no-pager diff --no-index --ws-error-highlight=all \
                "$work/fork.yml" "$meta" || true
        fi
    else
        warn "could not fetch $FORK_RAW"
    fi
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
