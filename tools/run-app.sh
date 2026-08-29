#!/usr/bin/env bash
# Build the debug APK, install it on the attached device or emulator, and open
# it — the shortest path from an edit to seeing it on screen.
#
# `adb` is not on PATH here and the SDK location is machine-specific, so it is
# read from local.properties (which git does not track) with ANDROID_HOME as an
# override. Gradle finds the SDK the same way, so if the install works the
# launch will too.
set -euo pipefail
cd "$(dirname "$0")/.."

APP_ID=io.github.derweh.bayesianbahn

sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$(sed -n 's/^sdk\.dir=//p' local.properties)}}"
adb="$sdk/platform-tools/adb"

# Gradle's answer to a stopped emulator is "No connected devices!" after it has
# built everything, which says nothing about what to do — and an emulator that
# exited an hour ago looks exactly like one that never started. Check first and
# name the AVDs that exist.
if [ -x "$adb" ]; then
    attached=$("$adb" devices | tail -n +2 | grep -v '^[[:space:]]*$' || true)
    if ! printf '%s\n' "$attached" | grep -qE '[[:space:]]device$'; then
        if [ -n "$attached" ]; then
            # `offline` is what a booting emulator reports, and it is the state
            # you hit by running this straight after starting one. Telling you
            # to start an emulator then would be advice for the wrong problem.
            echo "A device is attached but not ready:" >&2
            printf '%s\n' "$attached" >&2
            echo "If it is still booting, wait for it:" >&2
            echo "  until [ \"\$($adb shell getprop sys.boot_completed | tr -d '\\r')\" = 1 ]; do sleep 2; done" >&2
        else
            echo "No device or emulator is attached." >&2
            avds=$("$sdk/emulator/emulator" -list-avds 2>/dev/null | head -3 | tr '\n' ' ' || true)
            if [ -n "${avds// /}" ]; then
                echo "Start one with:  $sdk/emulator/emulator -avd ${avds%% *} -no-boot-anim &" >&2
                echo "Available AVDs:  $avds" >&2
            else
                echo "No AVDs either — see 'Running it on an emulator' in the README." >&2
            fi
        fi
        exit 1
    fi
fi

./gradlew :app:installDebug "$@"

if [ ! -x "$adb" ]; then
    echo "installed, but no adb at $adb — start it yourself" >&2
    exit 0
fi
# -S stops it first, so a relaunch shows the new code rather than the old
# process resumed from the recents list.
"$adb" shell am start -S -n "$APP_ID/.MainActivity"
