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

./gradlew :app:installDebug "$@"

sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$(sed -n 's/^sdk\.dir=//p' local.properties)}}"
adb="$sdk/platform-tools/adb"
if [ ! -x "$adb" ]; then
    echo "installed, but no adb at $adb — start it yourself" >&2
    exit 0
fi
# -S stops it first, so a relaunch shows the new code rather than the old
# process resumed from the recents list.
"$adb" shell am start -S -n "$APP_ID/.MainActivity"
