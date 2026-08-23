"""Compare our metadata mirror with the copy fdroiddata actually serves.

Until the merge request landed, the authority was our fork branch and a strict
byte-diff was right. Since then `AutoUpdateMode: Version` has been in charge:
F-Droid's own bot appends a build entry to fdroiddata *master* whenever it sees
a new tag, and our fork branch stopped moving the day the MR merged. Diffing
against the fork therefore fails for ever and says nothing.

The relation that matters is not equality. This file is allowed to run *ahead*
of what is published — a release is prepared here before its tag is pushed, and
the bot cannot know about it yet. What must never happen is the reverse: if
fdroiddata describes a build this file does not, then F-Droid is building
something we are not tracking, and the drift guards here are blind to it.

Usage:
    python tools/fdroid_compare.py OURS.yml PUBLISHED.yml "what published is"
"""

from __future__ import annotations

import sys


def builds(meta: dict) -> dict[str, tuple]:
    """versionName -> (versionCode, commit) for every build entry."""
    return {
        str(b.get("versionName")): (b.get("versionCode"), str(b.get("commit", "")))
        for b in (meta.get("Builds") or [])
    }


def compare(ours: dict, published: dict, what: str) -> tuple[list[str], list[str]]:
    """Returns (problems, notes). Being ahead is a note; being behind is a problem."""
    mine, theirs = builds(ours), builds(published)
    problems, notes = [], []

    for name, spec in theirs.items():
        if name not in mine:
            problems.append(f"{what} builds {name}, which this file does not describe")
        elif mine[name] != spec:
            problems.append(
                f"{name}: {what} has versionCode/commit {spec}, this file has {mine[name]}")

    ahead = sorted(set(mine) - set(theirs))
    if ahead:
        notes.append("prepared here but not yet published: " + ", ".join(ahead))

    their_code, our_code = published.get("CurrentVersionCode"), ours.get("CurrentVersionCode")
    if their_code is not None and our_code is not None and their_code > our_code:
        problems.append(
            f"{what} is at versionCode {their_code}, ahead of this file's {our_code}")
    return problems, notes


def main() -> int:
    # Imported here, not at module scope: ruamel lives in the checker's own
    # venv, and the comparison above is the part worth testing from the
    # project environment, which has no reason to carry a YAML parser.
    from ruamel.yaml import YAML

    mine_path, theirs_path, what = sys.argv[1:4]
    load = YAML(typ="safe").load
    problems, notes = compare(load(open(mine_path, encoding="utf-8")),
                              load(open(theirs_path, encoding="utf-8")), what)
    for note in notes:
        print(f"  note: {note}")
    for problem in problems:
        print(f"  {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
