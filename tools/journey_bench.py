"""Measure the journey search's transfer strategy against live IRIS.

The search is a heuristic with three tuning constants, and they were guesses.
This replays the algorithm over many journeys and reports how often it finds a
connection as a function of the attempt budget, which is what setting
MAX_TRANSFER_ATTEMPTS actually needs.

Ground truth is constructed, not assumed: `--generate` builds origin/destination
pairs by following a real train from the origin to a station on its route and a
real connecting train onward from there. A one-transfer connection therefore
provably exists for every query, so a miss is the search's fault and not a
journey that needs two changes. Pairs reachable without changing are excluded,
so the transfer search is what is being measured.

    python tools/journey_bench.py --generate queries.json --count 24
    python tools/journey_bench.py --run queries.json --max-attempts 14

Every IRIS document is cached on disk (plan documents for a past hour never
change), so strategies are compared on identical data and re-runs are free.
Please keep the query count modest: this talks to a keyless public API.

Limitations worth remembering when reading the numbers: it uses planned data
only (no cancellations), and it measures whether a *route* is found, not
whether enough delay history exists to score it afterwards.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://iris.noncd.db.de/iris-tts/timetable"
UA = "BayesianBahn/0.1 (F-Droid; FOSS delay prediction)"
ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).parent / ".iris-cache"

# Mirrors JourneyPlanner / ConnectionPlanner.
MAX_TRANSFER_SCAN = 15
MAX_TRANSFER_RESULTS = 3
MIN_TRANSFER_WEIGHT = 40
DETOUR_TOLERANCE = 1.25
TRANSFER_MINUTES = 5
ORIGIN_HOURS, TRANSFER_HOURS = 3, 4
DT_CATEGORIES = {"RE", "RB", "S", "IRE", "RS"}

DESIGNATIONS = {"bahnhof", "hauptbahnhof", "personenbahnhof", "haltepunkt", "haltestelle"}
ABBREVIATIONS = {"hbf": "hauptbahnhof", "bf": "bahnhof", "bhf": "bahnhof",
                 "pbf": "personenbahnhof", "hp": "haltepunkt", "hst": "haltestelle"}
requests_made = 0


def core(name: str) -> tuple:
    """Mirrors StationNames.core."""
    n = name.lower().replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss")
    tokens = [ABBREVIATIONS.get(t, t) for t in re.sub(r"[^a-z0-9]+", " ", n).split()]
    while tokens and tokens[-1] in DESIGNATIONS:
        tokens.pop()
    return tuple(tokens)


def fetch(path: str) -> str:
    global requests_made
    CACHE.mkdir(exist_ok=True)
    key = CACHE / (re.sub(r"[^A-Za-z0-9]+", "_", path) + ".xml")
    if key.exists():
        return key.read_text(encoding="utf-8")
    try:
        req = urllib.request.Request(f"{BASE}/{path}", headers={"User-Agent": UA})
        body = urllib.request.urlopen(req, timeout=60).read().decode()
    except Exception:
        body = ""
    requests_made += 1
    time.sleep(0.05)
    key.write_text(body, encoding="utf-8")
    return body


def board(eva: str, start: dt.datetime, hours: int) -> list[dict]:
    stops = []
    for i in range(hours):
        t = start + dt.timedelta(hours=i)
        for m in re.finditer(r'<s id="[^"]*".*?</s>', fetch(f"plan/{eva}/{t:%y%m%d}/{t:%H}"), re.S):
            s = m.group(0)
            tl = re.search(r'<tl[^>]*c="([^"]*)"[^>]*n="([^"]*)"', s)
            if not tl:
                continue
            dp, ar = re.search(r"<dp ([^>]*)>", s), re.search(r"<ar ([^>]*)>", s)

            def attr(tag, key):
                if not tag:
                    return None
                found = re.search(f'{key}="([^"]*)"', tag.group(1))
                return found.group(1) if found else None

            stops.append({"cat": tl.group(1), "num": tl.group(2),
                          "dep": attr(dp, "pt"), "arr": attr(ar, "pt"),
                          "path": (attr(dp, "ppth") or "").split("|") if dp else []})
    return stops


_iris_names: dict[str, str] = {}


def iris_name(eva: str) -> str | None:
    """Mirrors RouteStationMatcher: ask IRIS what it calls this station."""
    if eva not in _iris_names:
        xml = fetch(f"station/{eva}")
        m = (re.search(r'<station[^>]*name="([^"]*)"[^>]*eva="0*%s"' % eva.lstrip("0"), xml)
             or re.search(r'<station[^>]*name="([^"]*)"', xml))
        _iris_names[eva] = m.group(1) if m else ""
    return _iris_names[eva] or None


def stations():
    by_name, by_eva = {}, {}
    for line in (ROOT / "app/src/main/assets/stations.csv").read_text(encoding="utf-8").splitlines():
        p = line.split(";")
        if len(p) < 5:
            continue
        st = {"eva": p[0], "name": p[1], "weight": int(p[2]),
              "lat": float(p[3]), "lon": float(p[4])}
        by_name.setdefault(core(p[1]), st)
        by_eva[p[0]] = st
    return by_name, by_eva


def km(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    x = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * 6371 * math.asin(min(1, math.sqrt(x)))


def when(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%y%m%d%H%M")


def search(origin, destination, depart, by_name, *, ranking, per_feeder, max_attempts):
    """Replays the transfer search; returns the attempt index of each itinerary."""
    dest_iris = iris_name(destination["eva"])

    def is_dest(entry: str) -> bool:
        return ((dest_iris is not None and entry.lower() == dest_iris.lower())
                or core(entry) == core(destination["name"]))

    deps = sorted((s for s in board(origin["eva"], depart, ORIGIN_HOURS)
                   if s["dep"] and when(s["dep"]) >= depart and s["cat"] in DT_CATEGORIES),
                  key=lambda s: s["dep"])
    others = [s for s in deps if not any(is_dest(p) for p in s["path"])]
    direct = len(deps) - len(others)

    goal = km(origin, destination)
    attempts, hits, tried = 0, [], []
    for feeder in others[:MAX_TRANSFER_SCAN]:
        if attempts >= max_attempts:
            break
        cands = [by_name[core(p)] for p in feeder["path"]
                 if core(p) in by_name
                 and by_name[core(p)]["weight"] >= MIN_TRANSFER_WEIGHT
                 and by_name[core(p)]["eva"] != destination["eva"]
                 and by_name[core(p)]["name"] not in tried]
        if ranking == "distance":
            cands = [c for c in cands if km(c, destination) <= goal * DETOUR_TOLERANCE]
            cands.sort(key=lambda c: km(c, destination))
        else:
            cands.sort(key=lambda c: -c["weight"])
        for c in cands[:per_feeder]:
            if attempts >= max_attempts:
                break
            attempts += 1
            tried.append(c["name"])
            at = board(c["eva"], when(feeder["dep"]), TRANSFER_HOURS)
            here = [x for x in at
                    if x["cat"] == feeder["cat"] and x["num"] == feeder["num"] and x["arr"]]
            if not here:
                continue
            arrival = when(here[0]["arr"])
            if any(x["dep"] and x["cat"] in DT_CATEGORIES
                   and any(is_dest(p) for p in x["path"])
                   and when(x["dep"]) >= arrival + dt.timedelta(minutes=TRANSFER_MINUTES)
                   for x in at):
                hits.append(attempts)
    return {"direct": direct, "hits": hits, "attempts": attempts, "tried": tried}


def generate(out: Path, count: int, seed: int) -> None:
    by_name, by_eva = stations()
    random.seed(seed)
    origins = [s for s in by_eva.values() if 250 <= s["weight"] <= 900]
    random.shuffle(origins)
    times = ["16:00", "17:00", "18:00"]
    day = (dt.date.today()).isoformat()
    queries, used = [], set()

    for origin in origins:
        if len(queries) >= count:
            break
        depart = dt.datetime.fromisoformat(f"{day}T{times[len(queries) % 3]}")
        deps = sorted((s for s in board(origin["eva"], depart, ORIGIN_HOURS)
                       if s["dep"] and when(s["dep"]) >= depart
                       and s["cat"] in DT_CATEGORIES and s["path"]), key=lambda s: s["dep"])
        # Anything reachable without changing cannot measure the transfer search.
        direct_reach = {core(p) for s in deps for p in s["path"]}
        for feeder in deps[:8]:
            made = False
            for name in feeder["path"]:
                via = by_name.get(core(name))
                if not via or via["weight"] < MIN_TRANSFER_WEIGHT or via["eva"] == origin["eva"]:
                    continue
                at = board(via["eva"], when(feeder["dep"]), TRANSFER_HOURS)
                here = [x for x in at if x["cat"] == feeder["cat"]
                        and x["num"] == feeder["num"] and x["arr"]]
                if not here:
                    continue
                arrival = when(here[0]["arr"])
                for onward in at:
                    if (not onward["dep"] or onward["cat"] not in DT_CATEGORIES
                            or when(onward["dep"]) < arrival + dt.timedelta(minutes=TRANSFER_MINUTES)):
                        continue
                    for dest_name in onward["path"]:
                        dest = by_name.get(core(dest_name))
                        if (not dest or core(dest_name) in direct_reach
                                or dest["eva"] in (origin["eva"], via["eva"])
                                or (origin["eva"], dest["eva"]) in used
                                or not 25 <= km(origin, dest) <= 100):
                            continue
                        used.add((origin["eva"], dest["eva"]))
                        queries.append({
                            "from": origin["eva"], "to": dest["eva"],
                            "depart": depart.isoformat(),
                            "witness": {"via": via["name"],
                                        "feeder": f"{feeder['cat']} {feeder['num']}",
                                        "onward": f"{onward['cat']} {onward['num']}"},
                        })
                        made = True
                        break
                    if made:
                        break
                if made:
                    break
            if made:
                break

    out.write_text(json.dumps(queries, indent=1), encoding="utf-8")
    print(f"wrote {len(queries)} solvable queries to {out} ({requests_made} IRIS requests)")


def run(path: Path, max_attempts: int) -> None:
    by_name, by_eva = stations()
    queries = json.loads(path.read_text(encoding="utf-8"))
    strategies = [("weight", 2), ("distance", 1), ("distance", 2), ("distance", 3)]
    results = {f"{r}/{p}": [] for r, p in strategies}

    for i, q in enumerate(queries, 1):
        origin, destination = by_eva[q["from"]], by_eva[q["to"]]
        depart = dt.datetime.fromisoformat(q["depart"])
        for ranking, per_feeder in strategies:
            results[f"{ranking}/{per_feeder}"].append(search(
                origin, destination, depart, by_name,
                ranking=ranking, per_feeder=per_feeder, max_attempts=max_attempts))
        first = results["distance/2"][-1]["hits"]
        print(f"[{i:2}/{len(queries)}] {origin['name'][:22]:22} -> {destination['name'][:22]:22}"
              f"  first_hit={first[0] if first else None}", flush=True)

    n = len(queries)
    print(f"\nn = {n} queries, each with a one-transfer connection that provably exists")
    print("\nshare solved, by attempt budget")
    print("budget       " + "".join(f"{b:>5}" for b in range(1, max_attempts + 1)))
    for key, rows in results.items():
        print(f"{key:<12} " + "".join(
            f"{sum(1 for r in rows if r['hits'] and min(r['hits']) <= b) / n:>5.0%}"
            for b in range(1, max_attempts + 1)))

    print("\ncost of the chosen strategy (1 attempt = 1 transfer board = 5 IRIS requests)")
    rows = results["distance/2"]
    for b in range(2, max_attempts + 1, 2):
        spent = []
        for r in rows:
            got = [h for h in r["hits"] if h <= b]
            spent.append(got[MAX_TRANSFER_RESULTS - 1] if len(got) >= MAX_TRANSFER_RESULTS
                         else min(b, r["attempts"]))
        solved = sum(1 for r in rows if r["hits"] and min(r["hits"]) <= b)
        print(f"  budget {b:2}: solved {solved:2}/{n} ({solved / n:.0%})  "
              f"mean attempts {sum(spent) / n:4.1f}  ~{sum(spent) / n * 5:4.0f} requests")

    print(f"\nIRIS requests this run: {requests_made} (rest served from {CACHE})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate", type=Path, help="write a solvable query set to this file")
    ap.add_argument("--run", type=Path, help="benchmark the strategies on a query set")
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=14)
    args = ap.parse_args()

    if args.generate:
        generate(args.generate, args.count, args.seed)
    elif args.run:
        run(args.run, args.max_attempts)
    else:
        ap.error("need --generate or --run")


if __name__ == "__main__":
    main()
