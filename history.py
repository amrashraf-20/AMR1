"""
Local telemetry history — SQLite, 24h retention, browsable by the hour.

Why this exists and why it is NOT the postgres `db`:
    `db.ok` is False on this machine (the psycopg2 .pyd is blocked by an
    Application Control policy, see the ledger). History has to keep working
    when the car keeps working, so this is a plain stdlib `sqlite3` file with
    zero project dependencies. Postgres keeps doing whatever it was doing.

Shape of the data:
    The firmware publishes telemetry at 10 Hz moving / 5 Hz idle
    (`hardware_test.ino`: TELEMETRY_MOVING_MS / TELEMETRY_MS). Storing all of
    it for 24h is 432k-864k rows/day for numbers the dashboard samples at 1 Hz
    anyway. So we store ONE ROW PER SECOND and keep the raw series intact for
    plotting. ~86k rows/day, ~5-8 MB/day, hard-capped by the retention delete.

Hour buckets are LOCAL wall-clock, not UTC. "14:00-15:00 today" has to mean the
operator's wall clock, so every row carries a `hour` TEXT column ("2026-09-30T14:00")
computed at insert time. That makes GROUP BY / WHERE a plain indexed lookup and
avoids float-modulo bucketing entirely.

Every public method is failure-isolated on purpose: a broken history must never
be able to take the car server down. Same principle as the psycopg2 import
guard — the optional thing may fail, the required thing may not.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent

DB_PATH = Path(os.environ.get("HISTORY_DB", str(HERE / "history.sqlite")))
SNAPSHOT_DIR = Path(os.environ.get("HISTORY_SNAPSHOT_DIR", str(HERE / "history_snapshots")))
RETENTION_HOURS = float(os.environ.get("HISTORY_RETENTION_HOURS", "24"))
SAMPLE_MS = int(os.environ.get("HISTORY_SAMPLE_MS", "1000"))
PRUNE_EVERY_TICKS = 60          # ~ once a minute at the default 1 Hz cadence

# Columns lifted out of the telemetry payload. Keeping these as real columns
# (not a JSON blob) is what makes AVG()/GROUP BY cheap and the file small.
_SAMPLE_COLS = (
    "ts", "hour", "state", "mode", "distance", "distance_valid",
    "speed", "pwm_now", "pwm_l", "pwm_r", "pwm_target", "pwm_lease",
    "rssi", "uptime_ms", "obstacle_blocked", "sensor_dead", "lifted",
    "interlock", "avoid_phase",
    "polls", "poll_failures", "hints_received", "hints_rejected",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    ts               REAL PRIMARY KEY,
    hour             TEXT NOT NULL,
    state            TEXT,
    mode             TEXT,
    distance         REAL,
    distance_valid   INTEGER,
    speed            INTEGER,
    pwm_now          INTEGER,
    pwm_l            INTEGER,
    pwm_r            INTEGER,
    pwm_target       INTEGER,
    pwm_lease        INTEGER,
    rssi             INTEGER,
    uptime_ms        INTEGER,
    obstacle_blocked INTEGER,
    sensor_dead      INTEGER,
    lifted           INTEGER,
    interlock        TEXT,
    avoid_phase      TEXT,
    polls            INTEGER,
    poll_failures    INTEGER,
    hints_received   INTEGER,
    hints_rejected   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_samples_hour ON samples(hour);

CREATE TABLE IF NOT EXISTS commands (
    ts          REAL,
    hour        TEXT NOT NULL,
    command_id  TEXT,
    kind        TEXT,
    direction   TEXT,
    speed       INTEGER,
    approved_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_commands_hour ON commands(hour);

CREATE TABLE IF NOT EXISTS snapshots (
    ts        REAL,
    hour      TEXT NOT NULL,
    filename  TEXT NOT NULL,
    direction TEXT,
    distance  REAL,
    rssi      INTEGER,
    -- What YOLO saw at that instant. Captured from the LIVE vision state at
    -- the moment the snapshot is written; it is a read of what the pipeline
    -- already had, never a second inference and never a change to the stream.
    description   TEXT,
    path_clear    INTEGER,
    object_count  INTEGER,
    objects_in_path TEXT,
    objects_beside  TEXT
);
CREATE INDEX IF NOT EXISTS idx_snapshots_hour ON snapshots(hour);
"""

# Added after the first release of this table. SQLite has no
# `ADD COLUMN IF NOT EXISTS`, so each is applied only when absent — an
# existing history.sqlite keeps its rows and simply gains the columns.
_SNAPSHOT_MIGRATIONS = (
    ("description", "TEXT"),
    ("path_clear", "INTEGER"),
    ("object_count", "INTEGER"),
    ("objects_in_path", "TEXT"),
    ("objects_beside", "TEXT"),
)


def hour_key(ts: float) -> str:
    """Epoch seconds -> local wall-clock hour label, e.g. '2026-09-30T14:00'."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:00")


def hour_bounds(hour: str) -> tuple[float, float]:
    """'2026-09-30T14:00' -> (start_epoch, end_epoch) in local time."""
    start = datetime.strptime(hour, "%Y-%m-%dT%H:%M")
    return start.timestamp(), (start + timedelta(hours=1)).timestamp()


def hour_label(hour: str) -> str:
    """'2026-09-30T14:00' -> '14:00-15:00' (the operator's wording for it)."""
    start, end = hour_bounds(hour)
    return f"{datetime.fromtimestamp(start):%H:%M}-{datetime.fromtimestamp(end):%H:%M}"


def _num(v: Any) -> Optional[float]:
    """Coerce to a number or None. Booleans are ints in Python — refuse them."""
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _flag(v: Any) -> Optional[int]:
    return None if v is None else (1 if v else 0)


# Words that carry no search meaning. Without this, "what happened at 16:40"
# becomes the query `what happened at` and matches nothing, because none of
# those words appear in a scene description. The question is the frame; the
# content is the nouns. Deliberately small — this is a search box, not a
# language model.
_STOPWORDS = frozenset({
    "what", "when", "where", "which", "who", "why", "how", "did", "does", "do",
    "was", "were", "is", "are", "the", "and", "or", "that", "this", "it",
    "at", "in", "on", "of", "to", "for", "from", "about", "tell", "me",
    "show", "give", "happen", "happened", "happens", "see", "saw", "seen",
    "تم", "في", "من", "على", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "كان", "كانت", "يكون", "هل", "ماذا", "متى", "اين", "أين", "كيف",
    "لماذا", "لم", "هل happened", "اخبرني", "قولي", "وريني", "اعرض",
    "عرض", "قي", "شي", "شيء",
})




class History:
    def __init__(self, path: Path = DB_PATH, snapshot_dir: Path = SNAPSHOT_DIR):
        self.path = Path(path)
        self.snapshot_dir = Path(snapshot_dir)
        self.retention_hours = RETENTION_HOURS
        self.sample_ms = SAMPLE_MS
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._task: Optional[asyncio.Task] = None
        self._ticks = 0
        self.last_error: Optional[str] = None

    # ---------- lifecycle ----------
    def connect(self) -> None:
        """Open the file and make the schema. Safe to call twice."""
        with self._lock:
            if self._conn is not None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False: telemetry arrives on the MQTT thread
            # (via handle_car_event) while reads happen on the event loop.
            # One lock serialises them; an insert at 1 Hz holds it ~100us.
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            have = {r["name"] for r in self._conn.execute("PRAGMA table_info(snapshots)")}
            for col, kind in _SNAPSHOT_MIGRATIONS:
                if col not in have:
                    self._conn.execute(f"ALTER TABLE snapshots ADD COLUMN {col} {kind}")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                with contextlib.suppress(Exception):
                    self._conn.close()
                self._conn = None

    async def start(self, get_telemetry, get_guide) -> None:
        """Start the 1 Hz sampler. `get_*` are callables returning dicts."""
        self.connect()
        self._task = asyncio.get_running_loop().create_task(self._loop(get_telemetry, get_guide))
        await asyncio.sleep(0)
        self.prune()          # a laptop that was off for a week wakes up clean

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self.close()

    async def _loop(self, get_telemetry, get_guide) -> None:
        interval = max(0.05, self.sample_ms / 1000.0)
        while True:
            try:
                await asyncio.sleep(interval)
                self.sample(get_telemetry(), get_guide())
                self._ticks += 1
                if self._ticks % PRUNE_EVERY_TICKS == 0:
                    self.prune()
            except asyncio.CancelledError:
                raise
            except Exception as exc:                       # never kill the server
                self.last_error = f"{type(exc).__name__}: {exc}"
                print(f"[history] sampler error: {self.last_error}", flush=True)

    # ---------- writes ----------
    def sample(self, tele: Optional[dict], guide: Optional[dict]) -> None:
        """One row per second. No-ops when there is no live telemetry yet."""
        if not tele:
            return
        now = time.time()
        g = guide or {}
        row = (
            now, hour_key(now),
            tele.get("state"), tele.get("mode"),
            _num(tele.get("distance")), _flag(tele.get("distance_valid")),
            _num(tele.get("speed")), _num(tele.get("pwm_now")),
            _num(tele.get("pwm_l")), _num(tele.get("pwm_r")),
            _num(tele.get("pwm_target")), _num(tele.get("pwm_lease")),
            _num(tele.get("rssi")), _num(tele.get("uptime_ms")),
            _flag(tele.get("obstacle_blocked")), _flag(tele.get("sensor_dead")),
            _flag(tele.get("lifted")), tele.get("interlock"), tele.get("avoid_phase"),
            _num(g.get("polls")), _num(g.get("poll_failures")),
            _num(g.get("hints_received")), _num(g.get("hints_rejected")),
        )
        with self._lock:
            if self._conn is None:
                return
            self._conn.execute(
                f"INSERT OR REPLACE INTO samples ({','.join(_SAMPLE_COLS)}) "
                f"VALUES ({','.join('?' * len(_SAMPLE_COLS))})", row)
            self._conn.commit()

    def record_command(self, cmd: dict, direction: Optional[str] = None,
                       speed: Optional[int] = None) -> None:
        """
        One row per ISSUED motion command.

        Hooked next to `db.save_command` in `MotionController.start`. Counting
        commands from the telemetry stream instead would be wrong: it cannot
        tell an operator command from a firmware auto-avoid, and it would
        miss a command whose telemetry arrived after the stop.
        """
        try:
            now = time.time()
            row = (
                now, hour_key(now),
                str(cmd.get("command_id") or ""),
                str(cmd.get("kind") or ""),
                direction or cmd.get("direction"),
                _num(speed if speed is not None else cmd.get("speed")),
                cmd.get("approved_by"),
            )
            with self._lock:
                if self._conn is None:
                    return
                self._conn.execute(
                    "INSERT INTO commands (ts,hour,command_id,kind,direction,speed,approved_by)"
                    " VALUES (?,?,?,?,?,?,?)", row)
                self._conn.commit()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[history] command record failed: {self.last_error}", flush=True)

    def record_snapshot(self, raw: bytes, tele: Optional[dict],
                        scene: Optional[dict] = None) -> Optional[str]:
        """
        Save a JPEG into history_snapshots/ and link it to the telemetry state
        at that moment. Returns the filename, or None.

        Called from `store_evidence`, which is the single choke point every
        snapshot already goes through (the dashboard SNAPSHOT button, the
        waypoint confirm, and auto-capture all funnel there). One hook covers
        all of them instead of three call sites.

        `scene` is the already-computed vision read-out for this instant
        (`vision.describe(...)` output). It is passed in, never computed here,
        so this module stays independent of the vision pipeline and cannot
        trigger or perturb inference.
        """
        if not raw or len(raw) < 512:
            return None
        try:
            now = time.time()
            name = f"snap-{datetime.fromtimestamp(now):%Y%m%d-%H%M%S}-{int(now * 1000) % 1000:03d}.jpg"
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            (self.snapshot_dir / name).write_bytes(raw)
            t = tele or {}
            s = scene or {}
            with self._lock:
                if self._conn is None:
                    self.connect()
                if self._conn is None:
                    return None
                self._conn.execute(
                    "INSERT INTO snapshots (ts,hour,filename,direction,distance,rssi,"
                    "description,path_clear,object_count,objects_in_path,objects_beside)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (now, hour_key(now), name, t.get("state"),
                     _num(t.get("distance")), _num(t.get("rssi")),
                     s.get("description"), _flag(s.get("path_clear")),
                     _num(s.get("object_count")),
                     json.dumps(s.get("objects_in_path") or [], ensure_ascii=False),
                     json.dumps(s.get("objects_beside") or [], ensure_ascii=False)))
                self._conn.commit()
            return name
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[history] snapshot record failed: {self.last_error}", flush=True)
            return None

    def prune(self) -> int:
        """
        Delete anything older than the retention window, INCLUDING the files on
        disk. Deleting the row without unlinking the JPEG would leak a file per
        snapshot forever — the row and the file have the same lifetime.

        Returns the number of sample rows removed (for the log line / tests).
        """
        try:
            cutoff = time.time() - self.retention_hours * 3600.0
            with self._lock:
                if self._conn is None:
                    return 0
                stale = [r[0] for r in self._conn.execute(
                    "SELECT filename FROM snapshots WHERE ts < ?", (cutoff,))]
                n = self._conn.execute("DELETE FROM samples WHERE ts < ?", (cutoff,)).rowcount
                self._conn.execute("DELETE FROM commands WHERE ts < ?", (cutoff,))
                self._conn.execute("DELETE FROM snapshots WHERE ts < ?", (cutoff,))
                self._conn.commit()
            removed = 0
            for name in stale:
                # Guard the join: a name from the DB must never escape the dir.
                path = (self.snapshot_dir / name).resolve()
                if path.parent == self.snapshot_dir.resolve() and path.is_file():
                    with contextlib.suppress(OSError):
                        path.unlink()
                        removed += 1
            if n or removed:
                print(f"[history] pruned {n} samples, {removed} snapshot files "
                      f"(older than {self.retention_hours:g}h)", flush=True)
            return max(0, n)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[history] prune failed: {self.last_error}", flush=True)
            return 0

    # ---------- reads ----------
    def hours(self) -> list[dict]:
        """Every hour bucket that actually has data, newest first."""
        self.connect()
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.hour AS hour, COUNT(*) AS samples,"
                " (SELECT COUNT(*) FROM commands c WHERE c.hour = s.hour) AS commands,"
                " (SELECT COUNT(*) FROM snapshots p WHERE p.hour = s.hour) AS snapshots"
                " FROM samples s GROUP BY s.hour ORDER BY s.hour DESC").fetchall()
        return [dict(r) | {"label": hour_label(r["hour"])} for r in rows]

    def hour(self, hour: str) -> dict:
        """Summary + downsampled series + commands + snapshots for one hour."""
        self.connect()
        start, end = hour_bounds(hour)
        with self._lock:
            agg = self._conn.execute(
                "SELECT COUNT(*) AS samples, MIN(ts) AS first_ts, MAX(ts) AS last_ts,"
                " AVG(rssi) AS rssi_avg, MIN(rssi) AS rssi_min, MAX(rssi) AS rssi_max,"
                " AVG(distance) AS dist_avg, MIN(distance) AS dist_min, MAX(distance) AS dist_max,"
                " AVG(speed) AS speed_avg, MAX(speed) AS speed_max,"
                " MAX(pwm_now) AS pwm_max, MAX(pwm_l) AS pwm_l_max, MAX(pwm_r) AS pwm_r_max,"
                " MIN(uptime_ms) AS uptime_min, MAX(uptime_ms) AS uptime_max,"
                " SUM(COALESCE(obstacle_blocked,0)) AS obstacle_samples,"
                " SUM(COALESCE(sensor_dead,0)) AS sensor_dead_samples,"
                " SUM(COALESCE(lifted,0)) AS lifted_samples,"
                " MAX(polls) AS polls_max, MAX(poll_failures) AS poll_failures_max,"
                " MAX(hints_received) AS hints_received_max, MAX(hints_rejected) AS hints_rejected_max,"
                " MIN(polls) AS polls_min, MIN(poll_failures) AS poll_failures_min,"
                " MIN(hints_received) AS hints_received_min, MIN(hints_rejected) AS hints_rejected_min"
                " FROM samples WHERE hour = ?", (hour,)).fetchone()
            series = self._conn.execute(
                "SELECT ts, distance, speed, rssi FROM samples WHERE hour = ?"
                " ORDER BY ts", (hour,)).fetchall()
            commands = self._conn.execute(
                "SELECT ts, command_id, kind, direction, speed, approved_by FROM commands"
                " WHERE hour = ? ORDER BY ts", (hour,)).fetchall()
            snapshots = self._conn.execute(
                "SELECT ts, filename, direction, distance, rssi, description,"
                " path_clear, object_count, objects_in_path, objects_beside"
                " FROM snapshots WHERE hour = ? ORDER BY ts", (hour,)).fetchall()
            by_dir = self._conn.execute(
                "SELECT COALESCE(direction,'?') AS direction, COUNT(*) AS n"
                " FROM commands WHERE hour = ? GROUP BY 1 ORDER BY n DESC", (hour,)).fetchall()
            by_kind = self._conn.execute(
                "SELECT COALESCE(kind,'?') AS kind, COUNT(*) AS n"
                " FROM commands WHERE hour = ? GROUP BY 1 ORDER BY n DESC", (hour,)).fetchall()
            states = self._conn.execute(
                "SELECT COALESCE(state,'?') AS state, COUNT(*) AS n"
                " FROM samples WHERE hour = ? GROUP BY 1 ORDER BY n DESC", (hour,)).fetchall()

        d = dict(agg) if agg else {}
        n = d.get("samples") or 0
        uptime_span = None
        if d.get("uptime_min") is not None and d.get("uptime_max") is not None:
            uptime_span = round((d["uptime_max"] - d["uptime_min"]) / 1000.0, 1)

        # Counter deltas: the guide restarts these at 0, so a plain max would
        # hide errors that happened before the reset. min->max is the honest
        # "how much did this move during the hour".
        def delta(hi, lo):
            if d.get(hi) is None or d.get(lo) is None:
                return None
            return max(0, round(d[hi] - d[lo]))

        summary = {
            "hour": hour,
            "label": hour_label(hour),
            "start": start,
            "end": end,
            "samples": n,
            "coverage_seconds": n,
            "commands_total": len(commands),
            "directions": [{"direction": r["direction"], "count": r["n"]} for r in by_dir],
            "kinds": [{"kind": r["kind"], "count": r["n"]} for r in by_kind],
            "states": [{"state": r["state"], "seconds": r["n"]} for r in states],
            "rssi_avg": _round1(d.get("rssi_avg")),
            "rssi_min": d.get("rssi_min"),
            "rssi_max": d.get("rssi_max"),
            "distance_avg": _round1(d.get("dist_avg")),
            "distance_min": d.get("dist_min"),
            "distance_max": d.get("dist_max"),
            "speed_avg": _round1(d.get("speed_avg")),
            "speed_max": d.get("speed_max"),
            "pwm_max": d.get("pwm_max"),
            "pwm_l_max": d.get("pwm_l_max"),
            "pwm_r_max": d.get("pwm_r_max"),
            "uptime_span_seconds": uptime_span,
            "obstacle_samples": d.get("obstacle_samples") or 0,
            "sensor_dead_samples": d.get("sensor_dead_samples") or 0,
            "lifted_samples": d.get("lifted_samples") or 0,
            "polls": delta("polls_max", "polls_min"),
            "poll_failures": delta("poll_failures_max", "poll_failures_min"),
            "hints_received": delta("hints_received_max", "hints_received_min"),
            "hints_rejected": delta("hints_rejected_max", "hints_rejected_min"),
            "errors_total": sum(filter(None, (
                delta("poll_failures_max", "poll_failures_min"),
                delta("hints_rejected_max", "hints_rejected_min"),
            ))) + (d.get("obstacle_samples") or 0) + (d.get("sensor_dead_samples") or 0),
        }
        return {
            "summary": summary,
            "series": _downsample(series, 240),
            "commands": [dict(r) for r in commands],
            "snapshots": [dict(r) | {"url": f"/api/history/snapshot/{r['filename']}",
                                     "label": f"{datetime.fromtimestamp(r['ts']):%H:%M:%S}",
                                     "objects_in_path": _loads(r["objects_in_path"]),
                                     "objects_beside": _loads(r["objects_beside"])}
                          for r in snapshots],
        }

    # ---------- the history query assistant ----------
    #
    # Read-only. Nothing here writes, and nothing here touches the live
    # telemetry path: it searches rows that already exist inside the
    # retention window and pairs each snapshot with the telemetry sample that
    # was recorded closest to it in time.
    #
    # Deliberately NOT an LLM call. The question is answered by matching what
    # is on disk, so it works with no API key configured, costs nothing, and
    # cannot hallucinate a moment that never happened.

    def query(self, text: str = "", hour: Optional[str] = None,
              limit: int = 20) -> dict:
        """
        Search the saved record.

        `text` is free-form. An explicit `HH:MM` (or `HH:MM-HH:MM`) inside it
        wins over keyword matching, because "what happened at 14:00" is a
        question about a time, not a keyword. Remaining terms are matched
        against the recorded scene description, the detected object labels and
        the direction.

        Always returns a result set, even for a query that matches nothing,
        because an empty answer is a legitimate answer to "what did the car
        see?".
        """
        self.connect()
        raw = (text or "").strip()

        # The clock is read FIRST and then blanked out of the text before
        # tokenising. "what happened at 18:00" would otherwise tokenise to
        # ['what','happened','at','18','00'] — and since '18' and '00' are two
        # characters they survive any length filter, become search terms, match
        # no description, and the query returns nothing at all.
        wanted_hour = hour
        haystack = raw
        if not wanted_hour:
            m = re.search(r"\b(\d{1,2}):(\d{2})\b", raw)
            if m:
                hh, mi = int(m.group(1)), int(m.group(2))
                if 0 <= hh <= 23 and 0 <= mi <= 59:
                    wanted_hour = datetime.now().strftime("%Y-%m-%dT") + f"{hh:02d}:00"
                    haystack = raw[:m.start()] + " " + raw[m.end():]

        tokens = [t for t in re.split(r"[^\w؀-ۿ]+", haystack.lower()) if t]
        # A one-character token matches almost anything — an Arabic hamza or a
        # stray letter would let every snapshot through and turn the search
        # into noise. Meaningless on their own in any language.
        short = [t for t in tokens if len(t) < 2]
        words = [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]
        terms = words
        dropped = short + [t for t in tokens if len(t) >= 2 and t in _STOPWORDS]

# Nothing searchable was typed — every token was a question word ("what
        # happened"), a bare letter, or a clock we already consumed. Do NOT
        # return an empty box: in a query box, "what happened" plainly means
        # "show me what happened", so fall back to the most recent moments
        # and say so. An empty result here reads as "the record is empty",
        # which would be a lie.
        vague = bool(raw) and not terms
        note = (
            "nothing specific to search for - showing the most recent moments"
            if vague else None
        )

        with self._lock:
            if wanted_hour:
                rows = self._conn.execute(
                    "SELECT * FROM snapshots WHERE hour = ? ORDER BY ts DESC LIMIT 200",
                    (wanted_hour,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM snapshots ORDER BY ts DESC LIMIT 500").fetchall()
            retained_from = time.time() - self.retention_hours * 3600.0
            # Telemetry only for the hours we actually touched: at 1 Hz an
            # hour is 3600 rows, and this stays a lookup rather than a scan.
            stamps = [r["ts"] for r in rows]
            tele_by_ts: dict[int, sqlite3.Row] = {}
            for ts in stamps:
                near = self._conn.execute(
                    "SELECT * FROM samples WHERE ts BETWEEN ? AND ? ORDER BY ABS(ts - ?) LIMIT 1",
                    (ts - 3, ts + 3, ts)).fetchone()
                if near is not None:
                    tele_by_ts[int(near["ts"])] = near

        scored = []
        for r in rows:
            labels = " ".join(
                filter(None, [
                    r["description"] or "",
                    r["objects_in_path"] or "",
                    r["objects_beside"] or "",
                    str(r["direction"] or ""),
                ])
            ).lower()
            hits = [t for t in terms if t in labels]
            if terms and not hits:
                continue
            near = tele_by_ts.get(int(r["ts"]))
            scored.append({
                "score": len(hits),
                "ts": r["ts"],
                "hour": r["hour"],
                "label": f"{datetime.fromtimestamp(r['ts']):%H:%M:%S}",
                "day": f"{datetime.fromtimestamp(r['ts']):%Y-%m-%d}",
                "filename": r["filename"],
                "url": f"/api/history/snapshot/{r['filename']}",
                "direction": r["direction"],
                "distance": r["distance"],
                "rssi": r["rssi"],
                "description": r["description"],
                "path_clear": None if r["path_clear"] is None else bool(r["path_clear"]),
                "object_count": r["object_count"],
                "objects_in_path": _loads(r["objects_in_path"]),
                "objects_beside": _loads(r["objects_beside"]),
                "matched": hits,
                # The exact numbers held at that instant, straight from the
                # 1 Hz sample rows — never recomputed.
                "telemetry": _telemetry_row(near),
            })
        scored.sort(key=lambda s: (s["score"], s["ts"]), reverse=True)

        return {
            "ok": True,
            "query": raw,
            "hour": wanted_hour,
            "hour_label": hour_label(wanted_hour) if wanted_hour else None,
            "matched_terms": terms,
            "dropped_terms": dropped,
            "total_scanned": len(rows),
            "retained_from": retained_from,
            "note": note,
            "count": len(scored[:limit]),
            "results": scored[:limit],
        }

    def snapshot_file(self, filename: str) -> Optional[Path]:
        """Resolve a stored filename to a real path, or None.

        The join is resolved and re-checked against the directory: a name that
        arrived over HTTP must never be able to walk out of the folder.
        """
        if not filename or "/" in filename or "\\" in filename or filename.startswith("."):
            return None
        path = (self.snapshot_dir / filename).resolve()
        if path.parent != self.snapshot_dir.resolve() or not path.is_file():
            return None
        return path

    def stats(self) -> dict:
        self.connect()
        with self._lock:
            try:
                size = self.path.stat().st_size
            except OSError:
                size = None
            s = self._conn.execute("SELECT COUNT(*) AS n, MIN(ts) AS a, MAX(ts) AS b"
                                   " FROM samples").fetchone()
            c = self._conn.execute("SELECT COUNT(*) AS n FROM commands").fetchone()
            p = self._conn.execute("SELECT COUNT(*) AS n FROM snapshots").fetchone()
        return {
            "ok": self.last_error is None,
            "path": str(self.path),
            "snapshot_dir": str(self.snapshot_dir),
            "retention_hours": self.retention_hours,
            "sample_ms": self.sample_ms,
            "size_bytes": size,
            "samples": s["n"] if s else 0,
            "oldest_ts": s["a"] if s else None,
            "newest_ts": s["b"] if s else None,
            "commands": c["n"] if c else 0,
            "snapshots": p["n"] if p else 0,
            "last_error": self.last_error,
        }


def _round1(v) -> Optional[float]:
    n = _num(v)
    return None if n is None else round(n, 1)


def _loads(v) -> list:
    """JSON text column -> list. Never raises on a bad column."""
    if not v:
        return []
    try:
        out = json.loads(v)
    except (TypeError, ValueError):
        return []
    return out if isinstance(out, list) else []


def _telemetry_row(row) -> Optional[dict]:
    """The stored telemetry at one instant, as the answer box needs to show it."""
    if row is None:
        return None
    return {
        "ts": row["ts"],
        "uptime_s": round((row["uptime_ms"] or 0) / 1000.0, 1) if row["uptime_ms"] else None,
        "distance_cm": row["distance"],
        "distance_valid": None if row["distance_valid"] is None else bool(row["distance_valid"]),
        "stop_distance_cm": None,
        "speed": row["speed"],
        "direction": row["state"],
        "mode": row["mode"],
        "pwm_now": row["pwm_now"],
        "pwm_left": row["pwm_l"],
        "pwm_right": row["pwm_r"],
        "pwm_target": row["pwm_target"],
        "pwm_lease": row["pwm_lease"],
        "rssi": row["rssi"],
        "obstacle_blocked": None if row["obstacle_blocked"] is None else bool(row["obstacle_blocked"]),
        "sensor_dead": None if row["sensor_dead"] is None else bool(row["sensor_dead"]),
        "lifted": None if row["lifted"] is None else bool(row["lifted"]),
        "interlock": row["interlock"],
        "avoid_phase": row["avoid_phase"],
        "polls": row["polls"],
        "poll_failures": row["poll_failures"],
        "hints_received": row["hints_received"],
        "hints_rejected": row["hints_rejected"],
    }


def _downsample(rows, limit: int) -> list[dict]:
    """
    An hour is up to 3600 rows. Keep at most `limit` for the chart by taking
    evenly spaced indices — the raw data stays in the DB, this is only what
    the browser gets.
    """
    total = len(rows)
    if total <= limit:
        return [dict(r) for r in rows]
    step = total / float(limit)
    picked = {int(i * step) for i in range(limit)}
    return [dict(rows[i]) for i in sorted(picked) if i < total]


history = History()
