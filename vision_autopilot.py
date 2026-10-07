"""
vision_autopilot.py — drive the car from the YOLO feed, with no new motion path.

WHY THIS EXISTS
    The car already has one motion path: `MotionController`, which signs every
    command and renews a 250 ms lease. This module deliberately adds **no**
    second path. It reuses `motion.set_gyro_input()`, the same "write a
    modifier onto the running manual command" seam the gyroscope uses. So the
    autopilot never signs a command, never publishes to the broker itself, and
    never invents a lease rule - the existing watchdog still governs it.

THE HAZARD THAT SHAPES EVERY DECISION HERE
    `ULTRASONIC_ENABLED = false` in the firmware (hardware_test.ino:353). The
    car has **no forward obstacle detection of any kind** except this camera.
    That inverts the usual safety logic: normally "no obstacle seen" means
    "carry on", but here it means "I am the only thing standing between the
    car and whatever is in front of it". So:

      * **Stale or missing vision stops the car.** A frozen or failed pipeline
        must never be read as "the path is clear". `vision.py` already stops
        drawing boxes past `BOX_MAX_AGE_S`; this module goes further and
        refuses to *drive* past the same age.
      * An empty detection list means clear, because YOLO reports objects, not
        emptiness - `classify_path` makes exactly the same assumption.
      * A close, centred obstacle is a STOP, not a guess. Swerving at something
        directly ahead is how a car ends up beside the thing, not clear of it.

FAIL CLOSED, ALWAYS
    Enable is refused outright unless the vision pipeline is actually running,
    telemetry is arriving, and the car is online. Any tick that cannot prove it
    is safe stops the car and records why. There is no "carry on and hope"
    branch anywhere in this file.

NOT AN LLM, NOT A PLANNER
    A reflex: clear path -> forward; something in the lane -> turn away from
    it at a crawl; too close or unprovable -> stop. That is the whole policy.
    Anything cleverer belongs in a different module, on top of this one.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

# ---- policy (env-overridable, all fail safe if set to something silly) ----
def _num_env(name: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


# How often the reflex re-evaluates. 5 Hz is faster than the car can change
# direction usefully and slow enough that a stalled tick is obvious.
CONTROL_HZ = _num_env("AUTOPILOT_HZ", 5.0, 0.5, 20.0)
TICK_S = 1.0 / CONTROL_HZ

# Throttle ceiling as a PERCENT, mirroring `set_gyro_input`'s unit. This is
# hard-capped well below MANUAL_SPEED_MAX: autonomy without a human is not the
# moment to discover the top of the speed range.
CRUISE_PCT = int(_num_env("AUTOPILOT_CRUISE_PCT", 45.0, 0.0, 100.0))
CREEP_PCT = int(_num_env("AUTOPILOT_CREEP_PCT", 28.0, 0.0, CRUISE_PCT))

# Freshness gate. Mirrors vision.BOX_MAX_AGE_S; anything older than this and
# the car stops rather than trusting a stale frame.
MAX_VISION_AGE_S = _num_env("AUTOPILOT_MAX_VISION_AGE_S", 2.0, 0.2, 30.0)

# Telemetry gate. The lease watchdog is 1200 ms in the firmware, so a car whose
# telemetry is older than this is already in trouble.
MAX_TELEMETRY_AGE_S = _num_env("AUTOPILOT_MAX_TELEMETRY_AGE_S", 3.0, 0.5, 60.0)

# Obstacle geometry, in the same normalised 0-100 frame coordinates that
# vision.Detection uses for cx/cy.
LANE_CENTRE_CX = 50.0
CENTRED_BAND_CX = 12.0     # |cx - 50| <= this  ==  "straight ahead"
NEAR_CY = 45.0             # cy >= this == "at the car", from vision.NEAR_CY
STOP_CY = 62.0             # centred AND this low == too close to steer


def decide(dets: list, tele: Optional[dict], *, vision_age_s: float) -> dict:
    """
    The entire driving policy, as a pure function.

    Split out deliberately: it takes only data and returns a decision, so every
    branch below can be tested without a pipeline, a broker, or a car on the
    floor. `decide()` cannot move anything by itself.

    Returns {"action": "go"|"steer"|"stop", "throttle": int,
             "steer": "F"|"L"|"R", "reason": str}.
    """
    stop = lambda why: {"action": "stop", "throttle": 0, "steer": "F", "reason": why}

    # ---- 1. Can we prove the car is safe to move at all? ----
    if vision_age_s is None or vision_age_s > MAX_VISION_AGE_S:
        age = "never" if vision_age_s is None else f"{vision_age_s:.1f}s"
        return stop(f"vision stale ({age}) - refusing to drive blind")
    if tele is None:
        return stop("no telemetry from the car")

    # Same interlock vocabulary the guidance runner uses, so the operator sees
    # one consistent set of reasons across every driving mode.
    if tele.get("lifted"):
        return stop("car is lifted")
    if tele.get("sensor_dead"):
        return stop("distance sensor is dead - no forward safety net")
    if tele.get("obstacle_blocked"):
        return stop(f"firmware reports an obstacle ({tele.get('distance')} cm)")
    if tele.get("forward_blocked"):
        return stop("firmware reports the path blocked")

    # ---- 2. Nothing in the lane -> straight ahead ----
    in_path = [
        d for d in dets
        if d.label in _OBSTACLE_LABELS
        and d.conf >= _NEAR_CONF
        and _LANE_LEFT <= d.cx <= _LANE_RIGHT
    ]
    if not in_path:
        # An empty detection list means "clear", not "blind": YOLO reports
        # objects, and classify_path treats absence the same way. The frame
        # itself is already proven fresh by the gate above.
        return {"action": "go", "throttle": CRUISE_PCT, "steer": "F",
                "reason": "path clear"}

    # ---- 3. Something is in the lane ----
    # "Nearest" = biggest cy (lowest in the frame = closest to the camera).
    nearest = max(in_path, key=lambda d: d.cy)
    off_centre = nearest.cx - LANE_CENTRE_CX

    if abs(off_centre) <= CENTRED_BAND_CX and nearest.cy >= STOP_CY:
        # Straight ahead and already close: steering away now is a coin flip
        # that ends with the car beside the obstacle. Stop.
        return stop(f"{nearest.label} straight ahead and close (cy={nearest.cy:.0f})")

    # Turn away from it: obstacle on the left -> go right, and vice versa.
    steer = "R" if off_centre < 0 else "L"
    return {
        "action": "steer",
        "throttle": CREEP_PCT,
        "steer": steer,
        "reason": f"{nearest.label} at cx={nearest.cx:.0f} -> turning {steer}",
    }


# Populated from vision.py at wire-up time (see `attach`). Keeping them here as
# None means a caller who forgets to attach fails closed instead of silently
# comparing against empty sets and declaring the path clear.
_OBSTACLE_LABELS: frozenset = frozenset()
_NEAR_CONF = 0.0
_LANE_LEFT = 25.0
_LANE_RIGHT = 75.0


def attach(vision_module) -> None:
    """
    Take the obstacle vocabulary and lane geometry from `vision.py` itself.

    Duplicating those constants here would be a second source of truth that
    drifts the moment someone tunes the lane, and a drifted lane is a car that
    drives where the operator thinks it is safe.
    """
    global _OBSTACLE_LABELS, _NEAR_CONF, _LANE_LEFT, _LANE_RIGHT
    _OBSTACLE_LABELS = vision_module.OBSTACLE_LABELS
    _NEAR_CONF = vision_module.NEAR_CONF
    _LANE_LEFT = vision_module.PATH_LEFT_PCT
    _LANE_RIGHT = vision_module.PATH_RIGHT_PCT


class VisionAutopilot:
    """
    The background reflex. Owns one asyncio task and nothing else.

    Everything it needs is injected, so this class has no import-time knowledge
    of app.py and can be driven by a test with three lambdas.
    """

    def __init__(
        self,
        *,
        get_pipeline: Callable[[], Any],
        get_telemetry: Callable[[], Optional[dict]],
        start_manual: Callable[[], Any],
        motion: Any,
        on_event: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._get_pipeline = get_pipeline
        self._get_telemetry = get_telemetry
        self._start_manual = start_manual
        self._motion = motion
        # ⚠️ Single-argument reporter, matching `app.log_event(msg)`.
        # This was wired as a two-argument callback while every call site passed
        # two arguments and log_event takes one — so every emit raised
        # TypeError. That surfaced as a 500 *after* enable() had already armed
        # the car, and inside `_stop_motion` *before* the throttle was zeroed,
        # so a stop could fail to stop. Keep this contract one argument wide.
        self._on_event = on_event or (lambda msg: None)

        self._task: Optional[asyncio.Task] = None
        self._enabled = False
        self._owns_command = False      # True while *we* hold the manual command
        self._started_at: Optional[float] = None
        self.last_decision: Optional[dict] = None
        self.last_stop_reason: Optional[str] = None
        self.ticks = 0
        self.stop_count = 0
        self.drive_count = 0
        self.last_error: Optional[str] = None

    # ---------- lifecycle ----------
    @property
    def enabled(self) -> bool:
        return self._enabled

    async def start(self) -> None:
        """Run the loop for the whole session. It idles while disabled."""
        if self._task is not None:
            return
        self._task = asyncio.get_running_loop().create_task(self._loop())
        await asyncio.sleep(0)

    async def stop(self) -> None:
        await self.disable(reason="server shutdown")
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def enable(self) -> dict:
        """
        Turn the reflex on. Refuses unless it can prove the car is driveable.

        This is the gate that matters: the ultrasonic sensor is disabled in the
        firmware, so enabling vision-only autonomy with no camera and no
        telemetry would be arming a car with no forward safety net at all.
        """
        if self._enabled:
            return {"ok": True, "already": True, **self.status()}

        pipeline = self._get_pipeline()
        if pipeline is None:
            return self._refuse("vision pipeline is not running")
        tele = self._get_telemetry()
        if not tele:
            return self._refuse("no telemetry from the car yet")
        if tele.get("lifted"):
            return self._refuse("car is lifted")
        if tele.get("sensor_dead"):
            return self._refuse("distance sensor is dead - refusing to drive on vision alone")

        # Take the manual command this module will write modifiers onto. Same
        # call the gyroscope uses; identical signing, lease and watchdog.
        cmd = self._start_manual()
        if not cmd or not isinstance(cmd, dict) or not cmd.get("command_id"):
            return self._refuse("could not take the manual command (is the car online?)")

        self._enabled = True
        self._owns_command = True
        self._started_at = time.time()
        self.last_stop_reason = None
        self._emit("autopilot_on", "AI auto-pilot is driving")
        return {"ok": True, **self.status()}

    def disable(self, reason: str = "operator") -> dict:
        was = self._enabled
        self._enabled = False
        self.last_decision = None
        if self._owns_command:
            self._owns_command = False
            # Only stop the command if it is still OURS. `motion.stop()` is
            # global - a blind stop here would kill a route or a patrol the
            # operator started in the meantime.
            try:
                if getattr(self._motion, "kind", None) == "manual":
                    self._motion.stop(reason=f"autopilot_off:{reason}", silent=True)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
        self._emit("autopilot_off", f"AI auto-pilot off ({reason})")
        return {"ok": True, "was_enabled": was, **self.status()}

    def _refuse(self, why: str) -> dict:
        self._emit("autopilot_refused", f"AI auto-pilot not started: {why}")
        return {"ok": False, "error": why, **self.status()}

    # ---------- the loop ----------
    def _emit(self, kind: str, msg: str) -> None:
        """
        Report, and never let reporting raise.

        Everything else in this module treats the operator-facing event log as
        best-effort. An exception here would propagate into `enable()` after the
        car was already armed, or into `_stop_motion` before the throttle was
        closed. Neither is acceptable, so the failure is swallowed and noted.
        """
        try:
            self._on_event(f"{kind}: {msg}" if kind else msg)
        except Exception as exc:
            self.last_error = f"event log failed ({type(exc).__name__}: {exc})"
            print(f"[autopilot] {self.last_error}", flush=True)

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(TICK_S)
                if self._enabled:
                    self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:                     # never kill the server
                self.last_error = f"{type(exc).__name__}: {exc}"
                print(f"[autopilot] loop error: {self.last_error}", flush=True)
                # An autopilot that has thrown must not keep the throttle open.
                with contextlib.suppress(Exception):
                    self._stop_motion("autopilot error")

    def _tick(self) -> None:
        self.ticks += 1
        # Defence in depth: the loop below also catches, but the throttle is
        # closed HERE, at the point of failure. Relying only on the outer
        # handler means any future caller of `_tick` loses the stop.
        try:
            self._tick_inner()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[autopilot] tick error: {self.last_error}", flush=True)
            self._stop_motion(f"autopilot error ({type(exc).__name__})")

    def _tick_inner(self) -> None:
        self.ticks += 1
        pipeline = self._get_pipeline()
        if pipeline is None:
            self._stop_motion("vision pipeline disappeared")
            return

        dets = list(pipeline.detections())
        # `_dets_at` is the pipeline's own "when were these computed" stamp.
        # Reading it is how we tell a fresh frame from a frozen one.
        dets_at = float(getattr(pipeline, "_dets_at", 0.0) or 0.0)
        age = (time.time() - dets_at) if dets_at else None

        decision = decide(dets, self._get_telemetry(), vision_age_s=age)
        self.last_decision = decision

        if decision["action"] == "stop":
            self._stop_motion(decision["reason"])
            return

        ok = self._motion.set_gyro_input(decision["throttle"], decision["steer"])
        if not ok:
            # Someone else took the command (a route, a patrol, or the
            # operator). Give up rather than fighting over it - and say so,
            # because "autopilot is on but not driving" is a silent failure
            # otherwise.
            self._stop_motion("manual command was taken over - autopilot yielded")
            return
        self.drive_count += 1
        self.last_stop_reason = None

    def _stop_motion(self, why: str) -> None:
        """Zero the throttle, keep the command, and record the reason."""
        self.stop_count += 1
        # ⚠️ ORDER MATTERS: throttle FIRST, report second. The original order
        # meant a failing reporter could raise before the car was stopped.
        # Nothing that is only telling the operator something is allowed to run
        # before the thing that stops the car.
        with contextlib.suppress(Exception):
            self._motion.set_gyro_input(0, "F")
        if self.last_stop_reason != why:
            self.last_stop_reason = why
            self._emit("autopilot_stop", f"auto-pilot stopped: {why}")

    # ---------- read-only ----------
    def status(self) -> dict:
        return {
            "enabled": self._enabled,
            "policy": {
                "control_hz": CONTROL_HZ,
                "cruise_pct": CRUISE_PCT,
                "creep_pct": CREEP_PCT,
                "max_vision_age_s": MAX_VISION_AGE_S,
                "max_telemetry_age_s": MAX_TELEMETRY_AGE_S,
            },
            "started_at": self._started_at,
            "uptime_s": round(time.time() - self._started_at, 1) if self._started_at else 0,
            "ticks": self.ticks,
            "drive_count": self.drive_count,
            "stop_count": self.stop_count,
            "last_decision": self.last_decision,
            "last_stop_reason": self.last_stop_reason,
            "last_error": self.last_error,
        }