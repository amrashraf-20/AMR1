r"""
vision_autopilot.py — the driving policy, proven without a car.

Run:  .venv\Scripts\python.exe test_vision_autopilot.py
(PS 5.1: $env:PYTHONIOENCODING='utf-8' first.)

`decide()` is pure, so every branch of the reflex is checked here with nothing
connected. The class tests use fakes for the pipeline, the telemetry and the
motion controller, and assert the *seams*: that enable refuses when it cannot
prove safety, that a stale frame stops the car, and that a failing tick can
never leave the throttle open.
"""

import asyncio
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\Users\user\.gemini\antigravity\scratch\hardware_test\car-server-py")

import vision_autopilot as VA  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return ok


def section(t):
    print(f"\n{t}", flush=True)


class Det:
    """Same shape as vision.Detection: label, conf, cx, cy (0-100)."""

    def __init__(self, label, cx, cy, conf=0.9):
        self.label, self.cx, self.cy, self.conf = label, cx, cy, conf


class FakeVision:
    """Stand-in for vision.py, so the real constants get used."""

    PATH_LEFT_PCT = 25.0
    PATH_RIGHT_PCT = 75.0
    NEAR_CY = 45.0
    NEAR_CONF = 0.45
    OBSTACLE_LABELS = frozenset({"chair", "potted plant", "car", "bottle", "person", "dining table"})


VA.attach(FakeVision)


def tele(**over):
    base = {"lifted": False, "sensor_dead": False, "obstacle_blocked": False,
            "forward_blocked": False, "distance": 999}
    base.update(over)
    return base


FRESH = 0.05


def main() -> int:
    section("wiring to vision.py")
    check("lane geometry comes from vision, not a copy",
          VA._LANE_LEFT == 25.0 and VA._LANE_RIGHT == 75.0
          and VA._NEAR_CONF == FakeVision.NEAR_CONF and "chair" in VA._OBSTACLE_LABELS)
    check("autopilot is capped below full speed",
          VA.CREEP_PCT < VA.CRUISE_PCT <= 100,
          f"creep={VA.CREEP_PCT} cruise={VA.CRUISE_PCT}")

    section("decide() - the clear case")
    d = VA.decide([], tele(), vision_age_s=FRESH)
    check("no objects -> go straight at cruise",
          d["action"] == "go" and d["steer"] == "F" and d["throttle"] == VA.CRUISE_PCT, str(d))
    d = VA.decide([Det("car", 90, 30)], tele(), vision_age_s=FRESH)
    check("an object OUTSIDE the lane does not stop the car",
          d["action"] == "go" and d["steer"] == "F", str(d))
    d = VA.decide([Det("car", 90, 30, conf=0.05)], tele(), vision_age_s=FRESH)
    check("a low-confidence detection does not stop the car", d["action"] == "go", str(d))

    section("decide() - the safety gates (each must STOP)")
    for name, args, why in (
        ("stale vision", ([], tele(), 99.0), "vision stale"),
        ("vision never ran", ([], tele(), None), "vision stale"),
        ("no telemetry", ([], None, FRESH), "no telemetry"),
        ("car lifted", ([], tele(lifted=True), FRESH), "lifted"),
        ("sensor dead", ([], tele(sensor_dead=True), FRESH), "no forward safety net"),
        ("firmware obstacle", ([], tele(obstacle_blocked=True, distance=12), FRESH), "obstacle"),
        ("firmware path blocked", ([], tele(forward_blocked=True), FRESH), "path blocked"),
    ):
        dets, t, age = args
        r = VA.decide(dets, t, vision_age_s=age)
        check(f"{name} -> stop, 0 throttle",
              r["action"] == "stop" and r["throttle"] == 0 and why in r["reason"], r["reason"])

    section("decide() - steering away")
    d = VA.decide([Det("chair", 35, 40)], tele(), vision_age_s=FRESH)
    check("obstacle on the LEFT -> turn right, at a crawl",
          d["action"] == "steer" and d["steer"] == "R"
          and d["throttle"] == VA.CREEP_PCT, str(d))
    d = VA.decide([Det("chair", 65, 40)], tele(), vision_age_s=FRESH)
    check("obstacle on the RIGHT -> turn left, at a crawl",
          d["action"] == "steer" and d["steer"] == "L", str(d))
    d = VA.decide([Det("potted plant", 36, 40), Det("car", 64, 42)], tele(), vision_age_s=FRESH)
    check("with obstacles both sides, the nearest one decides",
          d["steer"] == "L" and "car" in d["reason"], d["reason"])

    section("decide() - too close to swerve")
    d = VA.decide([Det("person", 50, 75)], tele(), vision_age_s=FRESH)
    check("centred AND close -> STOP, never a coin-flip swerve",
          d["action"] == "stop" and d["throttle"] == 0, d["reason"])
    d = VA.decide([Det("person", 50, 20)], tele(), vision_age_s=FRESH)
    check("centred but FAR -> it steers instead of stopping",
          d["action"] == "steer", str(d))
    d = VA.decide([Det("person", 30, 75)], tele(), vision_age_s=FRESH)
    check("close but well off-centre -> still steers away",
          d["action"] == "steer" and d["steer"] == "R", str(d))

    section("decide() - pure, no side effects")
    dets = [Det("chair", 35, 40)]
    before = list(dets)
    VA.decide(dets, tele(), vision_age_s=FRESH)
    check("decide() did not mutate the detection list", dets == before)
    check("decide() returns the same answer twice (deterministic)",
          VA.decide([Det("chair", 35, 40)], tele(), vision_age_s=FRESH)
          == VA.decide([Det("chair", 35, 40)], tele(), vision_age_s=FRESH))

    section("the class - enable refuses without proof of safety")

    class FakePipe:
        def __init__(self, dets=(), at=None):
            self._d = list(dets)
            self._dets_at = time.time() if at is None else at

        def detections(self):
            return list(self._d)

    class FakeMotion:
        def __init__(self, accept=True, kind="manual"):
            self.kind = kind
            self.accept = accept
            self.inputs = []
            self.stops = []
            self.command_id = "cmd-1"

        def set_gyro_input(self, throttle, steer):
            if not self.accept:
                return False
            self.inputs.append((throttle, steer))
            return True

        def stop(self, reason="", silent=False):
            self.stops.append(reason)

    class Harness:
        def __init__(self, pipe, t, motion, cmd_id="cmd-1", on_event=None):
            self.pipe, self.t, self.motion = pipe, t, motion
            self.started = []

            def start_manual():
                self.started.append(True)
                return {"command_id": cmd_id} if cmd_id else {}

            self.ap = VA.VisionAutopilot(
                get_pipeline=lambda: self.pipe,
                get_telemetry=lambda: self.t,
                start_manual=start_manual,
                motion=self.motion,
                on_event=on_event,
            )

    m = FakeMotion()
    h = Harness(None, tele(), m)
    r = h.ap.enable()
    check("no pipeline -> refused", r["ok"] is False and "pipeline" in r["error"], r.get("error"))
    check("refused means nothing was started", h.started == [])

    h = Harness(FakePipe(), None, m)
    r = h.ap.enable()
    check("no telemetry -> refused", r["ok"] is False and "telemetry" in r["error"], r.get("error"))

    h = Harness(FakePipe(), tele(sensor_dead=True), m)
    r = h.ap.enable()
    check("dead distance sensor -> refused (vision alone is not a safety net)",
          r["ok"] is False and "sensor" in r["error"], r.get("error"))

    h = Harness(FakePipe(), tele(lifted=True), m)
    check("lifted car -> refused", h.ap.enable()["ok"] is False)

    h = Harness(FakePipe(), tele(), m, cmd_id=None)
    check("car not taking a command -> refused", h.ap.enable()["ok"] is False)

    section("the class - driving, and the gates while driving")
    m = FakeMotion()
    h = Harness(FakePipe([Det("chair", 35, 40)]), tele(), m)
    check("enable succeeds when everything checks out", h.ap.enable()["ok"] is True)
    check("enable took the manual command exactly once", len(h.started) == 1)
    h.ap._tick()
    check("a fresh frame with a left obstacle -> turns right",
          m.inputs and m.inputs[-1] == (VA.CREEP_PCT, "R"), str(m.inputs[-1:]))
    check("last_decision is exposed for the UI",
          h.ap.last_decision["action"] == "steer", str(h.ap.last_decision))

    m2 = FakeMotion()
    h2 = Harness(FakePipe([], at=time.time() - 99), tele(), m2)
    h2.ap.enable()
    h2.ap._tick()
    check("a STALE frame stops the car (0 throttle)",
          m2.inputs and m2.inputs[-1] == (0, "F"), str(m2.inputs[-1:]))
    check("the stop reason is recorded for the operator",
          h2.ap.last_stop_reason and "stale" in h2.ap.last_stop_reason,
          str(h2.ap.last_stop_reason))

    m3 = FakeMotion()
    h3 = Harness(FakePipe([]), tele(sensor_dead=True), m3)
    h3.ap.enable.__self__  # noqa: B018 - readability only
    h3.ap._enabled = True   # forced on to prove the tick still refuses to drive
    h3.ap._tick()
    check("a tick that cannot prove safety still stops (0 throttle)",
          m3.inputs and m3.inputs[-1] == (0, "F"), str(m3.inputs[-1:]))

    m4 = FakeMotion(accept=False)
    h4 = Harness(FakePipe([]), tele(), m4)
    h4.ap.enable()
    h4.ap._tick()
    check("a lost manual command yields instead of fighting for it",
          h4.ap.last_stop_reason and "yield" in h4.ap.last_stop_reason,
          str(h4.ap.last_stop_reason))

    section("the class - a failing tick can never leave the throttle open")
    class Exploding:
        def detections(self):
            raise RuntimeError("pipeline exploded")

    m5 = FakeMotion()
    h5 = Harness(Exploding(), tele(), m5)
    h5.ap.enable()
    try:
        h5.ap._tick()
        check("an exploding pipeline does not raise out of the tick", True)
    except RuntimeError:
        check("an exploding pipeline does not raise out of the tick", False, "it raised")
    check("but it still zeroed the throttle at the point of failure",
          m5.inputs and m5.inputs[-1] == (0, "F"), str(m5.inputs[-1:]))
    check("and the error is recorded rather than swallowed",
          h5.ap.last_error and "exploded" in h5.ap.last_error, str(h5.ap.last_error))

    section("disable only stops a command it still owns")
    m6 = FakeMotion()
    h6 = Harness(FakePipe([]), tele(), m6)
    h6.ap.enable()
    h6.ap.disable()
    check("disabling stops the manual command it holds", len(m6.stops) == 1, str(m6.stops))
    m7 = FakeMotion()
    h7 = Harness(FakePipe([]), tele(), m7)
    h7.ap.enable()
    h7.motion.kind = "patrol"          # a route took over while we were driving
    h7.ap.disable()
    check("a command that is no longer ours is NOT stopped (motion.stop is global)",
          m7.stops == [], str(m7.stops))
    check("disable still reports ok and clears enabled", h7.ap.status()["enabled"] is False)

    section("status is honest")
    h8 = Harness(FakePipe([]), tele(), FakeMotion())
    h8.ap.enable()
    s = h8.ap.status()
    check("status exposes the policy so the UI can show it",
          s["policy"]["cruise_pct"] == VA.CRUISE_PCT and s["enabled"] is True, str(s["policy"]))
    check("counters start at zero and are integers",
          isinstance(s["ticks"], int) and s["ticks"] == 0)

    section("a broken event reporter can never break control flow")
    # Regression, and a dangerous one: `on_event` was wired to app.log_event,
    # which takes ONE argument, while every call site passed two. Every
    # _on_event(...) raised TypeError -- which surfaced as a 500 *after* the car
    # was armed, and inside _stop_motion *before* the throttle was zeroed.

    def one_arg(msg):
        ONE_ARG.append(msg)

    ONE_ARG = []
    m8 = FakeMotion()
    h8b = Harness(FakePipe([]), tele(), m8, on_event=one_arg)
    check("a single-argument reporter works (the app.log_event contract)",
          h8b.ap.enable()["ok"] is True, str(h8b.ap.status().get("last_error")))
    check("and it actually received the message", ONE_ARG and "autopilot_on" in ONE_ARG[0],
          str(ONE_ARG[:1]))

    def raising(*a, **k):
        raise RuntimeError("the event log is down")

    m9 = FakeMotion()
    h9 = Harness(FakePipe([]), tele(), m9, on_event=raising)
    check("a reporter that RAISES does not break enable()",
          h9.ap.enable()["ok"] is True, str(h9.ap.status().get("last_error")))
    check("and the failure is recorded rather than swallowed",
          h9.ap.last_error and "event log" in h9.ap.last_error, str(h9.ap.last_error))
    h9.ap._tick()
    check("the throttle is still driven normally after a reporter failure",
          m9.inputs and m9.inputs[-1][0] == VA.CRUISE_PCT, str(m9.inputs[-1:]))
    check("disable() also survives a broken reporter",
          h9.ap.disable()["ok"] is True and h9.ap.enabled is False)

    # The one that matters: a stop must zero the throttle even when reporting fails.
    m10 = FakeMotion()
    h10 = Harness(FakePipe([], at=time.time() - 99), tele(), m10, on_event=raising)
    h10.ap._enabled = True
    h10.ap._tick()
    check("STOP happens even when the reporter raises (throttle first)",
          m10.inputs and m10.inputs[-1] == (0, "F"), str(m10.inputs[-1:]))
    check("and the reason is still recorded for the operator",
          h10.ap.last_stop_reason and "stale" in h10.ap.last_stop_reason,
          str(h10.ap.last_stop_reason))

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{passed}/{len(RESULTS)} passed", flush=True)
    if failed:
        print("FAILED:", flush=True)
        for n in failed:
            print(f"  - {n}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())