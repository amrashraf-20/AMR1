"""
ESP32 Car Control Adapter  —  FastAPI
============================================================
ده "الـ control adapter" المذكور في الكراسة. بيعمل:

   • MQTT مع Mosquitto  ← العربية بتتكلم معاه
   • داشبورد + WebSocket للمشغّل
   • كاميرا الموبايل
   • **قاعدة بيانات PostgreSQL** — كل حدث وأمر بيتخزن
   • **أوامر الحركة بالعقد الكامل** + lease كل 250ms

  العقد الموحّد (Unified Schema 1.0):
    العربية تبعت  →  car/<id>/event
    الأدابتر يبعت →  car/<id>/command   (JSON)
                     car/<id>/lease     (كل 250ms)
                     car/<id>/checkpoint

التشغيل:
    .venv\\Scripts\\python.exe app.py
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import os
import secrets
import socket
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import paho.mqtt.client as mqtt
from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import cmd_sign
import point_nav
import tap_blind
import tap_nav
import vision
import vision_autopilot as autopilot_mod
import visual_servo
from camera import camera
from trim import TRIM, apply_to_lease as trim_apply, describe as trim_describe
from db import db
from device_api import api as device_api
import guide as guide_mod
import history as history_mod
from guide import guide
from guide_runner import GuideRunner
from ingest import INGEST_SOURCE_ID, platform
from settings import (
    CAM_KEY,
    CAMERA_SOURCE,
    DASH_PASS,
    DASH_USER,
    MQTT_PASS_DEVICE,
    MQTT_PASS_SERVER,
    MQTT_USER_DEVICE,
    MQTT_USER_SERVER,
    mask,
)

# ================== الإعدادات ==================
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

BROKER_MODE = os.environ.get("BROKER_MODE", "external").strip().lower()
BROKER_HOST = os.environ.get("BROKER_HOST", "127.0.0.1")
BROKER_PORT = int(os.environ.get("BROKER_PORT", "1883"))

CAR_ID = os.environ.get("CAR_ID", "car1")
DEVICE_ID = os.environ.get("DEVICE_ID", "toy_car_1")
BASE = f"car/{CAR_ID}"

# مدة صلاحية الأمر (الكراسة: ٥ ثواني كحد أقصى)
COMMAND_TTL_SECONDS = int(os.environ.get("COMMAND_TTL_SECONDS", "5"))
# الفاصل بين نبضات الـ lease (الكراسة: ٢٥٠ms كحد أقصى)
LEASE_INTERVAL = float(os.environ.get("LEASE_INTERVAL", "0.25"))
# أقصى مدة للدورية الواحدة (أمان) — 60 ثانية كانت قليلة أوي لدورية
# بنقاط حقيقية (5 نقط × 5 ثواني + وقت استقرار + تأخيرات = أكتر من 60).
PATROL_MAX_SECONDS = int(os.environ.get("PATROL_MAX_SECONDS", "300"))
# الأوامر اليدوية: لو المشغّل ساب الزرار (أو الشبكة قطعت) تقف لوحدها
MANUAL_DEADMAN_SECONDS = float(os.environ.get("MANUAL_DEADMAN_SECONDS", "1.5"))
# ⚠️ المهلة الافتراضية لما حد ينادي `note_hint(True)` (توافق قديم) — من
# غير `stale_after_ms` الحقيقي مفيش رقم أدق من كده.
HINT_GRACE_SECONDS = float(os.environ.get("HINT_GRACE_SECONDS", "1.0"))

# ⚠️ سياسة التفادي في أوامر التوجيه/النقط (شوف الشرح عند `"avoid"` تحت):
#   1 = العربية تلف حوالين الحاجز وتكمّل (**الافتراضي دلوقتي** — قرار مالك
#       العربية، 2026-09-21: «عايزها تتفادى الحواجز بدل ما تقف»)
#   0 = تقف على الحاجز وتستنى (السلوك القديم — للرجوع من غير لمس الكود)
GUIDE_AVOID = 1 if os.environ.get("GUIDE_AVOID", "1").strip() not in ("0", "false", "no") else 0

HERE = Path(__file__).resolve().parent
PUBLIC = HERE / "public"
PASSWD_FILE = HERE / "broker" / "passwd"

SESSION_TTL = 12 * 3600
COOKIE_NAME = "car_session"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("amqtt").setLevel(logging.WARNING)
# لما الموبايل يقفل الصفحة فجأة، ويندوز بيرمي ConnectionResetError جوه asyncio —
# مش مشكلة حقيقية، وبنكسّت ضجيجها عشان اللوج يفضل مقروء.
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# ================== الحالة ==================
STATE: dict[str, Any] = {
    "telemetry": None,
    "last_seen": 0.0,
    "online": False,
    "events": [],
    "history": [],
    "device": {"boot_id": None, "sequence": 0, "last_event_at": None},
}

LOOP: Optional[asyncio.AbstractEventLoop] = None
SESSIONS: dict[str, float] = {}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_session() -> str:
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = time.time() + SESSION_TTL
    return token


def _token_ok(token: Optional[str]) -> bool:
    if not token:
        return False
    exp = SESSIONS.get(token)
    if exp is None:
        return False
    if exp < time.time():
        SESSIONS.pop(token, None)
        return False
    return True


def _same(a: str, b: str) -> bool:
    return secrets.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


async def require_auth(request: Request) -> None:
    if not _token_ok(request.cookies.get(COOKIE_NAME)):
        raise HTTPException(status_code=401, detail="لازم تسجّل دخول")


# ================== مفتاح كاميرا الموبايل ==================
# الموبايل مش بيسجّل دخول بالباسورد — بيدخل بمفتاح قصير من .env.
# المفتاح ده بيسمح بحاجتين بس: يبعت صورة، ويتحكم في العربية.
def _cam_ok(request: Request) -> bool:
    if _token_ok(request.cookies.get(COOKIE_NAME)):
        return True
    given = request.query_params.get("k") or ""
    return bool(given) and bool(CAM_KEY) and _same(given, CAM_KEY)


def _cam_ok_ws(ws: WebSocket) -> bool:
    if _token_ok(ws.cookies.get(COOKIE_NAME)):
        return True
    given = ws.query_params.get("k") or ""
    return bool(given) and bool(CAM_KEY) and _same(given, CAM_KEY)


def _no_cache(resp):
    """
    ⚠️⚠️ **مهم جدًا**: من غير الهيدرز دي، المتصفح بيخزّن `app.js` و`app.css`
    وبيعرض **نسخة قديمة** لحد ما حد يعمل Ctrl+F5 بنفسه.

    وده كان بيخلي كل إصلاح في الواجهة «مش شغال» عند المشغّل: إحنا نصلّح
    ونبني، وهو لسه شايف الكود القديم (زراير بتسيب نفسها، مفتاح مش موجود…)
    ويقول «برضو نفس المشكلة». فالواجهة كلها بقت **مابن تتخزّنش**.
    """
    if resp is not None:
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp


# ================= اكتشاف السيرفر (UDP) =================
#
# ⚠️ **ليه ده موجود؟** عنوان اللابتوب بيتغيّر مع كل شبكة/راوتر (حصل ٣ مرات
# في يوم واحد)، والفيرموير متفلّش على عنوان **واحد** — فأول ما الشبكة
# تتغيّر، العربية تفضل ميتة والمشغّل مش عارف السبب.
#
# الحل: العربية بتسأل الشبكة بصوت عالي («مين البروكر؟») على بورت UDP،
# والسيرفر بيرد بعنوانه الحقيقي. يعني العنوان بقى **يكتشف نفسه**.
#
# الأمان: الرد بيقول العنوان بس (مفيش أي سر)، والاتصال نفسه لسه محتاج
# يوزر/باسورد MQTT. ولو حد رد كدب، الفيرموير بيكمّل بالعنوان المحفوظ.
UDP_DISCOVERY_PORT = int(os.environ.get("UDP_DISCOVERY_PORT", "48899"))
UDP_PROBE = b"CAR-BROKER?"


def start_udp_discovery() -> None:
    """ثريد بيسمع على UDP ويرد بعنوان الأدابتر لما العربية تسأل."""
    import socket
    import threading

    def serve() -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", UDP_DISCOVERY_PORT))
        except OSError as exc:
            print(f"[udp] مش قادر أسمع على {UDP_DISCOVERY_PORT}: {exc}", flush=True)
            return
        print(f"[udp] اكتشاف السيرفر شغال على بورت {UDP_DISCOVERY_PORT}", flush=True)
        while True:
            try:
                data, addr = sock.recvfrom(256)
            except OSError:
                continue
            if data.strip() != UDP_PROBE:
                continue
            ip = primary_ip() or (all_ips() or ["127.0.0.1"])[0]
            with contextlib.suppress(OSError):
                sock.sendto(f"CAR-BROKER={ip}:{BROKER_PORT}".encode(), addr)
            print(f"[udp] العربية سألت من {addr[0]} — رديت بـ {ip}:{BROKER_PORT}",
                  flush=True)

    threading.Thread(target=serve, daemon=True, name="udp-discovery").start()


# ================== الداشبورد ==================
class ConnectionManager:
    def __init__(self) -> None:
        self.active: set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self.lock:
            self.active.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self.lock:
            self.active.discard(ws)

    async def broadcast(self, payload: dict) -> None:
        async with self.lock:
            targets = list(self.active)
        if not targets:
            return
        data = json.dumps(payload, ensure_ascii=False, default=str)
        for ws in targets:
            try:
                await ws.send_text(data)
            except Exception:
                async with self.lock:
                    self.active.discard(ws)


manager = ConnectionManager()


def log_event(msg: str) -> None:
    STATE["events"].append({"t": time.time(), "msg": msg})
    if len(STATE["events"]) > 200:
        STATE["events"] = STATE["events"][-200:]
    print(f"[car] {msg}", flush=True)
    if LOOP is not None:
        asyncio.run_coroutine_threadsafe(manager.broadcast({"type": "event", "msg": msg}), LOOP)


def _broadcast_threadsafe(payload: dict) -> None:
    if LOOP is not None:
        asyncio.run_coroutine_threadsafe(manager.broadcast(payload), LOOP)


# ================== معالجة أحداث العربية ==================
def _distance_parts(payload: dict) -> tuple[Optional[int], bool, str]:
    """
    المسافة بتيجي من الفيرموير بشكل كائن `{value, unit, valid}` (ده اللي
    العقد الموحّد بيطلبه). بنقبل الرقم المجرّد كمان عشان الإصدارات القديمة.
    يرجّع (القيمة, صالحة؟, الوحدة).
    """
    d = payload.get("distance")
    if isinstance(d, dict):
        value = d.get("value")
        unit = str(d.get("unit") or "cm")
        valid = bool(d.get("valid")) and isinstance(value, (int, float))
        return (int(value) if valid else None), valid, unit
    if isinstance(d, (int, float)) and not isinstance(d, bool):
        valid = payload.get("distance_valid")
        valid = True if valid is None else bool(valid)
        return (int(d) if valid else None), valid, str(payload.get("distance_unit") or "cm")
    return None, False, "cm"


def _note_boot(boot_id: str) -> None:
    """
    بنسجّل كل قيام للعربية (ونربطه بسبب القيام لما التيليمتري توصل).

    ⚠️ **ليه ده مهم؟** «الـ ESP بيرستر» كانت شكوى مالهاش مقياس: مفيش حاجة
    تقول للمشغّل إذا كانت المشكلة اتصلحت ولا لأ. بالعدّاد ده بيشوف:
    «٦ ريستارت في آخر ساعة» → لسه، و«صفر في آخر ساعة» → اتصلحت.
    """
    boots = STATE.setdefault("boots", [])
    if boots and boots[-1].get("boot_id") == boot_id:
        return
    boots.append({"t": time.time(), "boot_id": boot_id, "reason": None})
    if len(boots) > 200:
        del boots[:-200]


def _note_boot_reason(reason) -> None:
    """أول تيليمتري بعد القيام بتحمل سبب القيام (`reset_reason`)."""
    boots = STATE.setdefault("boots", [])
    if boots and boots[-1].get("reason") is None and reason:
        boots[-1]["reason"] = str(reason)


def _reset_stats() -> dict:
    """عدد الريستارتات في آخر ٥ دقايق / ساعة + آخر سبب."""
    boots = STATE.get("boots") or []
    now = time.time()
    last_min = [b for b in boots if now - b["t"] <= 60]
    last_hour = [b for b in boots if now - b["t"] <= 3600]
    reasons: dict[str, int] = {}
    for b in last_hour:
        r = b.get("reason") or "unknown"
        reasons[r] = reasons.get(r, 0) + 1
    last = boots[-1] if boots else None
    age = None if not last else round(now - last["t"], 1)
    return {
        "last_minute": len(last_min),
        "last_hour": len(last_hour),
        "reasons": reasons,
        "last_reason": None if not last else last.get("reason"),
        "last_age_s": age,
        # أعمار الجلسات في آخر ساعة (بترتيب زمني) — دي اللي بتوريك النمط
        "ages_s": [round(b["t"] - a["t"], 1)
                   for a, b in zip(last_hour, last_hour[1:])],
    }


def _compat_telemetry(payload: dict) -> dict:
    """
    بيدّي أسماء الحقول القديمة كمان عشان الواجهات تفضل شغالة.
    كمان بيفكّ كائن المسافة لرقم، لأن الداشبورد بيحب رقم.

    ⚠️ **ليه ده مهم؟** العقد الموحّد بيسمّي الحقول بأسماء طويلة
    (`distance` كائن · `stop_distance` · `sensor_dead` · `obstacle_blocked`
    · `uptime_ms` · `auto_avoid` · `reset_reason`)، والواجهات (الرياكت
    وcam/fpv) بتقرا الأسماء القصيرة (`dist` · `stop` · `sensorDead` ·
    `obstacle` · `uptime` · `avoid`). لما الأسماء ماتتطابقش، الداشبورد
    بيعرض «—» في المسافة والحساس والوقت — يعني المشغّل بيشوف عربية
    «مش بتتحرك» من غير أي أرقام تساعده. فبنعمل الترجمة **في مكان واحد**
    (هنا) بدل ما نصلّحها في كل واجهة لوحدها.
    """
    dist, valid, unit = _distance_parts(payload)
    uptime_ms = payload.get("uptime_ms")
    return {
        **payload,
        # المسافة (رقم + صلاحية + وحدة)
        "distance": dist,
        "distance_valid": valid,
        "distance_unit": unit,
        "dist": dist,
        # باقي الأسماء القصيرة اللي الواجهات بتقراها
        "stop": payload.get("stop_distance"),
        "blocked": payload.get("obstacle_blocked"),
        "obstacle": payload.get("obstacle_blocked"),
        "sensorDead": payload.get("sensor_dead"),
        "forwardBlocked": payload.get("forward_blocked"),
        "avoid": payload.get("auto_avoid"),
        "uptime": (round(uptime_ms / 1000.0, 1)
                   if isinstance(uptime_ms, (int, float)) else None),
        "uptime_ms": uptime_ms,
        "resetReason": payload.get("reset_reason"),
        "timeSynced": payload.get("time_synced"),
        "liftLock": False,
    }


def handle_car_event(ev: dict) -> None:
    etype = ev.get("event_type", "unknown")
    payload = ev.get("payload") or {}

    # ⚠️ تحقق قبل التخزين: حدث ناقص مينفعش يوقف الداتابيز.
    # (أي حد على البروكر يقدر ينشر على car/+/event — لازم نتحقق)
    if not ev.get("event_id") or not etype or etype == "unknown":
        log_event(f"حدث مرفوض (ناقص event_id / event_type)")
        print(f"[car] payload مرفوض: {json.dumps(ev, ensure_ascii=False)[:160]}", flush=True)
        return

    # ---- نربط أي صورة بنقطة التفتيش دي (أو نصوّر تلقائي) ----
    captured = None
    if etype == "checkpoint_confirmed":
        cp_id = str(payload.get("checkpoint_id") or "")
        ev_id = str(ev.get("event_id") or "")
        incident = payload.get("incident_id")
        route = payload.get("route_id")

        pending = PENDING_EVIDENCE.pop(cp_id, None) if cp_id else None

        if not pending and cp_id:
            # مفيش صورة مستنية → **تصوير تلقائي** من الكاميرا وقت الوصول
            pending = auto_capture(cp_id, incident, route, "checkpoint")

        if pending and ev_id:
            refs = ev.get("evidence_refs") or []
            if isinstance(refs, list) and pending not in refs:
                refs.append(pending)
            ev["evidence_refs"] = refs
            db.link_evidence_to_event(pending, ev_id)
            log_event(f"🔗 الصورة {pending} اتربطت بنقطة {cp_id}")
        captured = pending

    db.save_event(ev)

    # ---- نبلّغ مشغّل الدورية (لو فيه دورية شغالة) ----
    if etype == "checkpoint_confirmed":
        checkpoint_run.note(str(payload.get("checkpoint_id") or ""), True,
                            evidence_id=captured, payload=payload)
    elif etype == "checkpoint_failed":
        checkpoint_run.note(str(payload.get("checkpoint_id") or ""), False,
                            reason=str(payload.get("reason") or "failed"),
                            payload=payload)

    # ---- نبعت للمنصة (صاحب الويب) ----
    # ده بيحصل في ثريد منفصل في ingest.py، فلو منصتهم واقعة العربية
    # بتكمل عادي والأحداث بتتكدس في طابور محدود.
    if platform.enabled:
        sim = bool(payload.get("sim"))
        platform.submit_event(ev, provenance="fixture" if sim else None)

    STATE["device"]["boot_id"] = ev.get("boot_id") or STATE["device"]["boot_id"]
    STATE["device"]["sequence"] = ev.get("sequence") or STATE["device"]["sequence"]
    STATE["device"]["last_event_at"] = ev.get("occurred_at")
    STATE["last_seen"] = time.time()
    STATE["online"] = True

    if etype in ("telemetry", "telemetry_update"):
        prev = STATE["telemetry"]
        _note_boot_reason(payload.get("reset_reason"))
        # بنخزّن نسخة فيها المسافة كرقم **والأسماء القصيرة** — عشان
        # `/api/state` (اللي الداشبورد بيسحب منه كل ثانيتين) يدّي نفس
        # الشكل اللي الـ WebSocket بيدّيه. من غير كده، أول ما الـ WS يقطع
        # الأرقام كلها تبقى «—» والعربية تبان ميتة وهي شغالة.
        dist, dist_ok, _unit = _distance_parts(payload)
        STATE["telemetry"] = _compat_telemetry(payload)
        STATE["telemetry"]["distance"] = dist
        STATE["telemetry"]["distance_valid"] = dist_ok
        payload = STATE["telemetry"]

        STATE["history"].append({"t": time.time(), "dist": dist,
                                 "state": payload.get("state")})
        if len(STATE["history"]) > 600:
            STATE["history"] = STATE["history"][-600:]

        if prev:
            if prev.get("state") != payload.get("state"):
                log_event(f"الحركة: {prev.get('state')} ← {payload.get('state')}")
            if not prev.get("obstacle_blocked") and payload.get("obstacle_blocked"):
                log_event(f"حيطة على {dist} سم — فرملة")
            if prev.get("lifted") != payload.get("lifted"):
                log_event("العربية اتشالت من الأرض" if payload.get("lifted")
                          else "العربية رجعت على الأرض")
        else:
            shown = f"{dist} سم" if dist_ok else "قراءة مش صالحة"
            log_event(f"العربية ظهرت — مسافة {shown}")

        _broadcast_threadsafe({"type": "telemetry", **payload})

    elif etype == "command_ack":
        cmd_id = payload.get("command_id", "")
        ack_state = payload.get("state", "")
        reason = payload.get("reason")
        log_event(f"إقرار الأمر {cmd_id}: {ack_state}" + (f" — {reason}" if reason else ""))
        db.save_ack(cmd_id, ack_state, reason)
        motion.note_ack(payload)
        # ⚠️ `stopped` / `rejected` = العربية **هي** اللي قفلت الأمر
        # (حاجز، رفع، صلاحية، ريستارت). لازم نفضّي حالتنا كمان، وإلا
        # هنفضل نأمّر على أمر ميّت (شوف MotionController.forget).
        if ack_state in ("stopped", "rejected"):
            motion.forget(cmd_id, reason=f"{ack_state}/{reason or '—'}")
        _broadcast_threadsafe({"type": "ack", "command_id": cmd_id,
                               "state": ack_state, "reason": reason})

    elif etype == "checkpoint_confirmed":
        cp_id = payload.get("checkpoint_id", "")
        log_event(f"نقطة تفتيش مؤكدة: {cp_id}")
        db.save_checkpoint({
            "checkpoint_id": cp_id,
            "route_id": payload.get("route_id"),
            "incident_id": payload.get("incident_id"),
            "device_id": DEVICE_ID,
            "method": payload.get("method"),
            "occurred_at": ev.get("occurred_at"),
            "event_id": ev.get("event_id"),
        })
        _broadcast_threadsafe({"type": "checkpoint", "checkpoint_id": cp_id,
                               "route_id": payload.get("route_id"),
                               "occurred_at": ev.get("occurred_at")})

    elif etype == "device_boot":
        _note_boot(str(ev.get("boot_id") or ""))
        stats = _reset_stats()
        log_event(f"arabia qaemat - boot_id {ev.get('boot_id')} "
                  f"(restarts last hour: {stats['last_hour']})")
        # Reconnect zero-state - the SERVER half of the runaway fix (2026-09-29).
        #
        # The firmware now calls forceZeroState() on every (re)connect, which
        # clears activeCmd - so any lease that follows is dropped by the
        # activeCmd[0]==0 guard in handleLeaseJson. That alone stops the CAR.
        #
        # But the SERVER was still convinced it was driving: the lease loop
        # kept publishing and the console showed "manual active" while the
        # car sat still. That is a lie on the screen AND it keeps the command
        # alive so a gyro/manual keepalive can renew stale motion. So we tear
        # the command down locally too.
        # ==================================================================
        if motion.command_id:
            log_event(f"STOP: car rebooted while command {motion.command_id} "
                      f"was live - immediate zero-state stop")
            motion.stop(reason="device_boot")
            _broadcast_threadsafe({"type": "motion", **motion.status()})
        _broadcast_threadsafe({"type": "boot", "boot_id": ev.get("boot_id"),
                               "resets": stats})

    elif etype == "lift_detected":
        log_event("العربية اتشالت من الأرض" if payload.get("lifted") else "العربية رجعت على الأرض")

    else:
        log_event(f"حدث: {etype}")


# ================== MQTT ==================
monitor = mqtt.Client(
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    # ⚠️ قابل للتغيير من البيئة: نسختين على نفس البروكر **بنفس المعرّف**
    # بيطردوا بعض (البروكر بيقفل الأولى). ده بيخلّي «تجربة تشغيل» على
    # بورت تاني آمنة تمامًا من غير ما تلمس النسخة الشغالة.
    client_id=os.environ.get("MQTT_CLIENT_ID", "car-adapter"),
    clean_session=True,
)
monitor.username_pw_set(MQTT_USER_SERVER, MQTT_PASS_SERVER)
monitor.reconnect_delay_set(min_delay=1, max_delay=5)


def _on_connect(client, userdata, flags, reason_code, properties=None):
    code = int(getattr(reason_code, "value", reason_code))
    if code == 0:
        client.subscribe("car/+/event")
        client.subscribe("car/+/status")
        print("  ✔ الأدابتر اتصل بالبروكر", flush=True)
    else:
        print(f"  ✘ البروكر رفض الأدابتر — كود {code}", flush=True)


def _on_message(client, userdata, msg):
    topic = msg.topic
    text = msg.payload.decode("utf-8", errors="replace")

    if topic.endswith("/status"):
        online = text.strip() == "online"
        STATE["online"] = online
        log_event("العربية اتصلت" if online else "العربية فصلت")
        _broadcast_threadsafe({"type": "status", "online": online})
        return

    if not topic.endswith("/event"):
        return

    try:
        ev = json.loads(text)
    except ValueError:
        return

    if isinstance(ev, dict) and ev.get("event_type"):
        handle_car_event(ev)


monitor.on_connect = _on_connect
monitor.on_message = _on_message


def publish(topic_suffix: str, payload: str) -> None:
    monitor.publish(f"{BASE}/{topic_suffix}", payload, qos=0)


def publish_checkpoint(checkpoint_id: str, route_id, incident_id,
                       leg_seconds, confirmed_by=None) -> dict:
    """
    كل رسالة نقطة تفتيش بتخرج من هنا — عشان التوقيع مايتنسيش في مكان.

    ⚠️ الفيرموير بيتحقق من التوقيع قبل ما يمشي، فلو الرسالة خرجت من
    غير توقيع هتترفض بـ `bad_signature`.

    ⚠️ `leg_seconds` **عدد صحيح** — الفيرموير بيقراها `long`، فبنقرّب
    هنا عشان اللي بيتوقّع يطابق اللي بيتقرا بالظبط.
    """
    leg_seconds = int(round(float(leg_seconds or 0)))
    body = {
        "checkpoint_id": checkpoint_id,
        "route_id": route_id,
        "incident_id": incident_id,
        "leg_seconds": leg_seconds,
    }
    if confirmed_by:
        body["confirmed_by"] = confirmed_by
    body["signature"] = cmd_sign.sign_checkpoint(checkpoint_id, route_id,
                                                 leg_seconds)
    publish("checkpoint", json.dumps(body))
    return body


# ================== متحكم الحركة (Command + Lease) ==================
# ================== السيرفو البصري (إضافة مستقلة) ==================
#
# ⚠️ حالة السيرفو لوحده هنا. مفيش أي حاجة بتشتغل لوحدها: الحلقة
# مابتبدأش غير لما المشغّل يدوس «يلا» من الصفحة. وفيه وضع تجريبي
# (`dry`) بيحسب كل حاجة من غير ما يبعت أي أمر للعربية.
POINT_VISUAL: dict = {
    "servo": None, "task": None, "dry": False,
    "cmd": None, "error": None, "running": False, "command_id": None,
}


# ================== معايرة «دوس على الكاميرا» ==================
#
# ⚠️ المعايرة بتتحفظ على **الجهاز** (مش في الكود): ارتفاع الكاميرا وميلها
# بيخصّوا تركيب الموبايل على العربية، وده بيتغيّر من عربية لعربية ومن
# موبايل لموبايل. فبنحفظها في ملف JSON جانب السيرفر، و`sync_export` مابياخدهوش
# مع نسخة التسليم (زي `.env`).
#
# و«الوضع البصري» (`mode: visual`) مابيحتاجش أي معايرة أصلًا — دي مجرد
# أرقام بتتعلّمها مرة وبتزوّد **العدّاد بالمتر** والشبكة على الشاشة.
TAP_CONFIG_FILE = HERE / "tap_config.json"
TAP_DEFAULTS: dict = {
    "height_m": 0.12, "pitch_deg": 20.0,
    "hfov_deg": tap_nav.DEFAULT_HFOV_DEG,
    "mode": tap_nav.MODE_VISUAL,
    "stop_band_px": tap_nav.STOP_BAND_PX,
    "metric_ok": False,          # المعايرة متأكدين منها (الشبكة طابقت الأرض)
    "taps": [],                  # نقط المعايرة المحفوظة
    "horizon_v": None,           # صف الأفق اللي المشغّل داس عليه
}


def _tap_config_load() -> dict:
    cfg = dict(TAP_DEFAULTS)
    try:
        raw = json.loads(TAP_CONFIG_FILE.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for k in TAP_DEFAULTS:
                if k in raw:
                    cfg[k] = raw[k]
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        print(f"[tap] ملف المعايرة باظ ({exc}) — بنشتغل بالافتراضي", flush=True)
    return cfg


def _tap_config_save(cfg: dict) -> None:
    try:
        TAP_CONFIG_FILE.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[tap] مش قادر أحفظ المعايرة: {exc}", flush=True)


TAP_CONFIG: dict = _tap_config_load()


def _tap_model_from(cfg: dict, *, img_w: int = 640, img_h: int = 480):
    """موديل الكاميرا من الإعدادات المحفوظة."""
    return tap_nav.CamModel(
        height_m=float(cfg.get("height_m", TAP_DEFAULTS["height_m"])),
        pitch_deg=float(cfg.get("pitch_deg", TAP_DEFAULTS["pitch_deg"])),
        hfov_deg=float(cfg.get("hfov_deg", TAP_DEFAULTS["hfov_deg"])),
        img_w=int(img_w), img_h=int(img_h))


def _tap_public_config(*, img_w: int = 640, img_h: int = 480) -> dict:
    """الإعدادات + طبقة الرسم — دي اللي الواجهة بترسم بيها الشبكة."""
    model = _tap_model_from(TAP_CONFIG, img_w=img_w, img_h=img_h)
    return {
        "ok": True,
        "mode": TAP_CONFIG.get("mode", tap_nav.MODE_VISUAL),
        "metric_ok": bool(TAP_CONFIG.get("metric_ok")),
        "height_m": model.height_m,
        "pitch_deg": model.pitch_deg,
        "hfov_deg": model.hfov_deg,
        "stop_band_px": int(TAP_CONFIG.get("stop_band_px",
                                           tap_nav.STOP_BAND_PX)),
        "taps": TAP_CONFIG.get("taps", []),
        "horizon_v": TAP_CONFIG.get("horizon_v"),
        "residual_m": tap_nav.verify_model(model, TAP_CONFIG.get("taps", [])),
        "overlay": tap_nav.grid_overlay(
            model, stop_band_px=int(TAP_CONFIG.get("stop_band_px",
                                                   tap_nav.STOP_BAND_PX))),
    }


def _cam_frame_size() -> tuple[int, int]:
    """مقاس آخر صورة جاية من الموبايل (وإلا 640×480)."""
    jpeg = camera.jpeg()
    gray = visual_servo.gray_from_jpeg(jpeg) if jpeg else None
    if gray is None:
        return 640, 480
    return int(gray.shape[1]), int(gray.shape[0])


class MotionController:
    """
    كل حركة لازم تعدي من هنا:
      1) أمر بالعقد الكامل (command_id / expires_at / approved_by)
      2) lease كل 250ms
      3) وقف فوري لو المشغّل ساب أو الدورية خلصت
    """

    # ⚠️ الأوضاع اللي **التوجيه** هو اللي بيقودها: مفيش توجيه طازة =
    # مفيش حركة. البوابة على الـ lease نفسه — يعني آخر نقطة قبل ما
    # الأمر يطلع للعربية، مهما كان اللي تحت.
    #
    # `point` (السيرفو البصري) و `patrol` (الدورية) و `manual` (المشغّل)
    # مش جواها: دول مابنبدأوش من حلقة السحب أبدًا، كلهم فعل صريح من
    # المشغّل على الداشبورد المحلي.
    HINT_DRIVEN = ("guide",)

    # ⚠️ الاتجاهات المسموحة — أي حاجة تانية **مرفوضة** مش بتتحوّل لـ "F"
    # (شوف `start`).
    VALID_DIRECTIONS = ("F", "B", "L", "R")

    # ⚠️⚠️ سرعة التحكم اليدوي — **محدودة عن قصد** (2026-09-20).
    #
    # قبل كده كان الوضع اليدوي بياخد سرعة العربية نفسها (`motorSpeed`) —
    # واللي كان **255 (الكامل)**. النتيجة: أول دوسة = انطلاق بأقصى سرعة،
    # والعربية تجري وماينفعش تتحكم فيها (شكوى المشغّل بالحرف).
    #
    # دلوقتي: الـ lease بيحمل سرعة **صريحة** (نسبة من سقف التوجيه 170)
    # وبتتحدد قبل ما توصل للعربية — فمهما كان `motorSpeed`، التحكم اليدوي
    # مابيعديش `MANUAL_SPEED_MAX`.
    MANUAL_SPEED_DEFAULT = int(os.environ.get("MANUAL_SPEED", "110"))
    MANUAL_SPEED_MAX = int(os.environ.get("MANUAL_SPEED_MAX", "150"))
    # ⚠️⚠️ وضع الجايرو (مايل الموبايل) — سقف **أبطأ** من السهم عن قصد
    # (2026-09-29): السهم = إصبع ثابت على الزرار، بس الجايرو = ركلة
    # ٥٠ درجة وبعدها ركلة ٥٠ درجة عكسها. نفس السقف (150) بيخلّي
    # ده يملّي الغرفة. فبنعمل سقف مستقل أبطأ (110 = نفس الافتراضي
    # بتاع اليدوي) ومتقيّد **داخل** MANUAL_SPEED_MAX — يعني حتى لو
    # حد رفع `GYRO_SPEED_MAX` غلط في `.env` مافيش طريق يعدّي سقف
    # اليدوي.
    GYRO_SPEED_MAX = max(0, min(int(os.environ.get("GYRO_SPEED_MAX", "110")),
                                MANUAL_SPEED_MAX))
    # ⚠️ `throttle` بيجي **كنسبة من ١٠٠** مش كرقم موتور خام: جهاز
    # الموبايل ماعرفش سقف العربية، فيبقى السقف المصداقي عندنا
    # (رقم واحد في مكان واحد، مش اتنين).
    GYRO_THROTTLE_MAX = 100
    # ⚠️⚠️ [2026-09-29] عزم التحويل الأدنى (skid-steer torque floor).
    # نفس القيمة اللي في الفيرموير (`STEER_SPEED_MIN`) — اتنين بنسخة
    # واحدة عمدًا، انظر `_manual_lease_speed`.
    STEER_SPEED_MIN = int(os.environ.get("STEER_SPEED_MIN", "145"))
    # ⚠️ حاجز الأمان: المشي المستمر مايفضلش شغال للأبد. لو المشغّل ساب
    # الداشبورد مفتوح وداس «مشي مستمر»، العربية مش هتفضل ماشية ١٠ دقايق.
    MANUAL_MAX_SECONDS = float(os.environ.get("MANUAL_MAX_SECONDS", "60"))
    # صعود ناعم لأول نص ثانية (تيار بدء أقل + مايفة شوطة في الأول)
    SOFT_START_S = float(os.environ.get("SOFT_START_S", "0.4"))
    SOFT_START_SPEED = int(os.environ.get("SOFT_START_SPEED", "60"))

    def __init__(self) -> None:
        self.command_id: Optional[str] = None
        self.route_id: Optional[str] = None
        self.incident_id: Optional[str] = None
        self.approved_by: Optional[str] = None
        self.kind: Optional[str] = None
        self.started_at: float = 0.0
        self.leases_sent = 0
        self.task: Optional[asyncio.Task] = None
        self.last_ack: Optional[dict] = None
        # dead-man: آخر مرة المشغّل أكّد إنه لسه دايس
        self.keepalive_at: float = 0.0
        # ---- وضع التوجيه ----
        # ⚠️ نفس مسار الأمر الواحد: توجيه الموبايل مابيفتحش مسار
        # أوامر تاني، هو بس بيغيّر الاتجاه والسرعة **جوه الـ lease**.
        # يعني كل الضمانات (أمر مصرّح + انتهاء الـ lease + الإنترلوك)
        # فضلت زي ما هي.
        self.guide_dir: str = "F"
        self.guide_speed: int = 0
        # ⚠️ سرعة التحكم اليدوي (بتتبعت صريحة في الـ lease — مش -1).
        # شوف `_manual_lease_speed`.
        self.manual_speed: int = self.MANUAL_SPEED_DEFAULT
        # ⚠️ التوجيه الطازة **مش علامة بتتثبّت** — دي **مهلة** بتتحسب من
        # `hint_age_ms` بتاع التوجيه نفسه (شوف `note_hint`). الافتراضي
        # `0.0` = منتهية = فشل مقفول: أمر جديد مامعاهوش توجيه طازة بعد.
        self.hint_fresh: bool = False
        self.hint_deadline: float = 0.0

    def note_hint(self, hint=None) -> None:
        """
        «التوجيه اللي إحنا ماشيين عليه لسه طازة؟» — و**لحد إمتا؟**

        ⚠️ ده كان باگ حقيقي: الطزاجة كانت **علامة بوليان** بتتثبّت مرة
        واحدة وتفضل `True` لحد ما حد يفتكر يصفّرها. يعني لو حلقة السحب
        **اتعطلت** (poll واخد ١.٥ ثانية، أو `accept` معلّق، أو اللوب
        مشغول) ماحدش بينادي `decide()` خالص، والـ lease بيفضل يجدّد
        **نفس الحركة** والعربية ماشية على توجيه عمره أكبر من
        `stale_after_ms`. طبقة «التوجيه القديم = وقف» مابتفصلش في الوقت
        ده لأنها جوه `decide()` — و`decide()` مش شغالة أصلًا.

        فبدل العلامة، بنسجّل **آخر لحظة يبقى فيها التوجيه صالح**، وبنعيد
        حسابها من `hint_age_ms` كل ما ييجي توجيه جديد. والطزاجة بتتسأل
        (`hint_valid`) عند **آخر نقطة قبل العربية** — يعني الـ lease نفسه
        بيكفّ عن تجديد الحركة لما المهلة تخلص، حتى لو `decide()` مااشتغلتش.

        المدخلات:
          • `guide.Hint` → المهلة = `stale_after_ms - hint_age_ms`
            (نفس قاعدة الكراسة بالحرف: `hint_age_ms <= stale_after_ms`).
          • `True` / `False` → توافق مع اللي بينده من غير التوجيه نفسه.
          • `None` / أي حاجة تانية → **منتهية** (فشل مقفول).
        """
        if hint is None or hint is False:
            self.clear_hint()
            return
        if hint is True:
            self.hint_fresh = True
            self.hint_deadline = time.monotonic() + HINT_GRACE_SECONDS
            return
        stale_after = getattr(hint, "stale_after_ms", None)
        age = getattr(hint, "backend_age_ms", None)
        if not isinstance(stale_after, (int, float)) or not isinstance(age, (int, float)):
            self.clear_hint()          # توجيه مش مفهوم = فشل مقفول
            return
        # ⚠️ `max(0, ...)` مقصودة: لو العمر عدّى الحد أصلًا، المهلة =
        # دلوقتي (يعني «منتهي») — مش بنصلّحه لرقم أكبر.
        left_ms = max(0.0, float(stale_after) - float(age))
        self.hint_fresh = True
        self.hint_deadline = time.monotonic() + left_ms / 1000.0

    @property
    def hint_valid(self) -> bool:
        """الطزاجة **بتتحسب دلوقتي** من المهلة، مش علامة اتثبّتت قبل كده."""
        return bool(self.hint_fresh) and time.monotonic() < self.hint_deadline

    def clear_hint(self) -> None:
        """التوجيه مابقاش صالح — تستخدمها كل مسارات الوقف."""
        self.hint_fresh = False
        self.hint_deadline = 0.0

    def status(self) -> dict:
        if not self.command_id:
            return {"active": False}
        return {
            "active": True,
            "command_id": self.command_id,
            "kind": self.kind,
            "route_id": self.route_id,
            "incident_id": self.incident_id,
            "approved_by": self.approved_by,
            "age_seconds": round(time.time() - self.started_at, 1),
            "leases_sent": self.leases_sent,
            "last_ack": self.last_ack,
        }

    def keepalive(self) -> None:
        """المشغّل لسه دايس — نجدد المهلة."""
        self.keepalive_at = time.time()

    def _manual_lease_speed(self) -> int:
        """
        السرعة اللي بتتبعت في lease التحكم اليدوي (نسبة من سقف التوجيه).

        بتتحدد من: سرعة المشغّل المختارة (`manual_speed`) → مقيّدة بالسقف
        → ومعاها **صعود ناعم** في أول `SOFT_START_S` ثانية.
        """
        # ⚠️⚠️ [2026-09-29] `0 or DEFAULT` كان **باگ حقيقي**: صفر باين
        # falsy في Python، فـ `set_manual_speed(0)` كان بيرجع للافتراضي
        # (165!) مش صفر. يعني `speed=0` — اللي معناه «قف» — كان
        # مستحيل تتعبّر عنه من هذه المسار، و«قف» كان بيتحوّل لـ «امشي
        # بسرعة كاملة». الجايرو مااتأثّرش لأن مساره بيبعت `motion_stop`
        # بدل `throttle=0`، بس ده لغم مدفون جنب safety-critical code.
        # الحل: `manual_speed` دايمًا number (بيتعملّه init للافتراضي)،
        # فبنقرأه صريح — صفر صفر.
        want = max(0, min(int(self.manual_speed), self.MANUAL_SPEED_MAX))

        # ==================================================================
        # [2026-09-29] STEERING TORQUE FLOOR - heavy 4WD skid-steer fix.
        # Deliberately duplicated from the firmware (STEER_SPEED_MIN in
        # turnSpeed()). Both, on purpose:
        #
        # The lease is HMAC-signed and the speed is inside the signature.
        # A floor on the server alone would therefore SIGN a number that
        # the firmware then overwrites - the signature would attest to a
        # value that is not the one actually executed. Applying it in both
        # places means the server sends the final signed number, and the
        # firmware still enforces it as a backstop for any other source.
        #
        # NOTE the "want > 0" guard: speed=0 means STOP (gyro inside its
        # deadzone, or the operator let go). A floor here would have turned
        # "stop" into "creep at 145" - the car moving when told to stop.
        # ==================================================================
        if want > 0 and self.guide_dir in ("L", "R") and want < self.STEER_SPEED_MIN:
            want = self.STEER_SPEED_MIN
        elapsed = time.time() - (self.started_at or time.time())
        if self.SOFT_START_S <= 0 or elapsed >= self.SOFT_START_S:
            return want
        k = max(0.0, elapsed / self.SOFT_START_S)
        return int(round(self.SOFT_START_SPEED
                         + (want - self.SOFT_START_SPEED) * k))

    def set_manual_speed(self, speed) -> int:
        """سرعة التحكم اليدوي من الواجهة (بتتقيّد بالسقف فورًا)."""
        try:
            v = int(speed)
        except (TypeError, ValueError):
            return self.manual_speed or self.MANUAL_SPEED_DEFAULT
        self.manual_speed = max(0, min(v, self.MANUAL_SPEED_MAX))
        return self.manual_speed

    def set_gyro_input(self, throttle, steer) -> bool:
        """
        ذراع الموبايل (الجايرو) — غير السرعة与会جاه **جوه نفس الأمر**.

        ⚠️ ليه مش أمر جديد كل ٢٥ هرتز؟ لأن الأمر هو اللي فيه التوقيع
        (`cmd_sign`) والصلاحية والانتهاء. لو بدأنا أمرًا جديد مع كل
        قراءة جايرو: ٢٥ أمر/ثانية، ٢٥ تسجيل في `db.save_command`، و
        ٢٥ نشرة broker — مقابل **صفر** فايدة. الجايرو بيدّعي على نفس
        الـ lease اللي اليدوي بيستعمله (نفس `guide_dir` + نفس
        `manual_speed`)، فكل الضمانات فضلت زي ما هي: أمر مصرّح +
        مهلة + إنترلوك + deadman.

        المدخلات: `throttle` نسبة ٠..١٠٠، `steer` ∈ F/B/L/R.
        الرجوع: `False` = اترفض (أمر مش يدوي، أو مدخلات مش مفهومة)
        → المشغّل يقف على طول. **فشل مقفول** زي كل مسارات الوقف.
        """
        if self.command_id and self.kind != "manual":
            return False                      # مش هنخطف دورية ولا توجيه
        try:
            pct = float(throttle)
        except (TypeError, ValueError):
            return False
        if not (0.0 <= pct <= float(self.GYRO_THROTTLE_MAX)):
            # نسبة برّه ٠..١٠٠ = رفض outright. **ماحنقوّهاش** على ١٠٠:
            # مدخل واحد غلط معناه «أقصى سرعة» على عربية حقيقية.
            return False
        raw = str(steer or "").strip().upper()
        if raw not in self.VALID_DIRECTIONS:
            return False

        # ⚠️ السرعة بتتقوّم من **نسبة** → رقم موتور، وبتتقيد بالسقف
        # الخاص بالجايرو (اللي هو جوّه سقف اليدوي). `set_manual_speed`
        # بيطبّق الحد الثاني — فمافيش طريق يعدّي أي منهم من هنا.
        self.manual_speed = self.set_manual_speed(
            int(round(pct / 100.0 * self.GYRO_SPEED_MAX)))
        self.guide_dir = raw
        self.keepalive()                      # الجايرو نفسه = نبضة
        return True

    def start(self, kind: str, route_id: str, incident_id: Optional[str],
              approved_by: str, direction: str = "F") -> dict:
        # ⚠️ **اتجاه مش مفهوم ≠ قدّم.** زمان كان أي اتجاه غلط (أو فاضي)
        # بيتحوّل لـ `'F'`، والفيرموير بينفّذ اتجاه الأمر لحظة استلامه →
        # `motion_start` من غير اتجاه كان بيطلّع **نبضة قدّم بالسرعة
        # الكاملة**. دلوقتي بنرفض ونقول السبب، ومننشرش أي حاجة.
        raw = str(direction or "").strip().upper()
        if raw not in self.VALID_DIRECTIONS:
            with contextlib.suppress(Exception):
                log_event(f"⛔ اتجاه غير مفهوم ({direction!r}) — مفيش حركة")
            return {"ok": False, "error": "bad_direction", "command_id": None}
        direction = raw

        self.stop(reason="replaced", silent=True)

        now = utcnow()
        expires = now + timedelta(seconds=COMMAND_TTL_SECONDS)
        cmd = {
            "command_id": f"cmd-{uuid.uuid4().hex[:8]}",
            "device_id": DEVICE_ID,
            "allowed_route_id": route_id,
            "incident_id": incident_id,
            "issuer": "backend_system",
            "approved_by": approved_by,
            "issued_at": iso(now),
            "expires_at": iso(expires),
            # حقل اختياري (امتداد للعقد) — مش مطلوب في الكراسة
            "direction": direction,
            # ⚠️⚠️ **`avoid` = سياسة التفادي عند الحاجز** (اتغيّرت 2026-09-21).
            #
            # القيمة **1** = العربية **تلف حوالين** الحاجز وتكمّل ناحية هدفها،
            # والقيمة **0** = تقف وتستنى (السلوك القديم).
            #
            # ليه اتغيّرت؟ القديم كان بيبعت `0` لأوامر التوجيه/النقط بحجّة
            # إن «العربية اللي تقرر تسيب نقطة لوحدها بتكدب على التقرير».
            # الحجّة دي **مابقاش لها لازمة**: في التوجيه بالكاميرا العربية
            # أصلًا **مابتـ reportش نقط** (شوف ملف صاحب المنصة §١.١ بند ٦) —
            # تقرير النقط شغل الستيرر بالكاميرا. والتفادي مابيبلغش أي نقطة:
            # بيلف حوالين الحاجز ويرجع **لنفس التوجيه**، والكاميرا بتصحّح.
            #
            # والنتيجة العملية للقديم كانت: العربية تقف على أي كرسي →
            # الأمر يتقفل (`stopActive("obstacle")`) → والأدابتر يفتح أمر
            # جديد على أول poll → تقف/تمشي/تقف… والدورية مابتوصلش.
            #
            # ⚠️ **مهم:** ده **مش** تعديل في عقد صاحب المنصة — `avoid`
            # حقل محلي من عندنا (مش موجود في جسم §١.٥ عنده خالص)، ومعناه
            # «الفيرموير يتصرّف لوحده إزاي عند الحاجز». رجّع السلوك القديم
            # بـ `GUIDE_AVOID=0` في `.env` من غير أي لمس للكود.
            #
            # ⚠️ وفي الحالتين الفرملة **إجبارية** في الفيرموير: `avoid=0`
            # معناها «اقف واستنى»، مش «اكتسح الحاجز» (شوف `hardware_test.ino`
            # بند «الحاجز: القاعدة الحاكمة»).
            "avoid": GUIDE_AVOID if kind in ("guide", "point") else 1,
            # ⚠️ علامة **وضع التوجيه** (امتداد للعقد، زي `direction`):
            # الفيرموير بيستخدمها عشان يرفض أي **نقطة تفتيش** توصل في
            # الوضع ده — لأن النقط ساعتها ملك صفحة التوجيه، وقبول نقطة
            # بيمشي العربية لوحدها وبعدها بيجمّد التوجيه (`cpHold`).
            # (`avoid` مش كفاية: المشغّل يقدر يقفل التفادي من الداشبورد.)
            "guide": 1 if kind in ("guide", "point") else 0,
        }
        # ⚠️ التوقيع: الفيرموير بيتحقق منه **قبل أي حركة**.
        # بيربط الأمر بالمصدر حتى لو حد وصل للبروكر نفسه.
        cmd["signature"] = cmd_sign.sign_command(cmd)

        self.command_id = cmd["command_id"]
        self.route_id = route_id
        self.incident_id = incident_id
        self.approved_by = approved_by
        self.kind = kind
        self.started_at = time.time()
        self.leases_sent = 0
        self.last_ack = None
        self.keepalive_at = time.time()
        self.guide_dir = direction
        self.guide_speed = 0
        # ⚠️ أمر جديد = **مفيش** توجيه طازة بعد. لو ده وضع توجيه، أول
        # نبضة lease هتوقف لو التوجيه ماجاش. (فشل مقفول)
        self.clear_hint()

        db.save_command(cmd)
        db.audit(approved_by, f"{kind}_started", cmd)

        # ---- السجل المحلي للساعة (مستقل عن postgres) ----
        # `db.ok` is False on this machine, so the hourly history cannot be
        # read back out of the commands table. Counting commands from the
        # telemetry stream instead would be wrong — it cannot tell an
        # operator command from a firmware auto-avoid. So we count them here,
        # at the one place a command is actually issued. `start()` carries no
        # speed (the lease loop sets it per pulse), so speed stays NULL and
        # the hourly view reads speed off the telemetry samples instead.
        history_mod.history.record_command(cmd, direction=direction)

        print(f"[cmd] {kind} → {cmd['command_id']} "
              f"(route={route_id}, incident={incident_id}, by={approved_by})", flush=True)
        log_event(f"أمر {kind} اتصدر: {cmd['command_id']}")

        publish("command", json.dumps(cmd))

        if LOOP is not None:
            self.task = LOOP.create_task(self._lease_loop())

        return cmd

    async def _lease_loop(self) -> None:
        try:
            while self.command_id:
                # ---- Dead-man للمشغّل اليدوي ----
                # لو المشغّل ساب الزرار أو الشبكة قطعت → مفيش keepalive
                # → نوقف فورًا. ده اللي يخليها "لازم تفضل دايس".
                if self.kind == "manual":
                    idle = time.time() - self.keepalive_at
                    if idle > MANUAL_DEADMAN_SECONDS:
                        print(f"[cmd] المشغّل ساب الزرار ({idle:.1f}ث) — وقف", flush=True)
                        log_event("المشغّل ساب الزرار — وقف")
                        self.stop(reason="deadman")
                        return
                    # ⚠️⚠️ حاجز «جريان» (2026-09-20): «المشي المستمر» خلّى
                    # الدوسة الواحدة تسيب العربية ماشية، ومع سرعة عالية
                    # العربية تجري وماينفعش تتحكم فيها. فبنحدّ المدة:
                    # أقصى مشوار يدوي متواصل = MANUAL_MAX_SECONDS وبعدها
                    # وقف صريح (المشغّل يدوس تاني لو عايز).
                    ran = time.time() - (self.started_at or time.time())
                    if self.MANUAL_MAX_SECONDS > 0 and ran > self.MANUAL_MAX_SECONDS:
                        print(f"[cmd] مشوار يدوي طويل ({ran:.0f}ث) — وقف أمان",
                              flush=True)
                        log_event(f"⏱️ وقف أمان: مشي يدوي {ran:.0f}ث متواصل")
                        self.stop(reason="manual_max_duration")
                        return

                # ⚠️ في وضع التوجيه بنبعت الاتجاه والسرعة جوه الـ lease
                # نفسه. النبضة محدودة بـ LEASE_INTERVAL (250ms) — يعني
                # أقصى حركة ممكنة قبل ما نعيد القياس.
                # ⚠️ دايماً بنبعت direction و speed (speed = -1 معناها
                # "مفيش سقف من الـ lease") عشان نص التوقيع يبقى واحد
                # ومفيش أي لبس في الطرفين.
                payload = {
                    "command_id": self.command_id,
                    "status": "continue",
                    # ⚠️ الأوضاع اللي إحنا بنقودها بتحط **اتجاهها** في الـ
                    # lease. الوضع اليدوي كان بيتبعت له "F" دايمًا → أسهم
                    # اليمين/الشمال في الداشبورد مكانت بتعمل أي حاجة
                    # (الأمر بيحمل الاتجاه، وأول lease بترجّعه F).
                    "direction": (self.guide_dir
                                  if self.kind in ("guide", "point", "manual")
                                  else "F"),
                    # ⚠️ في الوضع اليدوي، الـ lease كان بيبعت `speed = -1`
                # ومعناها «مفيش سقف من الـ lease» → الفيرموير ياخد
                # `motorSpeed` (اللي كان 255) → **أعلى سرعة فورًا**،
                # والعربية تجري وماينفعش تتحكم فيها. (شكوى المشغّل
                # 2026-09-20: «المواتير بتشتغل على أعلى سرعة ومعرفش
                # أتحكم فيها».) دلوقتي بنبعت سرعة **صريحة ومحدودة**
                # **للوضع اليدوي بس**. الدورية بتاخد `-1` عشان
                # الفيرموير يتحكم في السرعة عن طريق نقط التفتيش.
                "speed": (int(self.guide_speed)
                          if self.kind in ("guide", "point")
                          else self._manual_lease_speed()
                          if self.kind == "manual"
                          else -1),
                }

                # ---- ⛔ البوابة: مفيش توجيه طازة = مفيش حركة ----
                # ⚠️ دي آخر نقطة قبل ما الأمر يطلع للعربية. الحلقة دي
                # مابتستناش حد يفتكر يوقفها: لو التوجيه بقى `null` أو
                # عمره مجهول أو `hint_age_ms > stale_after_ms`، بنوقف
                # **من هنا** — وقف صريح للعربية (مش speed = 0 وسكوت،
                # ومش `-1` اللي معناها «مفيش سقف» = العربية تجري).
                if self.kind in self.HINT_DRIVEN and not self.hint_valid:
                    print(f"[cmd] ⛔ مفيش توجيه طازة — وقف "
                          f"{self.command_id} ({self.kind})", flush=True)
                    log_event("⛔ مفيش توجيه طازة — وقف العربية")
                    self.stop(reason="no_fresh_hint")
                    return

                # ---- ترميم الانحراف (شوف trim.py) ----
                payload["direction"], payload["speed"] = trim_apply(
                    payload["direction"], payload["speed"], kind=self.kind)

                payload["signature"] = cmd_sign.sign_lease(
                    payload["command_id"], payload["status"],
                    payload["direction"], payload["speed"])

                publish("lease", json.dumps(payload))
                self.leases_sent += 1
                db.bump_lease(self.command_id)

                if self.kind == "patrol" and (time.time() - self.started_at) > PATROL_MAX_SECONDS:
                    print("[cmd] الدورية وصلت للحد الأقصى — وقف", flush=True)
                    log_event("الدورية خلصت (الحد الأقصى) — وقف")
                    self.stop(reason="patrol_max_duration")
                    return

                await asyncio.sleep(LEASE_INTERVAL)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            print(f"[cmd] خطأ في الـ lease: {exc}", flush=True)

    def stop(self, reason: str = "operator_stop", silent: bool = False) -> None:
        cid = self.command_id
        if cid:
            stop_payload = {
                "command_id": cid, "status": "stop",
                "direction": "F", "speed": -1,
            }
            stop_payload["signature"] = cmd_sign.sign_lease(
                cid, "stop", "F", -1)
            publish("lease", json.dumps(stop_payload))
            db.mark_stopped(cid)
            db.audit(self.approved_by or "system", "motion_stopped",
                     {"command_id": cid, "reason": reason})
            if not silent:
                print(f"[cmd] وقف {cid} — {reason}", flush=True)
                log_event(f"وقف الأمر {cid} — {reason}")

        self.command_id = None
        self.kind = None

    def note_ack(self, payload: dict) -> None:
        self.last_ack = payload

    def forget(self, command_id: Optional[str] = None, *, reason: str = "") -> bool:
        """
        العربية قفلت الأمر **بنفسها** — حاجز (`avoid=0`)، رفع، انتهاء
        صلاحية، أو ريستارت للبورد.

        ⚠️ بعد كده الفيرموير بيرمي أي lease جاي (`activeCmd` فاضي
        عنده). فلو سيبنا الحالة عندنا مفتوحة:
          • حلقة السحب بتفضل تأمّر على **أمر ميّت** والعربية مش بتتحرك
          • والمنصة بتشوف `driving` والعربية واقفة (كذب غير مقصود)
          • ومفيش إعادة إصدار نظيفة غير لما حاجة تانية تقفل الأمر —
            يعني العربية بتقف لحد ما التوجيه نفسه يقطع ويرجع

        بنفضّي محليًا **من غير** ما نبعت lease وقف: الأمر ميت أصلًا.
        """
        if not self.command_id:
            return False
        # ⚠️ إقرار أمر **قديم** (اتلغى واستبدل) مايقفلش الأمر الحالي.
        if command_id and str(command_id) != self.command_id:
            return False
        print(f"[cmd] ↩️ العربية قفلت الأمر {self.command_id} بنفسها "
              f"({reason or '—'}) — بنفضّي الحالة", flush=True)
        log_event(f"↩️ العربية قفلت الأمر بنفسها ({reason or '—'}) — "
                  "الحالة اتنضّفت، والتوجيه الجاي هيبدأ أمر جديد")
        self.command_id = None
        self.kind = None
        self.guide_speed = 0
        self.clear_hint()
        return True


motion = MotionController()


# ================== مشغّل نقاط التفتيش ==================
class CheckpointRun:
    """
    بياخد قائمة نقاط من الباك إند ويمشيها **بالترتيب**:

        النقطة ١ → العربية تمشي → توصل → 📸 صورة → النقطة ٢ → ...

    و**لو نقطة وقعت** (مش قادرة تمشي / اتأخرت / اترفعت / الحساس بايظ)
    بينط للي بعدها ويكمّل — مش بيوقف الدورية كلها.

    كل الصور بتتصور من كاميرا الموبايل المثبّتة على العربية،
    **وقت الوصول** مش وقت استلام الأمر.
    """

    MAX_POINTS = 12
    MAX_LEG = 15.0          # ثواني — أكتر من كده العربية هتسيب المكان
    MIN_TIMEOUT = 5.0

    def __init__(self) -> None:
        self.active = False
        self.route_id: Optional[str] = None
        self.incident_id: Optional[str] = None
        self.approved_by: Optional[str] = None
        self.command_id: Optional[str] = None
        self.points: list[dict] = []
        self.index = -1
        self.started_at = 0.0
        self.finished_at: Optional[float] = None
        self.error: Optional[str] = None
        self.task: Optional[asyncio.Task] = None
        self._wake: Optional[asyncio.Event] = None
        self._result: Optional[dict] = None

    # ---------- من ثريد الـ MQTT ----------
    def note(self, checkpoint_id: str, ok: bool, *, reason: str = "",
             evidence_id: Optional[str] = None, payload: Optional[dict] = None) -> None:
        """العربية ردّت على نقطة — بنصحّي الحلقة المنتظرة."""
        if not self.active or self._wake is None or LOOP is None:
            return
        if checkpoint_id and self.index >= 0:
            cur = self.points[self.index].get("checkpoint_id")
            if cur and checkpoint_id != cur:
                return          # رد على نقطة قديمة — بنتجاهله
        self._result = {
            "ok": ok, "reason": reason or ("arrived" if ok else "failed"),
            "evidence_id": evidence_id, "payload": payload or {},
        }
        with contextlib.suppress(RuntimeError):
            LOOP.call_soon_threadsafe(self._wake.set)

    # ---------- التشغيل ----------
    async def start(self, points: list[dict], *, route_id: str = "route_a",
                    incident_id: Optional[str] = None,
                    approved_by: Optional[str] = None, move: bool = True,
                    default_leg: float = 5.0, point_timeout: float = 20.0,
                    retry_failed: bool = False) -> dict:
        if self.active:
            return {"ok": False, "error": "فيه دورية شغالة بالفعل"}

        clean: list[dict] = []
        for raw in points[: self.MAX_POINTS]:
            if isinstance(raw, str):
                raw = {"checkpoint_id": raw}
            cid = str((raw or {}).get("checkpoint_id") or "").strip()
            if not cid:
                continue
            try:
                leg = float(raw.get("leg_seconds", default_leg))
            except (TypeError, ValueError):
                leg = default_leg
            clean.append({
                "checkpoint_id": cid,
                "leg_seconds": round(max(0.0, min(self.MAX_LEG, leg)), 1),
                "status": "pending",
                "reason": None,
                "evidence_id": None,
                "took_ms": None,
            })
        if not clean:
            return {"ok": False, "error": "مفيش نقاط صالحة"}

        if move and not STATE["online"]:
            return {"ok": False, "error": "العربية مش متصلة"}
        if move and motion.command_id:
            return {"ok": False, "error": "فيه حركة شغالة بالفعل"}

        self.active = True
        self.route_id = route_id
        self.incident_id = incident_id
        self.approved_by = approved_by or DASH_USER
        self.command_id = None
        self.points = clean
        self.index = -1
        self.error = None
        self.started_at = time.time()
        self.finished_at = None
        self._result = None
        self._wake = asyncio.Event()

        self.task = asyncio.create_task(
            self._run(move=move, point_timeout=max(self.MIN_TIMEOUT, point_timeout),
                      retry_failed=retry_failed))
        log_event(f"🗺️ الباك إند طلب {len(clean)} نقطة: " +
                  " → ".join(p["checkpoint_id"] for p in clean))
        _broadcast_threadsafe({"type": "checkpoint_run", **self.status()})
        return {"ok": True, "points": len(clean), "command_id": None}

    async def _wait_command_accepted(self, timeout: float = 8.0) -> bool:
        """
        ⚠️ لازم نستنى العربية تقبل أمر الحركة **قبل** ما نبعت أول نقطة.
        من غير كده النقطة بتوصل قبل الأمر → العربية مش ماشية →
        `no_active_command` والدورية كلها بتقع من غير سبب حقيقي.
        """
        cmd_id = self.command_id
        if not cmd_id:
            return False
        end = time.time() + timeout
        while time.time() < end:
            ack = motion.last_ack or {}
            if (ack.get("command_id") == cmd_id
                    and ack.get("state") in ("accepted", "active", "stopped", "rejected")):
                return ack.get("state") in ("accepted", "active")
            await asyncio.sleep(0.1)
        return False

    async def _run(self, *, move: bool, point_timeout: float,
                   retry_failed: bool) -> None:
        try:
            if move:
                motion.last_ack = None
                cmd = motion.start("patrol", self.route_id or "route_a",
                                   self.incident_id, self.approved_by or DASH_USER)
                self.command_id = cmd["command_id"]
                log_event(f"🚗 دورية النقاط بدأت — أمر {self.command_id}")

                if not await self._wait_command_accepted():
                    # ⚠️ العربية رفضت الأمر (مرفوعة / حساس بايظ / مفيش
                    # وقت / MQTT مش واصل). بنوقف **فوراً** ونلغي كل
                    # النقط — بدل ما نبعتهم ويفشلوا كلهم في ثانية
                    # والداشبورد يبيّن إنها «خلصت» وهي ماتحركتش.
                    self.error = "command_rejected"
                    log_event("⛔ العربية رفضت أمر الدورية — وقف فوري "
                              "(مفيش فايدة نبعت النقط)")
                    for p in self.points:
                        p["status"] = "cancelled"
                        p["reason"] = "command_rejected"
                    motion.stop(reason="command_rejected", silent=True)
                    return

                # ⚠️ نسيب العربية تستعد بعد قبول الأمر — الـ ESP32
                # بياخد شوية يعالج الأمر ويبدأ الـ lease loop.
                await asyncio.sleep(0.5)

            for i, point in enumerate(self.points):
                if not self.active:
                    point["status"] = "cancelled"
                    continue

                self.index = i
                point["status"] = "running"
                cid = point["checkpoint_id"]
                log_event(f"➡️ النقطة {i + 1}/{len(self.points)}: {cid} "
                          f"({point['leg_seconds']}ث)")

                if move and not motion.command_id:
                    point["status"] = "failed"
                    point["reason"] = "command_lost"
                    log_event(f"⚠️ الأمر اتلغى — باقي النقاط هتتخطى")
                    continue

                attempts = 2 if retry_failed else 1
                for attempt in range(attempts):
                    self._result = None
                    self._wake = asyncio.Event()

                    publish_checkpoint(cid, self.route_id, self.incident_id,
                                       point["leg_seconds"], self.approved_by)

                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=point_timeout)
                    except asyncio.TimeoutError:
                        self._result = {"ok": False, "reason": "timeout"}

                    if self._result and self._result.get("ok"):
                        break
                    if attempt + 1 < attempts and self.active:
                        log_event(f"↻ إعادة محاولة النقطة {cid}")

                result = self._result or {"ok": False, "reason": "timeout"}
                point["reason"] = None if result["ok"] else result.get("reason")
                point["evidence_id"] = result.get("evidence_id")
                point["took_ms"] = (result.get("payload") or {}).get("took_ms")
                point["status"] = "arrived" if result["ok"] else "skipped"

                if result["ok"]:
                    shot = result.get("evidence_id")
                    log_event(f"✅ النقطة {cid} وصلت" +
                              (f" — 📸 {shot}" if shot else " — ⚠️ مفيش صورة"))
                else:
                    # ⚠️ دي الحالة اللي طلبها: ننط للي بعدها
                    log_event(f"⏭️ النقطة {cid} وقعت ({result.get('reason')}) "
                              f"— بننط للي بعدها")

                _broadcast_threadsafe({"type": "checkpoint_run", **self.status()})

            self.index = len(self.points) - 1
        except asyncio.CancelledError:
            self.error = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            log_event(f"❌ الدورية وقعت: {self.error}")
        finally:
            if move and motion.command_id:
                motion.stop(reason="route_done")
            self.active = False
            self.finished_at = time.time()
            done = sum(1 for p in self.points if p["status"] == "arrived")
            shots = sum(1 for p in self.points if p.get("evidence_id"))
            log_event(f"🏁 الدورية خلصت: {done}/{len(self.points)} نقطة، "
                      f"{shots} صورة")
            _broadcast_threadsafe({"type": "checkpoint_run", **self.status()})

    def stop(self, reason: str = "operator_stop") -> bool:
        if not self.active:
            return False
        self.active = False
        if self.task and not self.task.done():
            with contextlib.suppress(Exception):
                self.task.cancel()
        if motion.command_id:
            motion.stop(reason=reason)
        log_event("⛔ الدورية اتوقفت")
        return True

    def status(self) -> dict:
        done = sum(1 for p in self.points if p["status"] == "arrived")
        skipped = sum(1 for p in self.points if p["status"] == "skipped")
        return {
            "active": self.active,
            "route_id": self.route_id,
            "incident_id": self.incident_id,
            "approved_by": self.approved_by,
            "command_id": self.command_id,
            "index": self.index + 1,
            "total": len(self.points),
            "arrived": done,
            "skipped": skipped,
            "photos": sum(1 for p in self.points if p.get("evidence_id")),
            "ageSeconds": round(time.time() - self.started_at, 1) if self.started_at else 0,
            "error": self.error,
            "points": [
                {k: p[k] for k in ("checkpoint_id", "leg_seconds", "status",
                                   "reason", "evidence_id", "took_ms")}
                for p in self.points
            ],
        }


checkpoint_run = CheckpointRun()


# ================== 🧭 التوجيه من الموبايل (Guide) ==================
def hint_is_fresh() -> bool:
    """
    التوجيه اللي إحنا ماشيين عليه **دلوقتي** طازة؟

    ⚠️ دي مش قاعدة طزاجة تانية: `Hint.stale()` هي نفسها بتاعة الكراسة
    (`hint_age_ms > stale_after_ms`). الدالة دي بس بتسأل «آخر توجيه
    وصلنا لسه صالح؟» — و`GuideState.put_hint` بتشيل التوجيه خالص لما
    مايكونش فيه توجيه، أو لما عمره يبقى مجهول. يعني `None` هنا معناها
    **مفيش توجيه** = وقوف.
    """
    h = guide.hint
    return h is not None and not h.stale()


class MotionCarLink:
    """
    بيوصّل `GuideRunner` بالعربية الحقيقية.

    ⚠️ مهم: **مفيش مسار أوامر تاني**. التوجيه بيغيّر الاتجاه والسرعة
    جوه نفس عقد الأمر + الـ lease اللي موجود. يعني:
      • مفيش أمر مصرّح  → مش هتتحرك
      • الـ lease قطع    → الفيرموير بيوقف لوحده خلال ثانية
      • الإنترلوك (رفع/حساس/حاجز) لسه أولوية عليا في الفيرموير

    وكمان: في وضع التوجيه العربية **ماتتفاداش** الحواجز (avoid=0)
    لأن التفادي معناه إنها تقرر تسيب نقطة لوحدها — وده مش شغلها.
    """

    KIND = "guide"

    def telemetry(self) -> dict:
        return dict(STATE["telemetry"] or {})

    def active_command(self):
        return motion.command_id

    def apply(self, action) -> None:
        # ⚠️ **المشغّل وقّف** = مايتفتحش أمر جديد مهما كان التوجيه طازة.
        # ده كان باگ حقيقي: `POST /api/guide/stop` كان بيقفل الأمر الحالي
        # وبس، والحلقة (اللي لسه شغالة) بتسحب تاني بعد ٢٠٠ms، تلاقي
        # الأمر لسه `active` عند المنصة، و`motion.command_id is None`
        # (الحالة **الافتراضية**) → تفتح أمر جديد وتكمّل. يعني وقف
        # المشغّل كان بيتلغى لوحده في أقل من ربع ثانية. القفل ده بيتشال
        # بـ **فعل صريح** من المشغّل (`POST /api/guide/resume`) وبس.
        if guide_runner.suspended:
            return
        # ⚠️ **مفيش توجيه طازة = مفيش حركة.** دي حادثة الحيطة نفسها:
        # أمر `active` و`hint = NULL` والعربية مشيت. `decide` بيمنع ده
        # أصلاً، ودي تأكيد تاني على نقطة التنفيذ (دفاع في العمق) — لو
        # أي حد نادى `apply` بحركة من غير توجيه، بنوقف.
        if action.moving and not hint_is_fresh():
            self.stop("no_fresh_hint")
            return

        # ⚠️ لو فيه **دورية** شغالة (من الداشبورد)، التوجيه بيسكت خالص.
        # من غير كده الاتنين هيتخانقوا على نفس العربية: التوجيه بيفتح
        # أمر جديد ويلغي أمر الدورية → النقط كلها تقع.
        if checkpoint_run.active:
            motion.guide_speed = 0
            return
        if motion.command_id is None or motion.kind != self.KIND:
            # ⚠️ **بنمرّر اتجاه النبضة الحالية جوه الأمر نفسه** — مش "F"
            # الافتراضي. الفيرموير بينفّذ اتجاه الأمر **لحظة استلامه**،
            # قبل ما أول lease يوصل. يعني أمر افتراضي بـ F = العربية
            # بتاخد نبضة «قدّم بالسرعة الكاملة» (leaseSpeed = -1 يعني
            # مفيش سقف) في **كل** مرة نبدأ أو نعيد أمر — وإحنا عايزينها
            # تلف في مكانها. ده كان بيخلي مسارها قوس/دايرة: لفّة، نبضة
            # قدّم، لفّة، نبضة قدّم… (شوف test_no_hint_stop.py §٦).
            motion.start(self.KIND, "route_guide", None, "guide_page",
                         direction=action.direction)
            log_event(f"🧭 التوجيه بدأ أمر {motion.command_id} "
                      f"({action.direction})")
        # ⚠️ الترتيب مهم: `start()` بتصفّر صلاحية التوجيه (فشل مقفول)،
        # فلازم نقول «اللي إحنا ماشيين عليه صالح» **بعدها**. من غير كده
        # أول نبضة lease بعد أي أمر جديد هتوقف العربية قبل ما تمشي أصلًا.
        # ⚠️ وبنمرّر **التوجيه نفسه** مش `True`: المهلة بتتحسب من
        # `stale_after_ms - hint_age_ms` (شوف `note_hint`)، فالـ lease
        # بيكفّ عن تجديد الحركة لوحده لما التوجيه يقدم — حتى لو حلقة
        # السحب اتعطلت وماحدش نادى `decide()` تاني.
        motion.note_hint(guide.hint)
        motion.guide_dir = action.direction
        motion.guide_speed = max(0, min(int(action.speed), guide_mod.MAX_SPEED))

    def stop(self, reason: str) -> None:
        # ⚠️ بنقول للمتحكم إن التوجيه مابقاش طازة **قبل** أي حاجة:
        # حتى لو الـ lease لسه شغال، أول نبضة جاية بتوقف.
        motion.note_hint(False)
        motion.guide_speed = 0
        if motion.kind == self.KIND and motion.command_id:
            motion.stop(reason=f"guide_{reason}", silent=True)


car_link = MotionCarLink()
guide_runner = GuideRunner(device_api, car_link, state=guide)


# ================== ⛔ حلقة السحب مابنبدأش حركة أبدًا ==================
#
# ⚠️ **أهم قاعدة في الملف ده** (حادثة الحيطة):
#
#     أمر `state` = active  و `hint` = null   →   **وقوف**.
#
# مش «أكمّل لوحدي» ومش «أستنى» — وقوف. ده حرفيًا اللي في الكراسة §2:
#
#     Drive on hint.to_target while hint_age_ms <= hint.stale_after_ms.
#     Stop otherwise. hint is null until guiding starts, so "no hint" is
#     the normal state before a run and it must mean STOPPED.
#
# **اللي كان بيحصل:** كان فيه هنا «دورية أوتوماتيك» — لو الباك إند بعت
# مسار فيه نقط ومن غير توجيه، كنا بنبدأ `checkpoint_run` وندي العربية
# أمر `patrol` والـ lease بتاعه `speed = -1`، و`-1` في الفيرموير معناها
# **«مفيش سقف» = العربية تجري**. النتيجة: العربية مشيت والسيرفر مابعتش
# ولا إشارة (`hint = NULL`)، والأدابتر في نفس اللحظة بيقول `idle` لأن
# قرار التوجيه كان صح (وقف) — بس مسار تاني كان بيقود لوحده. **اتشال.**
#
# اللي لسه بيقود من غير توجيه (المشغّل اليدوي / الدورية من الداشبورد /
# السيرفو البصري) **مابنبدأهاش من هنا**: كلها محتاجة فعل صريح من
# المشغّل على الداشبورد المحلي، وكلها بتسيب حلقة السحب تقرا وتبلّغ.
#
# يعني: **حلقة السحب بتقرا وتبلّغ بس. مابنبدأش حركة من أمر المنصة.**
# الحركة الوحيدة اللي أمر المنصة يقدر يعملها هي اللي جواها `hint` طازة.


# ================== بروكر مدمج (اختياري) ==================
def embedded_broker_config() -> dict:
    from pwdlib import PasswordHash

    hasher = PasswordHash.recommended()
    PASSWD_FILE.parent.mkdir(exist_ok=True)
    PASSWD_FILE.write_text(
        f"{MQTT_USER_SERVER}:{hasher.hash(MQTT_PASS_SERVER)}\n"
        f"{MQTT_USER_DEVICE}:{hasher.hash(MQTT_PASS_DEVICE)}\n",
        encoding="utf-8",
    )
    return {
        "listeners": {"default": {"type": "tcp", "bind": f"{HOST}:{BROKER_PORT}",
                                  "max_connections": 50}},
        "sys_interval": 0,
        "timeout_disconnect_delay": 2,
        "plugins": {"amqtt.plugins.authentication.FileAuthPlugin":
                    {"password_file": str(PASSWD_FILE)}},
        "topic_check": {"enabled": False},
    }


# ================== أدوات الشبكة ==================
def primary_ip() -> Optional[str]:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 53))
        return s.getsockname()[0]
    except Exception:
        return None
    finally:
        s.close()


def all_ips() -> list[str]:
    ips: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass
    return ips


def print_banner(ip: Optional[str], broker_running: str) -> None:
    """
    ⚠️ ممنوع نطبع أي باسورد أو توكن هنا.
    الشاشة دي بتتسجّل وبتتعرض في العروض — بنطبع مكان السر وبس.
    """
    broker_ip = ip or (all_ips() or ["192.168.x.x"])[0]
    # ⚠️ البروتوكول الحقيقي: لو HTTPS=0 السيرفر بيشتغل HTTP — وطبع
    # "https" ساعتها بيدّي لينك مش بيفتح (اتشاف في تجربة تشغيل).
    scheme = "https" if os.environ.get("HTTPS", "1") != "0" else "http"
    print()
    print("  ============================================================")
    print(f"  البروكر : {broker_running}")
    print(f"  الداتابيز: PostgreSQL — {'متصل' if db.ok else 'مش متاحة'}")
    print("  ============================================================")
    print()
    print("  الداشبورد:")
    print(f"      >>>  {scheme}://{broker_ip}:{PORT}   <<<")
    print(f"      دخول : {DASH_USER}  —  الباسورد في .env (DASH_PASS)")
    print()
    print("  الكاميرا:")
    if CAMERA_SOURCE == "phone":
        print("      المصدر: كاميرا الموبايل بس 📱")
    elif CAMERA_SOURCE == "laptop":
        print("      المصدر: كاميرا اللابتوب بس 💻")
    else:
        print("      المصدر: الموبايل لو شغال، وإلا اللابتوب")
    print(f"      كاميرا الموبايل : {scheme}://{broker_ip}:{PORT}/cam?k={CAM_KEY}")
    print(f"      دوس على الحتة   : {scheme}://{broker_ip}:{PORT}/go?k={CAM_KEY}")
    print(f"      التفرج + تحكم  : {scheme}://{broker_ip}:{PORT}/fpv")
    # ⚠️ الرؤية: مكتبش الـ M وده اختياري (شوف vision.py) — لو الـ
    # weights مش موجودة، التحسين بيشتغل والكشف لأ.
    w = vision.find_weights()
    print(f"      رؤية الحاسوب   : {'✔ YOLOv8n جاهز' if w else '⚠ مفيش yolov8n.pt — كشف مفعّلش'}")
    print(f"      وصف المشهد    : POST /api/vision/describe")
    print()
    print("  منصة الويب (صاحب الويب):")
    if platform.enabled:
        print(f"      ✔ شغالة → {platform.base}   (المصدر {INGEST_SOURCE_ID})")
        print(f"      🧭 التوجيه: بنسحب كل {guide_mod.POLL_MS}ms "
              f"من /api/devices/commands")
        print("      " + ("🔏 بنوقّع الطلبات (HMAC)"
                          if device_api.signing else
                          "⚠️ التوقيع مقفول (DEVICE_SIGN=0)"))
    else:
        print(f"      ✘ متوقفة — {platform.error}")
        print("      ظبّط INGEST_BASE / INGEST_TOKEN / INGEST_CA في .env")
    print()
    print("  ------------------------------------------------------------")
    print("  حط السطور دي في كود الـ ESP32 (hardware_test.ino):")
    print(f'      const char* MQTT_HOST = "{broker_ip}";')
    print(f'      const int   MQTT_PORT = {BROKER_PORT};')
    print(f'      const char* MQTT_USER = "{MQTT_USER_DEVICE}";')
    print("      const char* MQTT_PASS = (من secrets.h — مش في الكود)");
    print("  ------------------------------------------------------------")
    print()
    print("  مستني العربية...")


# ================== FastAPI ==================
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global LOOP
    LOOP = asyncio.get_running_loop()

    db.start()
    camera.configure(CAMERA_SOURCE)
    camera.start()
    # ⚠️ الرؤية (تحسين + YOLO) — إضافة مستقلة (شوف `vision.py`).
    # `VisionPipeline` بياخد فريماته من `camera` بس **بيقرا منه بس**،
    # فحتى لو YOLO علِق أو اتعطّل، `/video` والتايب والقيادة كلهم بيفضلوا
    # شغالين. ومفيش تحميل موديل هنا — ده كسول جوه أول استدلال.
    vision_pipe.start()
    platform.start()
    guide_runner.start_task()
    # السجل المحلي: 1 Hz + حذف كل اللي عدّى 24 ساعة. مستقل عن postgres عمدًا.
    await history_mod.history.start(lambda: STATE["telemetry"], guide.status)
    # AI Auto-Pilot: المهمة بتشتغل طول Sesshon بس **تعمل حاجة والـ toggle مقفول**.
    await autopilot.start()

    broker = None
    if BROKER_MODE == "embedded":
        from amqtt.broker import Broker

        broker = Broker(embedded_broker_config())
        await broker.start()
        running = f"مدمج (amqtt) على البورت {BROKER_PORT}"
    else:
        running = f"Mosquitto على {BROKER_HOST}:{BROKER_PORT}"

    monitor.connect_async(BROKER_HOST, BROKER_PORT, keepalive=30)
    monitor.loop_start()

    # ⚠️ العربية بتسأل «مين البروكر؟» على UDP — من غيره أي تغيير في عنوان
    # اللابتوب بيخلي العربية ميتة (شوف `start_udp_discovery`).
    start_udp_discovery()

    print_banner(primary_ip(), running)

    try:
        yield
    finally:
        motion.stop(reason="shutdown", silent=True)
        vision_pipe.stop()
        camera.stop()
        platform.stop()
        await guide_runner.stop_task()
        await autopilot.stop()
        await history_mod.history.stop()
        db.stop()
        monitor.loop_stop()
        with contextlib.suppress(Exception):
            monitor.disconnect()
        if broker is not None:
            await broker.stop()


app = FastAPI(title="ESP32 Car Control Adapter", lifespan=lifespan)


# ⚠️⚠️ التخزين المؤقت للواجهة: من غيره المتصفح بيعرض `app.js` **قديم**
# والمشغّل يقول «برضو نفس المشكلة» وإحنا فعلًا صلّحنا وبَنينا. (شوف `_no_cache`.)
@app.middleware("http")
async def _ui_no_cache(request: Request, call_next):
    resp = await call_next(request)
    p = request.url.path
    if p == "/" or p.startswith("/app") or p.endswith((".js", ".css", ".html")):
        return _no_cache(resp)
    return resp


# ⚠️ تطبيق الواجهة (React) مبني في `public/app` — لازم يتحمّل كملفات
# ساكنة، وإلا المتصفح يطلب `app.js` ويرجع 404 (اتشاف فعلًا).
if (PUBLIC / "app").exists():
    app.mount("/app", StaticFiles(directory=str(PUBLIC / "app"), html=True),
              name="app")


@app.get("/")
async def index():
    return FileResponse(PUBLIC / "index.html")


@app.get("/fpv")
async def fpv():
    return FileResponse(PUBLIC / "fpv.html")


@app.get("/cam")
async def cam_page():
    return FileResponse(PUBLIC / "cam.html")


# ================== «دوس على الحتة» (إضافة مستقلة) ==================
#
# ⚠️ كل الحتة دي **إضافة** ومش بتلمس أي حاجة موجودة:
#   • صفحة جديدة لوحدها (/go) — cam.html زي ما هي
#   • مسار حساب واحد (/api/point/measure) — دالة خالص من point_nav.py
#   • مفيش أي حاجة بتشتغل لوحدها: مفيش ثريد، مفيش حركة، مفيش أوامر
#   • محتاجة مفتاح الكاميرا زي باقي مسارات الموبايل
#
# الحساب نفسه في point_nav.py ومعاه اختباراته (test_point_nav.py).
@app.get("/go")
async def go_page():
    return FileResponse(PUBLIC / "go.html")


# ================== «الجاوسيان سبيلير» (إضافة مستقلة) ==================
#
# ⚠️ إضافة لوحدها — مفيش أي حاجة هنا بتلمس كود موجود:
#   • الكود كله (الصفحة + العارض + الاختبارات) في فولدر gaussian-splatting/
#     اللي جنب car-server-py — يعني **بره السيرفر خالص**.
#   • المشهد بيترسم جوه المتصفح بـ WebGL2: مفيش صور بتترفع هنا ولا API جديد.
#   • لو الفولدر مش موجود (زي نسخة التسليم to-github)، بنرجّع رسالة عربية
#     واضحة بدل ما السيرفر يقع أو العميل يشوف خطأ خام.
SPLAT_DIR = HERE.parent / "gaussian-splatting"

if SPLAT_DIR.is_dir():
    app.mount("/splat", StaticFiles(directory=str(SPLAT_DIR), html=True), name="splat")
else:
    @app.get("/splat")
    async def splat_missing():
        return Response(
            "<!doctype html><html lang='ar' dir='rtl'><meta charset='utf-8'>"
            "<body style='background:#0d1117;color:#e6edf3;font-family:system-ui;"
            "text-align:center;padding:60px 20px;line-height:2'>"
            "<h2>🧊 عارض الجاوسيان سبيلير مش موجود في النسخة دي</h2>"
            f"<p>الفولدر <code>{SPLAT_DIR}</code> مش موجود.</p>"
            "<p>انسخ فولدر <code>gaussian-splatting/</code> جنب <code>car-server-py</code> "
            "وهيشتغل على طول.</p></body></html>",
            status_code=404,
            media_type="text/html; charset=utf-8",
        )


@app.post("/api/point/measure")
async def api_point_measure(request: Request):
    """بكسل من دوسة على الشاشة → مكان على الأرض (متر + اتجاه)."""
    if not _cam_ok(request):
        raise HTTPException(status_code=401, detail="لازم مفتاح الكاميرا")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="مفيش JSON")

    def num(key: str) -> float:
        try:
            return float(data.get(key))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"قيمة غلط: {key}")

    try:
        calib = point_nav.Calib(
            height_m=num("height_m"),
            pitch_deg=num("pitch_deg"),
            hfov_deg=num("hfov_deg") if data.get("hfov_deg") is not None
                     else point_nav.DEFAULT_HFOV_DEG,
            width=int(num("w")),
            height=int(num("h")),
        )
    except point_nav.NavError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=200)

    u, v = num("u"), num("v")
    point = point_nav.pixel_to_ground(u, v, calib)

    # خط الأفق: فوقه مفيش أرض (بنرجّعه للرسم على الشاشة)
    horizon_v: Optional[float] = (
        round(point_nav.horizon_pixel(calib), 1)
        if point_nav.horizon_in_frame(calib) else None
    )

    if point is None:
        return JSONResponse({
            "ok": False,
            "error": "البكسل ده فوق خط الأفق — دوس على الأرض مش على الحيطة",
            "horizon_v": horizon_v,
        }, status_code=200)

    return {
        "ok": True,
        "distance_m": round(point.distance_m, 3),
        "bearing_deg": round(point.bearing_deg, 2),
        "x_m": round(point.x_m, 3),
        "y_m": round(point.y_m, 3),
        "in_range": point_nav.MIN_TARGET_M <= point.distance_m <= point_nav.MAX_TARGET_M,
        "horizon_v": horizon_v,
        "f_px": round(calib.f, 1),
        "pitch_deg": round(calib.pitch_deg, 2),
        "height_m": round(calib.height_m, 3),
    }


@app.post("/api/point/calibrate")
async def api_point_calibrate(request: Request):
    """
    «الحتة دي على مسافة معروفة — ظبّط الميل عليها».

    دي اللي بتشيل التخمين: المشغّل يحط حاجة على متر ويدوس عليها،
    وإحنا نطلع الميل اللي يخلي الحساب يطلع ١.٠٠ بالظبط — من غير
    أي حساس موبايل ومن غير أي قياس من المشغّل غير المسافة الواحدة.
    """
    if not _cam_ok(request):
        raise HTTPException(status_code=401, detail="لازم مفتاح الكاميرا")
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="مفيش JSON")

    def num(key: str, default=None) -> float:
        if data.get(key) is None:
            if default is None:
                raise HTTPException(status_code=400, detail=f"ناقص: {key}")
            return float(default)
        try:
            return float(data.get(key))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"قيمة غلط: {key}")

    try:
        pitch = point_nav.solve_pitch_for_distance(
            num("u"), num("v"),
            height_m=num("height_m"),
            target_m=num("target_m", 1.0),
            hfov_deg=num("hfov_deg", point_nav.DEFAULT_HFOV_DEG),
            width=int(num("w")), height=int(num("h")),
        )
    except point_nav.NavError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=200)

    # نتأكد إن الميل الجديد فعلًا بيطلع نفس المسافة
    check = point_nav.ray_to_ground(
        num("u"), num("v"), height_m=num("height_m"), pitch_deg=pitch,
        hfov_deg=num("hfov_deg", point_nav.DEFAULT_HFOV_DEG),
        width=int(num("w")), height=int(num("h")),
    )
    return {
        "ok": True,
        "pitch_deg": pitch,
        "target_m": num("target_m", 1.0),
        "result_m": round(check.distance_m, 3) if check else None,
    }


@app.post("/api/point/visual/start")
async def api_point_visual_start(request: Request):
    """
    «الكاميرا تشوف العربية، وأنا أسوقها للحتة» — من غير أي معايرة.

    بياخد **دوستين** من الموبايل (على العربية، وعلى الحتة) ويسوق
    العربية بنفسه لحد ما توصل.

    ⚠️ `dry: true` = **وضع تجريبي**: السيرفو بيحسب ويتابع وبيقول
    أوامره، بس **مش بيبعت ولا أمر للعربية**. آمن تمامًا للتجربة.
    """
    if not _cam_ok(request):
        raise HTTPException(status_code=401, detail="لازم مفتاح الكاميرا")
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="مفيش JSON")

    if POINT_VISUAL["servo"] is not None:
        return JSONResponse({"ok": False, "error": "السيرفو شغال بالفعل"}, status_code=200)

    dry = bool(data.get("dry"))
    if not dry and motion.command_id:
        return JSONResponse({
            "ok": False,
            "error": f"فيه أمر شغال ({motion.kind}) — استنى يخلص أو وقّفه الأول",
        }, status_code=200)

    jpeg = camera.jpeg()
    gray = visual_servo.gray_from_jpeg(jpeg) if jpeg else None
    if gray is None:
        return JSONResponse({
            "ok": False,
            "error": "مفيش صورة واصلة من الموبايل — افتح الكاميرا على /go الأول",
        }, status_code=200)

    try:
        car_uv = (float(data["car_u"]), float(data["car_v"]))
        tgt_uv = (float(data["target_u"]), float(data["target_v"]))
    except (KeyError, TypeError, ValueError):
        return JSONResponse({"ok": False, "error": "ناقص مكان العربية أو الحتة"},
                            status_code=200)

    # ⚠️ **موديل الكاميرا** (اختياري بس مهم): لو المشغّل بعت ارتفاع
    # الكاميرا وميلها، السيرفو بيقيس **مسافة حقيقية بالمتر** ويقرر بيها
    # (وصلنا؟ قربنا؟) — بدل ما يعتمد على الفرق بالبكسل، اللي بيضلّل:
    # لما العربية تبعد، الهدف بيقرب من الأفق والبكسل بيقول «بتقرب» وهي
    # بتشرد (اتشاف: مشيت ١١ متر في الاتجاه الغلط — شوف test_visual_servo).
    calib = None
    h_m = data.get("height_m", data.get("cam_height_m"))
    p_deg = data.get("pitch_deg")
    if h_m is not None and p_deg is not None:
        try:
            calib = point_nav.Calib(
                height_m=float(h_m), pitch_deg=float(p_deg),
                hfov_deg=float(data.get("hfov_deg", point_nav.DEFAULT_HFOV_DEG)))
        except Exception as exc:  # noqa: BLE001
            calib = None
            log_event(f"⚠️ موديل الكاميرا مرفوض ({exc}) — هنشتغل بالبكسل")

    try:
        servo = visual_servo.VisualServo(gray, car_uv, tgt_uv, calib=calib)
    except visual_servo.Lost as exc:
        return JSONResponse({"ok": False, "error": f"مش قادر أمسك البقعة: {exc}"},
                            status_code=200)
    if calib is not None:
        log_event(f"👁️ السيرفو بموديل كاميرا (ارتفاع {calib.height_m}م · "
                  f"ميل {calib.pitch_deg}°) — مسافة حقيقية بالمتر")

    POINT_VISUAL["servo"] = servo
    POINT_VISUAL["dry"] = dry
    POINT_VISUAL["cmd"] = None
    POINT_VISUAL["error"] = None
    POINT_VISUAL["running"] = True

    if not dry:
        cmd = motion.start("point", "visual_go", None, "operator")
        POINT_VISUAL["command_id"] = cmd["command_id"]

    POINT_VISUAL["task"] = asyncio.create_task(_visual_loop(servo, dry, POINT_VISUAL.get("command_id")))
    log_event(f"👁️ السيرفو البصري بدأ{' (تجريبي)' if dry else ''}")
    return {"ok": True, "dry": dry, "state": servo.public_state()}


async def _blind_loop(plan: tap_blind.BlindPlan, dry: bool,
                      cmd_id: str = None) -> None:
    """
    «دوس على الصورة → العربية تروح» بالنسخة البسيطة (المفتوحة على الوقت).

    ثلاث مراحل، **من غير أي تغذية راجعة** (شوف `tap_blind.py`):

        ١) اللف: `dir=L/R` لمدة `plan.turn_s`
        ٢) المشي: `dir=F` لمدة `plan.drive_s`
        ٣) الوقف: lease بـ `status=stop` → الفيرموير يكتب PWM 0

    ⚠️ **الـ watchdog الخاص بالفيرموير شغّال طول الوقت**: لو الـ server
    وقع أو الاتصال اتقطع، الـ lease بيقف بعد 1500ms والعربية بتقف لوحدها
    (`lease_lost`). يعني **مش ممكن** العربية تفضل شغالة لو السيرفر اتقفل
    في النص. السقوف (١.٥s / ٤s) هي حماية تانية من دوسة غلط.

    ⚠️ مفيش حساس على العربية → مفيش فرملة ولا تفادي. حط إيدك على المفتاح.
    """
    async def hold(direction: str, speed: int, seconds: float) -> None:
        """يشغّل اتجاهًا لمدة، ويبعت وقف قبل ما يرجع."""
        if seconds <= 0 or dry:
            return
        motion.guide_dir = direction
        motion.guide_speed = int(speed)
        # ⚠️ نقسّم المتدة لقطع صغيرة: كل قطعة بنتحقق فيها إن الأمر لسه
        # شغال. ده مهم لو حد وقّف المشوار من الداشبورد في النص — نوقف
        # فورًا بدل ما نكمّل ناكرين.
        end = time.time() + seconds
        while time.time() < end:
            if cmd_id and motion.command_id != cmd_id:
                return                       # الأمر اتقفل من برّه
            await asyncio.sleep(min(0.1, max(0.0, end - time.time())))
        motion.guide_speed = 0

    log_event(f"👆 تايب بسيط: لف {plan.turn_dir or '—'} "
              f"{plan.turn_s:.2f}s · مشي {plan.drive_s:.2f}s"
              f"{' — تجريبي' if dry else ''}")

    try:
        # ١) اللف
        if plan.turn_dir and plan.turn_s > 0:
            await hold(plan.turn_dir, tap_blind.TURN_SPEED, plan.turn_s)
        # ⚠️ وقفة قصيرة بين اللف والمشي: الدرايفر محتاج شوية عشان يهدى،
        # والفيرموير فيه حماية عكس اتجاه (DIR_HOLD) بتشتغل لو الاتجاه
        # اتغير في نفس اللحظة. بدونها العربية بتعمل «قفزة».
        if not dry:
            await asyncio.sleep(0.15)
        # ٢) المشي
        await hold("F", tap_blind.DRIVE_SPEED, plan.drive_s)
    except asyncio.CancelledError:
        pass
    finally:
        # ٣) الوقف — **دايمًا**، حتى لو حصل استثناء أو إلغاء.
        # `motion.stop` بيبعت lease بـ status=stop → الفيرموير بيوقّف
        # وبيصفّر الـ PWM. ده اللي يمنع أي «هروب».
        motion.guide_speed = 0
        try:
            if not dry and cmd_id and motion.command_id == cmd_id:
                motion.stop(reason="tap_blind_done", silent=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"[blind] فشل الإيقاف: {exc}", flush=True)
        POINT_VISUAL["running"] = False
        POINT_VISUAL["blind_done"] = True


async def _visual_loop(servo, dry: bool, cmd_id: str = None) -> None:
    """حلقة السيرفو: صورة → قرار → أمر، ٣-٤ مرات في الثانية."""
    last = time.time()
    # ⚠️ في الوضع التجريبي العربية **مابتتحركش**، فأي حكم على «التقدّم»
    # مالوش معنى (الصورة ثابتة). عشان كده التجريبي بيقف بعد كام نبضة
    # ويعرض القرارات — يعني «هتعمل إيه» من غير ما تتحرك.
    
    
    try:
        while POINT_VISUAL["servo"] is servo:
            jpeg = camera.jpeg()
            gray = visual_servo.gray_from_jpeg(jpeg) if jpeg else None
            now = time.time()
            dt = max(0.02, now - last)
            last = now

            if gray is None:
                print("[visual_loop] GRAY IS NONE!,", flush=True)
                POINT_VISUAL["error"] = "الصورة قطعت"
                break

            try:
                if type(servo).__name__ == "VisualServo":
                    cmd = await asyncio.to_thread(servo.tick, gray, dt, dry)
                else:
                    cmd = await asyncio.to_thread(servo.tick, gray, dt)
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                POINT_VISUAL["error"] = f"خطأ في التتبّع: {exc}"
                break

            POINT_VISUAL["cmd"] = {
                "direction": cmd.direction, "speed": cmd.speed,
                "seconds": cmd.seconds, "why": cmd.why,
                "done": cmd.done, "reason": cmd.reason,
            }

            if not dry:
                # ⚠️ نفس مسار التوجيه: بنغيّر الاتجاه والسرعة **جوه
                # الـ lease** بس — كل ضمانات الأمر الواحد زي ما هي.
                if motion.command_id == POINT_VISUAL.get("command_id"):
                    motion.guide_dir = cmd.direction
                    motion.guide_speed = 0 if cmd.direction == "S" else cmd.speed
                else:
                    POINT_VISUAL["error"] = "الأمر اتلغى"
                    break

            if cmd.done:
                print(f"[servo_done] {cmd.reason}", flush=True)
                break
            if dry:
                pass
                if False:
                    POINT_VISUAL["error"] = ("تجريبي: دي القرارات — "
                                             "العربية مابتحركتش")
                    break
            await asyncio.sleep(cmd.seconds)
    finally:
        if not dry and motion.command_id == cmd_id and cmd_id is not None:
            motion.stop(reason="visual_done", silent=True)
        POINT_VISUAL["running"] = False
        # ⚠️ لازم نفضّي مكان السيرفو كمان. من غيره `start` بيفضل يرد
        # «السيرفو شغال بالفعل» على **كل** ضغطة «يلا» بعد أول مشوار
        # (حتى لو المشوار ده وقف لوحده) — والعربية تبقى مش بتتحرك خالص
        # لحد ما حد يدوس «وقّف». اتشاف على الأرض.
        POINT_VISUAL["servo"] = None
        POINT_VISUAL["task"] = None
        log_event("👁️ السيرفو البصري خلص")


@app.get("/api/point/visual/state")
async def api_point_visual_state(request: Request):
    if not _cam_ok(request):
        raise HTTPException(status_code=401, detail="لازم مفتاح الكاميرا")
    servo = POINT_VISUAL["servo"]
    return {
        "running": bool(POINT_VISUAL.get("running")),
        "dry": POINT_VISUAL["dry"],
        "cmd": POINT_VISUAL["cmd"],
        "error": POINT_VISUAL["error"],
        "state": servo.public_state() if servo else None,
    }


@app.post("/api/point/visual/stop")
async def api_point_visual_stop(request: Request):
    if not _cam_ok(request):
        raise HTTPException(status_code=401, detail="لازم مفتاح الكاميرا")
    servo = POINT_VISUAL["servo"]
    POINT_VISUAL["servo"] = None
    task = POINT_VISUAL.get("task")
    if task and not task.done():
        task.cancel()
    if motion.command_id and motion.kind == "point":
        motion.stop(reason="operator_stop")
    log_event("👁️ السيرفو البصري اتوقف")
    return {"ok": True, "was": servo is not None}


@app.post("/api/tap/start")
async def api_tap_start(request: Request):
    """
    **«دوس على الكاميرا → العربية تروح للمكان ده»** (شوف `tap_nav.py`).

    بياخد **دوسة واحدة** على الأرض في صورة الكاميرا → يمسك البقعة ويتتبّعها
    ويسوق ناحية الهدف.

    ⚠️ **العربية مش لازم تبان في الكادر خالص** — ده الفرق الأساسي عن
    السيرفو القديم (اللي كان بياخد `car_u/car_v` كمان).

    **الوضعين:**

    | `mode` | محتاج معايرة؟ | إزاي بيمشي |
    |---|---|---|
    | `visual` (الافتراضي) | **لأ** | يلفّ لحد ما الهدف يبقى في نص الكادر، ويمشي لحد **خط الوقوف** تحت |
    | `metric` | أيوه (`metric_ok`) | نفس الحاجة + العدّاد بالمتر يوقّف بدري |

    لو مافيش أرقام موديل في الطلب، بنستخدم **المعايرة المحفوظة** (شوف
    `/api/tap/config`). `dry: true` = تجريبي: يحسب ويقول قراراته من غير
    ما يبعت أي حركة.
    """
    if not _cam_ok(request):
        raise HTTPException(status_code=401, detail="لازم مفتاح الكاميرا")
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="مفيش JSON")

    if POINT_VISUAL["servo"] is not None:
        return JSONResponse({"ok": False, "error": "فيه مشوار شغال بالفعل"},
                            status_code=200)

    dry = bool(data.get("dry"))
    if not dry and motion.command_id:
        return JSONResponse({
            "ok": False,
            "error": f"فيه أمر شغال ({motion.kind}) — وقّفه الأول",
        }, status_code=200)

    # ---- ⓪ «النسخة البسيطة» (blind) ----
    # ⚠️ بتتحسب **قبل** أي حاجة تانية: مفيش كاميرا، مفيش معايرة، مفيش
    # سيرفو. بتاخد `u`/`v` بس وتحسب المدة. ده اللي بيخلّيها سريعة
    # ومستقرة ومفيش فيها خطوة واحدة بالغلط.
    # (`dry: true` بيشتغل برضه — مفيد تشوف الخطة قبل ما تطلع العربية.)
    if str(data.get("mode") or "").lower() == tap_blind.MODE:
        try:
            u = float(data["u"])
            v = float(data["v"])
            img_w = float(data.get("img_w", 640))
            img_h = float(data.get("img_h", 480))
        except (KeyError, TypeError, ValueError) as exc:
            return JSONResponse(
                {"ok": False, "error": f"بيانات ناقصة/غلط: {exc}"},
                status_code=200)

        plan = tap_blind.plan_from_tap(u, v, img_w, img_h)
        POINT_VISUAL["servo"] = None
        POINT_VISUAL["cmd"] = None
        POINT_VISUAL["running"] = True
        POINT_VISUAL["command_id"] = None
        POINT_VISUAL["blind_plan"] = plan.as_dict()

        if not dry:
            # ⚠️ `direction` في الأمر نفسه = اتجاه **اللف** (أو F لو مفيش
            # لف). الفيرموير بينفّذ اتجاه الأمر لحظة استلامه، فمحتاجين
            # نبعت الاتجاه الصح من أول ثانية.
            first = plan.turn_dir or "F"
            cmd = motion.start("point", "tap_blind", None, "operator",
                               direction=first)
            if cmd.get("command_id") is None:
                POINT_VISUAL["running"] = False
                return JSONResponse(
                    {"ok": False, "error": cmd.get("error", "مرفوض")},
                    status_code=200)
            POINT_VISUAL["command_id"] = cmd["command_id"]
            # ⚠️ مفيش hint في الوضع ده: `kind="point"` مش من HINT_DRIVEN
            # فمفيش بوابة «توجيه طازة» بتوقّف المشوار نصه (شوف
            # HINT_DRIVEN فوق).
            motion.guide_speed = tap_blind.TURN_SPEED if plan.turn_dir \
                else tap_blind.DRIVE_SPEED

        log_event(f"👆 تايب بسيط: {plan.as_dict()}"
                  f"{' — تجريبي' if dry else ''}")
        POINT_VISUAL["task"] = asyncio.create_task(
            _blind_loop(plan, dry, POINT_VISUAL.get("command_id")))
        return {"ok": True, "dry": dry, "mode": tap_blind.MODE,
                "plan": plan.as_dict()}

    jpeg = camera.jpeg()
    gray = visual_servo.gray_from_jpeg(jpeg) if jpeg else None
    if gray is None:
        return JSONResponse({
            "ok": False,
            "error": "مفيش صورة واصلة — افتح كاميرا الموبايل الأول",
        }, status_code=200)

    # الصورة الحقيقية بتحدّد المقاس (بنعتمد أبعاد الصورة الواصلة نفسها)
    h_img, w_img = gray.shape[0], gray.shape[1]
    try:
        u = float(data["u"])
        v = float(data["v"])
        cfg = dict(TAP_CONFIG)
        for key, cast in (("height_m", float), ("pitch_deg", float),
                          ("hfov_deg", float)):
            if data.get(key) is not None:
                cfg[key] = cast(data[key])
        model = _tap_model_from(cfg,
                                img_w=int(data.get("img_w", w_img)),
                                img_h=int(data.get("img_h", h_img)))
        mode = str(data.get("mode") or TAP_CONFIG.get(
            "mode", tap_nav.MODE_VISUAL))
        if mode not in (tap_nav.MODE_VISUAL, tap_nav.MODE_METRIC):
            mode = tap_nav.MODE_VISUAL
        stop_band = int(data.get("stop_band_px",
                                 TAP_CONFIG.get("stop_band_px",
                                                tap_nav.STOP_BAND_PX)))
        metric_ok = bool(data.get("metric_ok",
                                  TAP_CONFIG.get("metric_ok", False)))
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": f"بيانات ناقصة/غلط: {exc}"},
                            status_code=200)

    # ⚠️ الدوسة بتيجي بإحداثيات الصورة اللي **المشغّل شايفها**؛ لو مقاس
    # البث مختلف عن مقاس الصورة اللي واصلتنا، بنعيد الحساب بالنسبة.
    if data.get("img_w") and data.get("img_h"):
        u = u * w_img / float(data["img_w"])
        v = v * h_img / float(data["img_h"])

    servo = tap_nav.TapServo(model=model, u=u, v=v, mode=mode,
                             stop_band_px=stop_band, metric_ok=metric_ok)
    try:
        servo.seed(gray)
    except tap_nav.Lost as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=200)

    POINT_VISUAL["servo"] = servo
    POINT_VISUAL["dry"] = dry
    POINT_VISUAL["cmd"] = None
    POINT_VISUAL["error"] = None
    POINT_VISUAL["running"] = True
    POINT_VISUAL["command_id"] = None

    if not dry:
        cmd = motion.start("point", "tap_go", None, "operator",
                           direction="F")
        if cmd.get("command_id") is None:
            POINT_VISUAL["servo"] = None
            POINT_VISUAL["running"] = False
            return JSONResponse({"ok": False, "error": cmd.get("error", "مرفوض")},
                                status_code=200)
        POINT_VISUAL["command_id"] = cmd["command_id"]

    t0 = servo.last
    where = (f"{t0.distance_m:.2f}م · زاوية {t0.bearing_deg:.0f}°"
             if t0 else "؟")
    log_event(f"👆 «روح للمكان ده» ({mode}): {where} · هدف على "
              f"{round(v)}px · خط الوقوف {servo.stop_v:.0f}"
              f"{' — تجريبي' if dry else ''}")
    POINT_VISUAL["task"] = asyncio.create_task(_visual_loop(servo, dry, POINT_VISUAL.get("command_id")))
    return {"ok": True, "dry": dry, "state": servo.public_state()}


@app.get("/api/tap/config")
async def api_tap_config(request: Request):
    """
    **معايرة «دوس على الكاميرا»**: الأرقام المحفوظة + طبقة الرسم (الشبكة).

    الواجهة بتنادي دي وهي فاتحة عشان ترسم:
      • **الشبكة** (خطوط على ٠.٢٥م · ٠.٥م · ١م · ٢م …) — دي اللي بتخلي
        المعايرة حاجة **تشوفها**: حط علامة على الأرض على متر، وشوف خط
        الـ ١م واقع عليها ولا لأ. مش مطابق؟ ظبّط لحد ما يطابق.
      • **خط الوقوف** — المكان اللي العربية هتوقف عنده بالظبط.
      • **خط الأفق** — أي دوسة فوقه مرفوضة (مش أرض).
    """
    if not _cam_ok(request):
        raise HTTPException(status_code=401, detail="لازم مفتاح الكاميرا")
    w, h = _cam_frame_size()
    return _tap_public_config(img_w=w, img_h=h)


@app.post("/api/tap/config")
async def api_tap_config_save(request: Request):
    """حفظ المعايرة (على الجهاز) — وبتفضل بعد ما السيرفر يقف ويقوم."""
    if not _cam_ok(request):
        raise HTTPException(status_code=401, detail="لازم مفتاح الكاميرا")
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="مفيش JSON")

    cfg = dict(TAP_CONFIG)
    if data.get("clear"):
        cfg = dict(TAP_DEFAULTS)
    for key in ("height_m", "pitch_deg", "hfov_deg"):
        if data.get(key) is not None:
            cfg[key] = float(data[key])
    if data.get("mode") in (tap_nav.MODE_VISUAL, tap_nav.MODE_METRIC):
        cfg["mode"] = data["mode"]
    if data.get("stop_band_px") is not None:
        band = float(data["stop_band_px"])
        cfg["stop_band_px"] = int(max(tap_nav.STOP_BAND_MIN_PX,
                                      min(tap_nav.STOP_BAND_MAX_PX, band)))
    if data.get("metric_ok") is not None:
        cfg["metric_ok"] = bool(data["metric_ok"])
    if data.get("taps") is not None:
        cfg["taps"] = data["taps"][:6]
    if "horizon_v" in data:
        cfg["horizon_v"] = data["horizon_v"]

    TAP_CONFIG.clear()
    TAP_CONFIG.update(cfg)
    _tap_config_save(TAP_CONFIG)
    w, h = _cam_frame_size()
    log_event("📐 معايرة «دوس على الكاميرا» اتحفظت: "
              f"ارتفاع {TAP_CONFIG['height_m']}م · ميل {TAP_CONFIG['pitch_deg']}° · "
              f"وضع {TAP_CONFIG['mode']}"
              + (" · موثّقة ✔" if TAP_CONFIG.get("metric_ok") else ""))
    return _tap_public_config(img_w=w, img_h=h)


@app.post("/api/tap/solve")
async def api_tap_solve(request: Request):
    """
    **حل المعايرة** من دوسات حقيقية — ودي الطريقة العملية اللي بننصح بيها:

      ١) **دوّس على خط الأفق** (آخر الأرض) → `horizon_v`. الميل بيتحدّد
         بالظبط، ومن غير أي قياس ولا مسطرة.
      ٢) حط علامة على مسافة مقيسة (١ متر مثلًا) من مقدمة العربية، دوّس
         عليها وقول `distance_m` → الارتفاع بيتحدّد (وحيد).

    ⚠️ ولو دوّست **نقطتين وخلاص** من غير أفق: (ارتفاع، ميل) فيهم تشابه،
    فبنرجّع `ambiguous: true` مع **مدى الارتفاعات الممكنة** بدل ما نكدب
    عليك برقم واثق. (`pitch_from_horizon` في `tap_nav.py` فيها الشرح.)

    `save: true` = احفظ الناتج فورًا.
    """
    if not _cam_ok(request):
        raise HTTPException(status_code=401, detail="لازم مفتاح الكاميرا")
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="مفيش JSON")

    w, h = _cam_frame_size()
    taps = data.get("taps") or []
    if data.get("u") is not None and data.get("distance_m") is not None:
        taps = [*taps, {"u": data["u"], "v": data.get("v"),
                        "distance_m": data["distance_m"]}]
    horizon_v = data.get("horizon_v", TAP_CONFIG.get("horizon_v"))
    try:
        out = tap_nav.solve_model(
            taps,
            hfov_deg=float(data.get("hfov_deg", TAP_CONFIG.get("hfov_deg",
                                                              tap_nav.DEFAULT_HFOV_DEG))),
            img_w=int(data.get("img_w", w)), img_h=int(data.get("img_h", h)),
            height_m=(float(data["height_m"]) if data.get("height_m") is not None
                      else None),
            pitch_deg=(float(data["pitch_deg"]) if data.get("pitch_deg") is not None
                       else None),
            horizon_v=(float(horizon_v) if horizon_v is not None else None))
    except (ValueError, TypeError, KeyError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=200)

    if data.get("save"):
        cfg = dict(TAP_CONFIG)
        cfg["height_m"] = out["height_m"]
        cfg["pitch_deg"] = out["pitch_deg"]
        cfg["hfov_deg"] = float(data.get("hfov_deg", TAP_CONFIG.get("hfov_deg",
                                                                    tap_nav.DEFAULT_HFOV_DEG)))
        cfg["taps"] = taps[:6]
        cfg["horizon_v"] = horizon_v
        # ⚠️ «موثّقة» بس لما الحل يبقى **وحيد** وخطؤه صغير: غير كده
        # العدّاد بالمتر بيتوقّف على رقم مش مضمون، وده أسوأ من مفيش عدّاد.
        cfg["metric_ok"] = bool(not out.get("ambiguous")
                                and (out.get("residual_m") or 0) <= 0.06
                                and out.get("samples", 0) >= 1
                                and not out.get("clamped"))
        TAP_CONFIG.clear()
        TAP_CONFIG.update(cfg)
        _tap_config_save(TAP_CONFIG)
        log_event(f"📐 حل المعايرة: ارتفاع {out['height_m']}م · ميل "
                  f"{out['pitch_deg']}° · خطأ {out.get('residual_m')}م"
                  + (" · مش وحيد ⚠️" if out.get("ambiguous") else ""))

    pub = _tap_public_config(img_w=w, img_h=h)
    return {**pub, "solve": out, "saved": bool(data.get("save"))}


@app.post("/api/tap/calibrate")
async def api_tap_calibrate(request: Request):
    """
    **معايرة بضغطة واحدة**: حط حاجة على الأرض على مسافة معروفة (مثلاً
    ١ متر)، دوس عليها في الصورة، وقولنا المسافة → نرجّع **ميل الكاميرا**.

    من غير المعايرة دي، «كام متر» بيبقى تخمين. (الارتفاع بتقيسه بمسطرة.)
    """
    if not _cam_ok(request):
        raise HTTPException(status_code=401, detail="لازم مفتاح الكاميرا")
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="مفيش JSON")
    try:
        u, v = float(data["u"]), float(data["v"])
        dist = float(data["distance_m"])
        height = float(data["height_m"])
        pitch = tap_nav.solve_pitch(
            u, v, dist, height_m=height,
            hfov_deg=float(data.get("hfov_deg", tap_nav.DEFAULT_HFOV_DEG)),
            img_w=int(data.get("img_w", 640)), img_h=int(data.get("img_h", 480)))
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": f"بيانات ناقصة/غلط: {exc}"},
                            status_code=200)
    log_event(f"📐 معايرة الدوسة: ميل الكاميرا = {pitch}° "
              f"(ارتفاع {height}م · نقطة على {dist}م)")
    # ⚠️ لو الحلّ وصل لحد البحث، يبقى التركيبة دي **مستحيلة هندسيًا**:
    # كاميرا على ١٢ سم مش هتشوف نقطة على ١ متر في **نفس البكسل** لو
    # الميل وصل حده. لازم يدوس على بكسل أقرب للأفق (أعلى شوية).
    clamped = (pitch <= tap_nav.MIN_PITCH_DEG + 0.01
               or pitch >= tap_nav.MAX_PITCH_DEG - 0.01)
    return {"ok": True, "pitch_deg": pitch, "height_m": height,
            "hfov_deg": float(data.get("hfov_deg", tap_nav.DEFAULT_HFOV_DEG)),
            "clamped": clamped,
            "note": ("التركيبة دي مش ممكنة هندسيًا (الارتفاع/المسافة/البكسل) — "
                     "دوس على نقطة أقرب للأفق أو ظبّط الارتفاع") if clamped else None}


# ================== ترميم انحراف العربية ==================
#
# عجلة أسرع من التانية → العربية بتهرب ناحية. بنرمّمها من برّه بنبضات
# لف صغيرة عكس الانحراف (شوف trim.py). مفيش أي لمسة في الفيرموير.
@app.get("/api/trim", dependencies=[Depends(require_auth)])
async def api_trim_get():
    return TRIM.status()


@app.post("/api/trim", dependencies=[Depends(require_auth)])
async def api_trim_set(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    preset = body.get("preset")
    if preset is not None:
        TRIM.preset(str(preset))
    else:
        TRIM.configure(
            enabled=body.get("enabled"),
            side=body.get("side"),
            period_ms=body.get("period_ms"),
            pulse_ms=body.get("pulse_ms"),
            speed=body.get("speed"),
        )
    st = TRIM.status()
    print(f"[trim] {trim_describe(TRIM)}", flush=True)
    log_event("🔧 " + trim_describe(TRIM))
    return {"ok": True, **st}


# ================== ⚖️ توازن المواتير (من جوه الفيرموير) ==================
#
# ⚠️ ده **الإصلاح الأصلي** لانحراف العربية — مش الترميم بالنبضات:
# لو ناحية أسرع من التانية، العربية بتنحرف وهي ماشية، ولو الفرق كبير
# **بتلف حوالين نفسها**، و«اللف في المكان» بيبقى دايرة بتتنقل (مركز
# الدوران بيتزح ناحية العجلة الأسرع).
#
# الفيرموير بقى فيه معامل لكل ناحية (`gainL`/`gainR`) بيعدّل الـ PWM
# قبل ما يخرج للموتور، وبيتخزن في NVS فبيفضل بعد الريستارت.
# المعامل بيتقرا **من العربية نفسها** (شوف `gain_l`/`gain_r` في التيليمتري).
MOTOR_GAIN_MIN, MOTOR_GAIN_MAX = 20, 150


def _clamp_gain(value) -> Optional[int]:
    """معامل الناحية بين ٢٠٪ و ١٥٠٪ — أي حاجة برّه بتتقص."""
    try:
        return max(MOTOR_GAIN_MIN, min(MOTOR_GAIN_MAX, int(value)))
    except (TypeError, ValueError):
        return None


@app.get("/api/motor/gains", dependencies=[Depends(require_auth)])
async def api_motor_gains_get():
    """
    المعاملات اللي **شغالة على البورد دلوقتي** — من التيليمتري، مش نسخة
    محلية عندنا. عشان اللي بتشوفه هو اللي العربية ماشية بيه فعلًا.
    """
    tele = STATE["telemetry"] or {}
    return {
        "gain_l": tele.get("gain_l"),
        "gain_r": tele.get("gain_r"),
        "online": bool(STATE["online"]),
        "min": MOTOR_GAIN_MIN,
        "max": MOTOR_GAIN_MAX,
        "trim": TRIM.status(),
        "hint": "اليمين أسرع → قلّل gain_r. والصح إنك تعاير والعربية ماشية.",
    }


@app.post("/api/motor/gains", dependencies=[Depends(require_auth)])
async def api_motor_gains_set(request: Request):
    """
    بيبعت المعاملات للعربية في رسالة إعدادات **موقّعة** (HMAC).

        {"gain_l": 100, "gain_r": 78}   ← اليمين أسرع فبنقلّله
        {"gain_l": 100, "gain_r": 100}  ← رجوع للمصنع

    ⚠️ التوقيع بيغطي المعاملات — أي حد يقدر يغيّرها يقدر يخلّي العربية
    تلف حوالين نفسها.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    gl = _clamp_gain(body["gain_l"]) if "gain_l" in body else None
    gr = _clamp_gain(body["gain_r"]) if "gain_r" in body else None
    if gl is None and gr is None:
        return JSONResponse(
            {"ok": False, "error": f"اكتب gain_l و/أو gain_r رقم من "
                                   f"{MOTOR_GAIN_MIN} لـ {MOTOR_GAIN_MAX}"},
            status_code=400)

    payload = {
        "gain_l": gl if gl is not None else -1,     # -1 = سيب اللي عندك
        "gain_r": gr if gr is not None else -1,
    }
    payload["signature"] = cmd_sign.sign_config(-1, -1, -1,
                                                payload["gain_l"],
                                                payload["gain_r"])
    publish("config", json.dumps(payload))

    def _shown(v):
        return "زي ما هو" if v < 0 else f"{v}%"

    print(f"[motor] ⚖️ توازن المواتير → شمال {_shown(payload['gain_l'])} · "
          f"يمين {_shown(payload['gain_r'])}", flush=True)
    log_event(f"⚖️ توازن المواتير: شمال {_shown(payload['gain_l'])} · "
              f"يمين {_shown(payload['gain_r'])}")
    return {"ok": True, **payload}


# ---------- الضبط الحي (تردد/سقف/قطبية) ----------
# ⚠️ دي الحاجات اللي بنظبطها واحنا بنجرّب على الأرض: **تردد الـ PWM**
# (الزمارة)، **سقف سرعة القيادة** (العزم)، و**قطبية كل موتور** (لو
# الاتجاهات مقلوبة). كلها كانت أوامر سيريال (`f` / `m` / `pl` / `pr`) —
# ودلوقتي بقت من الداشبورد كمان، بنفس التوقيع (HMAC) عشان محدش يقدر
# يلعب في موتور العربية من غير المفتاح.
TUNE_FREQ_MIN, TUNE_FREQ_MAX = 100, 40000
TUNE_CAP_MIN, TUNE_CAP_MAX = 50, 255


def _tune_now() -> dict:
    """القيم الحالية زي ما العربية قالتها في آخر تيليمتري."""
    t = STATE["telemetry"] or {}
    return {
        "pwm_freq": t.get("pwm_freq"),
        "pwm_cap": t.get("pwm_cap"),
        "pol_l": t.get("pol_l"),
        "pol_r": t.get("pol_r"),
        "gain_l": t.get("gain_l"),
        "gain_r": t.get("gain_r"),
        "online": bool(STATE.get("online")),
        "freq_min": TUNE_FREQ_MIN, "freq_max": TUNE_FREQ_MAX,
        "cap_min": TUNE_CAP_MIN, "cap_max": TUNE_CAP_MAX,
        "note": "التردد العالي بيشيل الزمارة — بس لو الدرايفر مش قادر "
                "عليه المواتير هتضعف، ساعتها ارجع 1000.",
    }


@app.get("/api/car/tune", dependencies=[Depends(require_auth)])
async def api_tune_get():
    """القيم الحالية + الحدود المسموحة."""
    return _tune_now()


@app.post("/api/car/tune", dependencies=[Depends(require_auth)])
async def api_tune_post(request: Request):
    """
    يبعت ضبط حي للعربية (موقّع):

        {"pwm_freq": 4000, "pwm_cap": 200, "pol_l": 0, "pol_r": 1}

    أي حقل مش موجود = «سيب اللي عندك» (-1).
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    def _num(key, lo, hi):
        if key not in body or body[key] in (None, ""):
            return -1
        try:
            v = int(float(body[key]))
        except (TypeError, ValueError):
            return None
        if v < lo or v > hi:
            return None
        return v

    freq = _num("pwm_freq", TUNE_FREQ_MIN, TUNE_FREQ_MAX)
    cap = _num("pwm_cap", TUNE_CAP_MIN, TUNE_CAP_MAX)
    pl = _num("pol_l", 0, 1)
    pr = _num("pol_r", 0, 1)
    # ⚠️⚠️ `ilock` كان **ناقص** من النقطة دي (اتمسك 2026-09-20):
    # مفتاح «Sensor interlock» في الداشبورد كان بيبعت `ilock` والسيرفر
    # **يتجاهله في السكوت** — يعني المفتاح ديكور، والإنترلوك يفضل زي ما هو.
    # والأسوأ: مع حساس ميت، الإنترلوك الشغال = **كل أمر يترفض**
    # (`sensor_invalid`) والعربية ماتتحركش، والمشغّل يفتكر إنه قفله من
    # الداشبورد وهو مقفولش.
    #   0 = تجاوز (اسمح بالحركة والحساس ميت) · 1 = عادي (ارفضها) · -1 = زي ما هو
    il = _num("ilock", 0, 1)
    if None in (freq, cap, pl, pr, il):
        return JSONResponse(
            {"ok": False, "error": f"قيم غلط — التردد {TUNE_FREQ_MIN}–"
                                   f"{TUNE_FREQ_MAX} · السقف {TUNE_CAP_MIN}–"
                                   f"{TUNE_CAP_MAX} · القطبية 0/1 · الإنترلوك 0/1"},
            status_code=400)

    payload = {"pwm_freq": freq, "pwm_cap": cap, "pol_l": pl, "pol_r": pr,
               "ilock": il}
    payload["signature"] = cmd_sign.sign_tune(freq, cap, pl, pr, il)
    # ⚠️ بنبعت على `config` (مش `tune`): قائمة صلاحيات البروكر بتسمح
    # للعربية تقرا `config` — وموضوع `tune` محتاج تحديث ACL + إعادة
    # تشغيل البروكر (بصلاحية مدير). الفيرموير بيقبل الضبط على الموضوعين،
    # فالتغيير ده بيخلي الضبط يشتغل من غير أي خطوة إضافية.
    publish("config", json.dumps(payload))

    def _shown(v, unit=""):
        return "زي ما هو" if v < 0 else f"{v}{unit}"

    line = (f"🎛️ ضبط حي: تردد {_shown(freq, 'Hz')} · سقف "
            f"{_shown(cap)} · قطبية شمال {_shown(pl)} / يمين {_shown(pr)}")
    print(f"[tune] {line}", flush=True)
    log_event(line)
    return {"ok": True, **payload}


# ---------- الدخول ----------
@app.get("/api/session")
async def api_session(request: Request):
    return {"authed": _token_ok(request.cookies.get(COOKIE_NAME))}


@app.post("/api/login")
async def api_login(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    user = str(data.get("user", ""))
    password = str(data.get("password", ""))

    if not (_same(user, DASH_USER) and _same(password, DASH_PASS)):
        await asyncio.sleep(0.7)
        print(f"[auth] محاولة دخول فاشلة (يوزر: {user!r})", flush=True)
        return JSONResponse({"ok": False, "error": "اليوزر أو الباسورد غلط"}, status_code=401)

    token = _new_session()
    resp = JSONResponse({"ok": True, "user": DASH_USER})
    resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="strict",
                    max_age=SESSION_TTL, path="/")
    db.audit(DASH_USER, "login")
    print("[auth] تسجيل دخول ناجح", flush=True)
    return resp


@app.post("/api/logout")
async def api_logout(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        SESSIONS.pop(token, None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


# ---------- الحالة ----------
@app.get("/api/state", dependencies=[Depends(require_auth)])
async def api_state():
    fresh = (time.time() - STATE["last_seen"]) < 2.0
    return {
        "online": fresh,
        "brokerConnected": monitor.is_connected(),
        "device": STATE["device"],
        "telemetry": STATE["telemetry"],
        # ⚠️ عدّاد الريستارتات: «الـ ESP بيرستر» كانت شكوى مالهاش مقياس —
        # بالحقل ده المشغّل يشوف بنفسه إذا كانت المشكلة اتصلحت.
        "resets": _reset_stats(),
        "history": STATE["history"][-120:],
        "events": STATE["events"][-40:],
        "motion": motion.status(),
        "checkpointRun": checkpoint_run.status(),
        "guide": guide.status(),
        "camera": camera.status(),
        "trim": TRIM.status(),
        "ingest": platform.stats(),
        "db": db.stats(),
        "history": history_mod.history.stats(),
    }


# ---------- سجل الساعات (آخر 24 ساعة) ----------
# Telemetry is kept at 1 Hz in a local SQLite file and browsable by the local
# wall-clock hour. Deliberately not the postgres `db` — that is down on this
# machine, and history has to keep working when the car keeps working.
@app.get("/api/history/hours", dependencies=[Depends(require_auth)])
async def api_history_hours():
    return {"hours": history_mod.history.hours(), "stats": history_mod.history.stats()}


@app.get("/api/history", dependencies=[Depends(require_auth)])
async def api_history(hour: Optional[str] = None):
    """
    One hour of history. Defaults to the current hour so a bare GET is useful.

    An unparseable `hour` is a 400, not a 500 — it is operator input, and a bad
    clock string must never surface as a server fault in the dashboard.
    """
    if not hour:
        # Module-level, not a History method (see the note below).
        hour = history_mod.hour_key(time.time())
    try:
        # Module-level, NOT a History method — `hour_bounds` validates the label
        # the server itself produced, so a malformed one can only be a bad
        # query string.
        history_mod.hour_bounds(hour)
    except (ValueError, TypeError):
        raise HTTPException(400, "hour must look like 2026-09-30T14:00")
    return history_mod.history.hour(hour)


@app.get("/api/history/snapshot/{filename}", dependencies=[Depends(require_auth)])
async def api_history_snapshot(filename: str):
    path = history_mod.history.snapshot_file(filename)
    if path is None:
        raise HTTPException(404, "no such snapshot")
    return FileResponse(str(path), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})


# ---------- AI Auto-Pilot ----------
# Additive. The reflex lives in `vision_autopilot.py` and drives through the
# EXISTING `motion.set_gyro_input()` seam, so these three routes only turn it
# on and off - they never sign a command or publish one themselves.
#
# ⚠️ ENABLE is a real safety gate, not a formality: the HC-SR04 is disabled in
# the firmware, so the camera is the only forward obstacle detection this car
# has. `enable()` refuses unless the pipeline is live, telemetry is arriving
# and the car is neither lifted nor sensor-dead.
@app.get("/api/autopilot", dependencies=[Depends(require_auth)])
async def api_autopilot_status():
    return {"ok": True, **autopilot.status()}


@app.post("/api/autopilot/enable", dependencies=[Depends(require_auth)])
async def api_autopilot_enable():
    res = autopilot.enable()
    if not res.get("ok"):
        # 409: the request was understood, the car is not in a state to accept
        # it. A 200 here would let the dashboard show "on" for a car that is
        # not moving, which is the failure mode that gets someone hurt.
        return JSONResponse(res, status_code=409)
    return res


@app.post("/api/autopilot/disable", dependencies=[Depends(require_auth)])
async def api_autopilot_disable():
    return autopilot.disable()


# ---------- مساعد البحث في السجل ----------
# READ-ONLY. This endpoint only searches rows that already exist inside the
# 24h retention window and returns the snapshot image + the AI scene read-out
# recorded at that instant + the telemetry sample nearest to it. It writes
# nothing, runs no inference, and does not participate in recording or in the
# video stream.
@app.post("/api/history/query", dependencies=[Depends(require_auth)])
async def api_history_query(request: Request):
    """
    Search the saved record and answer with what actually happened.

    Accepts `{"query": "..."}`, optional `"hour"` and `"limit"`. The answer is
    assembled from the stored rows, never generated, so it cannot invent a
    moment that did not happen and it works with no API key configured.
    """
    body = {}
    try:
        body = await request.json() or {}
    except Exception:
        body = {}                      # an empty body is a valid "show me everything"
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be a JSON object")

    text = str(body.get("query") or body.get("q") or "")
    hour = body.get("hour") or None
    if hour is not None:
        hour = str(hour)
        try:
            history_mod.hour_bounds(hour)
        except (ValueError, TypeError):
            raise HTTPException(400, "hour must look like 2026-09-30T14:00")
    try:
        limit = int(body.get("limit") or 20)
    except (TypeError, ValueError):
        raise HTTPException(400, "limit must be a number")
    limit = max(1, min(limit, 100))

    return history_mod.history.query(text=text, hour=hour, limit=limit)


@app.get("/api/db/stats", dependencies=[Depends(require_auth)])
async def api_db_stats():
    return db.stats()


@app.get("/api/db/events", dependencies=[Depends(require_auth)])
async def api_db_events(limit: int = 30, event_type: Optional[str] = None):
    return {"events": db.recent_events(limit=min(limit, 200), event_type=event_type)}


@app.get("/api/db/commands", dependencies=[Depends(require_auth)])
async def api_db_commands(limit: int = 20):
    return {"commands": db.recent_commands(limit=min(limit, 100))}


@app.get("/api/db/checkpoints", dependencies=[Depends(require_auth)])
async def api_db_checkpoints(limit: int = 10):
    return {"checkpoints": db.recent_checkpoints(limit=min(limit, 50))}


# ---------- منصة الويب ----------
@app.get("/api/ingest", dependencies=[Depends(require_auth)])
async def api_ingest():
    """حالة القناة مع منصة صاحب الويب (من غير ما نطبع التوكن)."""
    return platform.stats()


@app.post("/api/ingest/health", dependencies=[Depends(require_auth)])
async def api_ingest_health():
    """بيبعت نبضة حالًا للمنصة ويرجّع ردها — للتشخيص السريع."""
    if not platform.enabled:
        return JSONResponse({"ok": False, "error": platform.error}, status_code=409)
    status, answer = platform.health_now()
    ok = status == 200
    if ok:
        log_event(f"🛰️ المنصة ردت {status}")
    else:
        log_event(f"⚠️ المنصة مش رادة ({status}) — {str(answer)[:120]}")
    return {"ok": ok, "status": status, "answer": answer}


# ---------- الحركة ----------
@app.post("/api/patrol/start", dependencies=[Depends(require_auth)])
async def api_patrol_start(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    route_id = str(body.get("route_id") or "route_a")
    incident_id = body.get("incident_id")
    approved_by = str(body.get("approved_by") or DASH_USER)

    if not STATE["online"]:
        return JSONResponse({"ok": False, "error": "العربية مش متصلة"}, status_code=409)
    if motion.command_id:
        return JSONResponse({"ok": False, "error": "فيه حركة شغالة بالفعل"}, status_code=409)

    cmd = motion.start("patrol", route_id, incident_id, approved_by)
    return {"ok": True, "command": cmd}


@app.post("/api/patrol/stop", dependencies=[Depends(require_auth)])
async def api_patrol_stop():
    was = motion.command_id
    checkpoint_run.stop("operator_stop")
    motion.stop(reason="operator_stop")
    return {"ok": True, "stopped": was}


# ---------- دورية النقاط (الباك إند يبعت قائمة) ----------
@app.post("/api/patrol/route", dependencies=[Depends(require_auth)])
async def api_patrol_route(request: Request):
    """
    الباك إند بيبعت قائمة نقاط بالترتيب:
        { "points": [ {"checkpoint_id": "cp_1", "leg_seconds": 5}, ... ] }

    العربية تمشي كل نقطة، و**لما توصل** كاميرا الموبايل تصوّر.
    ولو نقطة وقعت، بننط للي بعدها ونكمّل.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    raw = body.get("points") or body.get("checkpoints") or []
    if isinstance(raw, str):
        raw = [p.strip() for p in raw.split(",") if p.strip()]
    if not isinstance(raw, list) or not raw:
        return JSONResponse({"ok": False, "error": "points مطلوبة (قائمة نقاط)"},
                            status_code=400)

    res = await checkpoint_run.start(
        raw,
        route_id=str(body.get("route_id") or "route_a"),
        incident_id=body.get("incident_id"),
        approved_by=str(body.get("approved_by") or DASH_USER),
        move=bool(body.get("move", True)),
        default_leg=float(body.get("default_leg_seconds", 5) or 5),
        point_timeout=float(body.get("point_timeout_seconds", 20) or 20),
        retry_failed=bool(body.get("retry_failed", False)),
    )
    if not res.get("ok"):
        return JSONResponse(res, status_code=409)

    db.audit(checkpoint_run.approved_by or DASH_USER, "patrol_route_started",
             {"route_id": checkpoint_run.route_id,
              "incident_id": checkpoint_run.incident_id,
              "points": [p["checkpoint_id"] for p in checkpoint_run.points]})

    return {"ok": True, "route_id": checkpoint_run.route_id,
            "points": len(checkpoint_run.points), "auto_photo": True,
            "note": "كل نقطة تتصور من كاميرا الموبايل وقت الوصول — "
                    "والنقطة اللي تقع بننط للي بعدها"}


@app.get("/api/patrol/route", dependencies=[Depends(require_auth)])
async def api_patrol_route_status():
    return checkpoint_run.status()


@app.post("/api/patrol/route/stop", dependencies=[Depends(require_auth)])
async def api_patrol_route_stop():
    stopped = checkpoint_run.stop("operator_stop")
    return {"ok": True, "stopped": stopped, "status": checkpoint_run.status()}


# ================== 🧭 التوجيه (Guide) ==================
# ⚠️ **مفيش نقطة داخلة.** مش بنستقبل توجيه من الموبايل ومابنستقبلش
# أي حاجة من برّه. إحنا اللي بنسحب: `GET /api/devices/commands`
# بترجّع مع كل أمر `hint` و `hint_age_ms`، و`guide_runner.py` هو
# اللي بيمشي الدورة. السبب إن صفحة التوجيه لازم تشتغل على HTTPS
# (WebXR مش بيتفتح غير في secure context)، والمتصفح بيرفض صفحة
# آمنة تبعت لـ HTTP على الشبكة المحلية.
#
# رابطين بس موجودين هنا، والاتنين **للداشبورد** (كوكي) — مش للصفحة:
#   GET  /api/guide          حالة التوجيه
#   POST /api/guide/stop     وقف فوري

@app.get("/api/guide", dependencies=[Depends(require_auth)])
async def api_guide_status():
    """حالة التوجيه للداشبورد + حالة القناة مع المنصة."""
    return {**guide_runner.status(), "deviceApi": device_api.stats()}


@app.post("/api/guide/stop", dependencies=[Depends(require_auth)])
async def api_guide_stop():
    """
    وقف فوري من المشغّل — **مقفول** ومابيتفتحش لوحده.

    ⚠️ القفل بيتحط **الأول** (قبل أي `await`) عشان حلقة السحب ما تلحقش
    تفتح أمر جديد في نفس اللحظة. وبيتشال بـ `POST /api/guide/resume` بس.
    """
    guide_runner.suspend("operator_stop")
    guide.clear()
    motion.stop(reason="operator_stop")
    log_event("🧭 التوجيه اتوقف من المشغّل (مقفول لحد resume)")
    return {"ok": True, "state": "stopped", "suspended": True}


@app.post("/api/guide/resume", dependencies=[Depends(require_auth)])
async def api_guide_resume():
    """
    إعادة تسليح التوجيه — الطريق **الوحيد** لشيل قفل `/api/guide/stop`.

    (من غير كده: وقف المشغّل كان بيتلغى لوحده في أقل من ربع ثانية.)
    """
    guide_runner.resume()
    log_event("🧭 التوجيه اتسلّح تاني من المشغّل")
    return {**guide_runner.status(), "ok": True, "suspended": False}


@app.post("/api/motion/stop", dependencies=[Depends(require_auth)])
async def api_motion_stop():
    was = motion.command_id
    motion.stop(reason="operator_stop")
    # ⚠️ نفس القفل: من غير كده حلقة السحب تلاقي الأمر لسه `active` عند
    # المنصة وتفتحه تاني (نفس باگ `/api/guide/stop`).
    guide_runner.suspend("operator_stop")
    return {"ok": True, "stopped": was, "suspended": True}


@app.post("/api/checkpoint", dependencies=[Depends(require_auth)])
async def api_checkpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    cp_id = str(body.get("checkpoint_id") or "").strip()
    if not cp_id:
        return JSONResponse({"ok": False, "error": "checkpoint_id مطلوب"}, status_code=400)

    route_id = str(body.get("route_id") or motion.route_id or "route_a")
    incident_id = body.get("incident_id") or motion.incident_id

    # leg_seconds = 0 → تأكيد فوري من غير حركة (المشغّل هو اللي أكّد)
    publish_checkpoint(cp_id, route_id, incident_id, 0, DASH_USER)
    db.audit(DASH_USER, "checkpoint_confirm",
             {"checkpoint_id": cp_id, "route_id": route_id, "incident_id": incident_id})
    log_event(f"تأكيد نقطة تفتيش من المشغّل: {cp_id}")
    return {"ok": True, "checkpoint_id": cp_id}


# ---------- الأدلة (الصور) ----------
EVIDENCE_DIR = HERE / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

# آخر صورة لكل نقطة تفتيش — عشان نربطها بالحدث اللي جاي من العربية
PENDING_EVIDENCE: dict[str, str] = {}


def _current_scene() -> dict:
    """
    The YOLO read-out for right now, or {} when vision is not running.

    Read-only by construction: it asks the pipeline for the detections it
    ALREADY has (`pipe.detections()`) and formats them with `vision.describe`.
    It never triggers inference, never touches the camera, and never changes
    how anything is recorded or streamed. Failures are swallowed on purpose —
    a snapshot with no scene text is still a useful snapshot.
    """
    try:
        pipe = _get_vision()
        if pipe is None:
            return {}
        return vision.describe(pipe.detections())
    except Exception:
        return {}


def store_evidence(raw: bytes, *, source: str, checkpoint_id: Optional[str] = None,
                   incident_id: Optional[str] = None, route_id: Optional[str] = None,
                   captured_by: Optional[str] = None) -> Optional[str]:
    """بيخزّن صورة (بايتات JPEG) ويرجّع الـ evidence_id."""
    if not raw or len(raw) < 512:
        return None

    evidence_id = "ev-" + uuid.uuid4().hex[:12]
    digest = hashlib.sha256(raw).hexdigest()
    path = EVIDENCE_DIR / f"{evidence_id}.jpg"
    path.write_bytes(raw)

    rec = {
        "id": evidence_id,
        "source": source,
        "content_type": "image/jpeg",
        "size_bytes": len(raw),
        "sha256": digest,
        "file_path": str(path),
        "checkpoint_id": checkpoint_id,
        "incident_id": incident_id,
        "route_id": route_id,
        "event_id": None,
        "captured_by": captured_by or DASH_USER,
    }
    db.save_evidence(rec)
    db.audit(captured_by or DASH_USER, "evidence_captured",
             {"evidence_id": evidence_id, "checkpoint_id": checkpoint_id,
              "incident_id": incident_id, "source": source, "size": len(raw)})

    # ---- صورة التاريخ (الساعة) ----
    # Every snapshot in this app funnels through this function — the dashboard
    # SNAPSHOT button, the waypoint confirm, and auto-capture all land here. So
    # one call covers all of them, and each one is linked to the telemetry state
    # at that instant (direction / distance / rssi) so a thumbnail explains
    # itself without having to be re-derived later.
    #
    # The YOLO read-out is captured from the LIVE vision state that already
    # exists at this moment. It is a read of what the pipeline last computed —
    # no second inference, no new inference request, and the stream itself is
    # untouched.
    history_mod.history.record_snapshot(raw, STATE["telemetry"], _current_scene())

    if checkpoint_id:
        PENDING_EVIDENCE[str(checkpoint_id)] = evidence_id

    log_event(f"📸 صورة ({source}): {evidence_id}" +
              (f" — نقطة {checkpoint_id}" if checkpoint_id else ""))

    # ---- نبعت الصورة للمنصة كمان ----
    # عندهم: استقبال صورة حقيقية هو اللي بيخلي المصدر "سليم"،
    # مش نبضة الصحة.
    if platform.enabled:
        platform.submit_frame(raw, captured_at=rec.get("captured_at")
                              if isinstance(rec, dict) else None)

    _broadcast_threadsafe({"type": "evidence", "evidence_id": evidence_id,
                           "checkpoint_id": checkpoint_id, "source": source,
                           "url": f"/evidence/{evidence_id}"})
    return evidence_id


def auto_capture(checkpoint_id: str, incident_id: Optional[str],
                 route_id: Optional[str], why: str) -> Optional[str]:
    """
    تصوير تلقائي — من غير ما حد يدوس أي زرار.
    بياخد آخر إطار من الكاميرا (موبايل أو لابتوب) ويخزّنه.
    """
    frame = camera.jpeg()
    if not frame:
        log_event(f"⚠️ مفيش كاميرا — نقطة {checkpoint_id} اتأكدت من غير صورة")
        _broadcast_threadsafe({"type": "evidence_missing",
                               "checkpoint_id": checkpoint_id, "why": why})
        return None

    ev_id = store_evidence(frame, source=f"auto_{why}",
                           checkpoint_id=checkpoint_id,
                           incident_id=incident_id, route_id=route_id,
                           captured_by="auto")
    if ev_id:
        log_event(f"📸 تصوير تلقائي وقت تأكيد {checkpoint_id} ({why})")
    return ev_id


@app.post("/api/evidence", dependencies=[Depends(require_auth)])
async def api_evidence(request: Request):
    """رفع صورة يدوي من الداشبورد (اختياري — التصوير التلقائي هو الأساس)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "bad json"}, status_code=400)

    data_b64 = body.get("image")
    if not data_b64:
        return JSONResponse({"ok": False, "error": "image مطلوب"}, status_code=400)

    if data_b64.strip().startswith("data:"):
        data_b64 = data_b64.split(",", 1)[1]

    try:
        raw = base64.b64decode(data_b64, validate=True)
    except Exception:
        return JSONResponse({"ok": False, "error": "base64 غلط"}, status_code=400)

    if len(raw) < 512:
        return JSONResponse({"ok": False, "error": "الصورة صغيرة أوي"}, status_code=400)
    if len(raw) > 6 * 1024 * 1024:
        return JSONResponse({"ok": False, "error": "الصورة كبيرة أوي (أقصى 6MB)"}, status_code=413)

    checkpoint_id = body.get("checkpoint_id")
    incident_id = body.get("incident_id")
    if incident_id is None:
        incident_id = motion.incident_id

    ev_id = store_evidence(
        raw,
        source=str(body.get("source") or "phone_camera"),
        checkpoint_id=checkpoint_id,
        incident_id=incident_id,
        route_id=body.get("route_id") or motion.route_id,
        captured_by=DASH_USER,
    )
    if not ev_id:
        return JSONResponse({"ok": False, "error": "فشل التخزين"}, status_code=500)

    rec = db.get_evidence(ev_id) or {}
    return {"ok": True, "evidence_id": ev_id, "sha256": rec.get("sha256"),
            "size_bytes": rec.get("size_bytes"), "url": f"/evidence/{ev_id}"}


# ---------- طلب من الباك إند ----------
@app.post("/api/dispatch/checkpoint", dependencies=[Depends(require_auth)])
async def api_dispatch_checkpoint(request: Request):
    """
    الباك إند بيطلب من العربية تروح لنقطة تفتيش.

    الأدابتر:
      1) يصدر أمر حركة مصرّح (لو مفيش حركة شغالة)
      2) يبعت النقطة للعربية
      3) **يصوّر تلقائيًا** لما العربية تأكد الوصول
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    cp_id = str(body.get("checkpoint_id") or "").strip()
    if not cp_id:
        return JSONResponse({"ok": False, "error": "checkpoint_id مطلوب"}, status_code=400)

    route_id = str(body.get("route_id") or "route_a")
    incident_id = body.get("incident_id")
    approved_by = str(body.get("approved_by") or DASH_USER)
    move = bool(body.get("move", True))

    # كام ثانية تمشي ناحية النقطة (0 = تأكيد فوري من غير حركة)
    try:
        leg = float(body.get("leg_seconds", 0) or 0)
    except (TypeError, ValueError):
        leg = 0.0
    leg = max(0.0, min(15.0, leg))

    if not STATE["online"]:
        return JSONResponse({"ok": False, "error": "العربية مش متصلة"}, status_code=409)

    # ⚠️ **حاجز مهم**: نقطة بمدّة مشي وإحنا في **جلسة توجيه** بتخلط
    # الاتنين: الفيرموير بيمشي النقطة بنفسه (وبيتجاهل أوامر التوجيه)،
    # وبعد ما تخلص بيدخل «وقفة بعد الوصول» (`cpHold`) → **التوجيه
    # يتجمّد** من غير أي سبب واضح للمشغّل. النقط في الوضع ده ملك صفحة
    # التوجيه. (الفيرموير كمان بيرفضها بحقل `guide` في الأمر — وده حاجز
    # عندنا عشان السبب يوصل واضح قبل ما تتبعت أصلاً.)
    if move and leg > 0 and motion.command_id and motion.kind == "guide":
        return JSONResponse({
            "ok": False,
            "error": "فيه جلسة توجيه شغالة — النقط ملك صفحة التوجيه. "
                     "وقّف التوجيه الأول (/api/guide/stop) وبعدين ابعت النقطة",
        }, status_code=409)

    # ١) أمر حركة مصرّح (لو مش ماشية بالفعل)
    started = None
    if move and not motion.command_id:
        started = motion.start("patrol", route_id, incident_id, approved_by)["command_id"]

    # ٢) نبعت النقطة للعربية — بتمشي بنفسها وتأكد **لما توصل**
    #    (التأكيد بيجيب معاه التصوير التلقائي من كاميرا الموبايل)
    publish_checkpoint(cp_id, route_id, incident_id, leg, approved_by)
    db.audit(approved_by, "checkpoint_dispatched",
             {"checkpoint_id": cp_id, "route_id": route_id,
              "incident_id": incident_id, "command_id": started,
              "leg_seconds": leg})

    log_event(f"📤 الباك إند طلب النقطة {cp_id}"
              + (f" (أمر {started})" if started else "")
              + (f" — تمشي {leg:g}ث" if leg else ""))

    return {"ok": True, "checkpoint_id": cp_id, "command_id": started,
            "leg_seconds": leg, "auto_photo": True,
            "note": "العربية هتمشي النقطة، والصورة هتتصور من كاميرا الموبايل "
                    "وقت الوصول"}


@app.get("/evidence/{evidence_id}")
async def get_evidence_image(evidence_id: str, request: Request):
    """بيسلّم الصورة نفسها."""
    if not _token_ok(request.cookies.get(COOKIE_NAME)):
        raise HTTPException(status_code=401, detail="لازم تسجّل دخول")
    rec = db.get_evidence(evidence_id)
    if not rec:
        raise HTTPException(status_code=404, detail="الصورة مش موجودة")
    path = Path(rec["file_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="الملف مش موجود")
    return FileResponse(path, media_type=rec.get("content_type") or "image/jpeg")


@app.get("/api/evidence", dependencies=[Depends(require_auth)])
async def api_evidence_list(limit: int = 20, checkpoint_id: Optional[str] = None):
    if checkpoint_id:
        return {"evidence": db.evidence_for_checkpoint(checkpoint_id)}
    return {"evidence": db.recent_evidence(limit=min(limit, 100))}


# ---------- الكاميرا ----------
BOUNDARY = "carframe"


def _placeholder_jpeg() -> bytes:
    """
    صورة بديلة بتتعرض في الداشبورد لما كاميرا الموبايل تكون لسه
    ماوصلتش — بدل شاشة سودة من غير أي تفسير.
    """
    import cv2
    import numpy as np

    img = np.full((360, 640, 3), 14, dtype=np.uint8)
    cv2.putText(img, "Phone camera not connected", (36, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (150, 150, 150), 2, cv2.LINE_AA)
    cv2.putText(img, "Open /cam on the phone", (36, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 210, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"source: {camera.source}", (36, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 118, 129), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else b""


_PLACEHOLDER: Optional[bytes] = None


def placeholder_jpeg() -> bytes:
    global _PLACEHOLDER
    if _PLACEHOLDER is None:
        with contextlib.suppress(Exception):
            _PLACEHOLDER = _placeholder_jpeg()
    return _PLACEHOLDER or b""


def _qr_png(text: str, scale: int = 6) -> bytes:
    """
    QR جاهز للموبايل — OpenCV بيرجّع 0 = خانة سودة، 255 = بيضا،
    فبنكبّره ونحطله هامش أبيض (quiet zone) وبس.
    """
    import cv2
    import numpy as np

    encoder = cv2.QRCodeEncoder_create()
    code = encoder.encode(text)                 # 0 = سودة، 255 = بيضا
    big = np.kron(code, np.ones((scale, scale), dtype=np.uint8))
    big = np.pad(big, scale * 4, constant_values=255)
    ok, buf = cv2.imencode(".png", big)
    return buf.tobytes() if ok else b""


async def _mjpeg():
    last = -1
    warned = False
    while True:
        seq = camera.seq()
        if seq != last:
            frame = camera.jpeg()
            if frame:
                last = seq
                warned = False
                yield (
                    b"--" + BOUNDARY.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                    + frame + b"\r\n"
                )
                continue
        if not warned:
            # لسه مفيش صورة → بنبعت صورة توضيحية بدل ما الشاشة تفضل سودة
            ph = placeholder_jpeg()
            if ph:
                warned = True
                yield (
                    b"--" + BOUNDARY.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(ph)).encode() + b"\r\n\r\n"
                    + ph + b"\r\n"
                )
        await asyncio.sleep(0.01)


@app.get("/api/camera", dependencies=[Depends(require_auth)])
async def api_camera():
    return camera.status()


@app.get("/api/camlink", dependencies=[Depends(require_auth)])
async def api_camlink(request: Request):
    """رابط كاميرا الموبايل + QR + حالة المصدر — للداشبورد."""
    ip = primary_ip() or (all_ips() or ["127.0.0.1"])[0]
    url = f"https://{ip}:{PORT}/cam?k={CAM_KEY}"
    st = camera.status()
    return {
        "url": url,
        "key": CAM_KEY,
        "ip": ip,
        "source": CAMERA_SOURCE,
        "laptopUrl": f"https://{ip}:{PORT}/cam",
        "status": st,
        "hint": "افتح الرابط ده على الموبايل (نفس شبكة الواي فاي) "
                "واقبل الشهادة ثم دوس تشغيل الكاميرا",
    }


@app.get("/api/camqr.png")
async def api_camqr(request: Request):
    """QR لكاميرا الموبايل — يفتح بيه الموبايل الرابط على طول."""
    if not _cam_ok(request):
        raise HTTPException(status_code=401, detail="لازم تسجّل دخول")
    ip = primary_ip() or (all_ips() or ["127.0.0.1"])[0]
    url = f"https://{ip}:{PORT}/cam?k={CAM_KEY}"
    png = _qr_png(url)
    if not png:
        raise HTTPException(status_code=500, detail="مش قادر أعمل QR")
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


# ================== الرؤية بالحاسوب (إضافة مستقلة) ==================
#
# ⚠️ كل حاجة هنا في `vision.py` — الشغل الحقيقي هناك. هنا مسارات
# FastAPI بس. سيبنا `/video` زي ما هو: بياخد من `camera` مباشرة. ده
# مقصود — لو YOLO علِق (استدلاله ≈ ٨٨ms على CPU) البثّ مايتأثرش أبدًا.
#
# ⚠️ **دي معلومات للمشغّل، مش أمان.** العربية مالهاش حساس (الـ HC-SR04
# اتشال)، فمفيش فرملة تلقائية ولا تفادي. اقراها كـ«إيه قدامي».
vision_pipe = vision.VisionPipeline(camera)


# ---- AI Auto-Pilot (وحدة مستقلة، شوف vision_autopilot.py) ----
# ⚠️启用 بتفتح不应: العربية مفيهاش حساس (`ULTRASONIC_ENABLED=false` في الفيرموير)،
# فالكاميرا هي **كل** حاجز الأمام. عشان كده الـ enable بيرفض لما مينفعش يثبت
# إن الكاميرا شغّالة والتليامتري واصل.
autopilot = autopilot_mod.VisionAutopilot(
    get_pipeline=lambda: _get_vision(),
    get_telemetry=lambda: STATE["telemetry"],
    # نفس استدعاء الدوسة اللي الجايروسكوب بيستخدمه — نفس التوقيع ونفس الـ lease.
    start_manual=lambda: motion.start("manual", "autopilot", None, DASH_USER, "F"),
    motion=motion,
    on_event=log_event,
)
# Take the lane geometry and obstacle vocabulary from vision.py itself, so the
# autopilot can never drift away from what the operator sees on screen.
autopilot_mod.attach(vision)


def _get_vision() -> Optional[vision.VisionPipeline]:
    """
    الـ pipeline نفسه — **بعد** ما `lifespan` يشغّله.

    ⚠️ لازم المسارات تشوف نفس الـ instance اللي شغّال في الـ thريد، وإلا
    كل مسار هيعمل pipe جديد مافيشش فريم. فبنرجع `None` غير لما يكون
    فيه thread شغالة فعلًا.
    """
    return vision_pipe if vision_pipe._thread else None


# ⚠️ **ليه `_cam_ok` مش `require_auth`؟** (2026-09-29)
# لوحة الرؤية اتنقلت لـ `cam.html` — وهي صفحة الموبايل، وبيفتحوا
# بـ `?k=CAM_KEY` **من غير كوكي داشبورد**. فلو المسارات هنا كانت
# `require_auth`، كل طلب `/api/vision/*` من الموبايل كان هيبقى 401
# والتogglES مش هيشتغلوا خالص.
#
# `_cam_ok` بيقبل الاتنين: كوكي الداشبورد **أو** مفتاح الكاميرا. فبنفس
# مستوى أمان باقي مسارات الموبايل (`/go`، `/wscam`).
async def _vision_auth(request: Request) -> None:
    if not _cam_ok(request):
        raise HTTPException(status_code=401, detail="لازم مفتاح الكاميرا")


@app.get("/api/vision", dependencies=[Depends(_vision_auth)])
async def api_vision_status(request: Request):
    """حالة الرؤية: إعدادات، الموديل محمّل ولا لأ، آخر كشف."""
    pipe = _get_vision()
    if pipe is None:
        return {"ok": False, "error": "الرؤية مش شغالة لسه"}
    return {"ok": True, "vision": pipe.status()}


@app.post("/api/vision/config", dependencies=[Depends(_vision_auth)])
async def api_vision_config(request: Request):
    """
    تبديل التحسين/الضباب/الكشف من غير ما نعيد تشغيل السيرفر.

    المفاتيح (كلها اختيارية — اللي ما تبعتهوش بيفضل زي ما هو):
        `enhance`        CLAHE            (افتراضي true)
        `detect`         YOLO             (افتراضي true)
        `annotate`       رسم المربعات     (افتراضي true)
        `show_conf`      أرقام الثقة     (افتراضي false)
        `unsharp_mask`   unsharp masking  (افتراضي false)
        `skip`           كل فريم رقمه     (افتراضي 5)
        `conf`           عتبة الثقة 0..1
    """
    pipe = _get_vision()
    if pipe is None:
        return JSONResponse({"ok": False, "error": "الرؤية مش شغالة لسه"},
                            status_code=200)
    try:
        data = await request.json() or {}
    except Exception:
        return JSONResponse({"ok": False, "error": "مفيش JSON"},
                            status_code=200)
    if not isinstance(data, dict):
        return JSONResponse({"ok": False, "error": "مفيش JSON object"},
                            status_code=200)

    try:
        pipe.configure(**data)
    except (TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": f"إعداد غلط: {exc}"},
                            status_code=200)
    return {"ok": True, "vision": pipe.status()}


@app.get("/api/vision/detections", dependencies=[Depends(_vision_auth)])
async def api_vision_detections():
    """آخر الحاجات اللي YOLO شافها (نِسَب ٠..١٠٠ مش بكسل)."""
    pipe = _get_vision()
    if pipe is None:
        return JSONResponse({"ok": False, "error": "الرؤية مش شغالة لسه"},
                            status_code=200)
    dets = pipe.detections()
    # ---- The card's data, derived from the SAME list the boxes use ----
    # The lane/geometry rule lives in `vision.classify_path`, next to the
    # drawing code that uses the same constants. Re-implementing it in JSX
    # would give two answers that drift apart the moment PATH_LEFT_PCT
    # changes, so we compute it once, here, and let the browser render.
    info = vision.classify_path(dets)
    nearest = info["nearest"]
    return {
        "ok": True,
        "count": len(dets),
        "detections": [d.as_dict() for d in dets],
        "path_clear": bool(info["clear"]),
        "in_path": [d.label for d in info["in_path"]],
        "beside": [d.label for d in info["beside"]],
        "obstacles": [d.label for d in info["obstacles"]],
        "nearest": nearest.as_dict() if nearest else None,
    }


@app.get("/api/vision/frame.jpg", dependencies=[Depends(_vision_auth)])
async def api_vision_frame():
    """آخر فريم **بعد التحسين** (JPEG). مفيد للت debugging ورؤية الفرق."""
    pipe = _get_vision()
    if pipe is None:
        return JSONResponse({"ok": False, "error": "الرؤية مش شغالة لسه"},
                            status_code=200)
    data = pipe.frame()
    if not data:
        return JSONResponse(
            {"ok": False, "error": "مفيش فريم محسّن لسه — في صورة جاية؟"},
            status_code=200)
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/vision/describe", dependencies=[Depends(_vision_auth)])
async def api_vision_describe():
    """
    وصف لما قدام العربية، من **آخر فريم محسّن + آخر كشف**.

    ⚠️ **مش كابشن بالمعنى الكامل** — مفيش VLM. ده و��اقع هندسي:
    إيه اللي اتشاف، أنهي جزء من الصورة، وقربه، وهل المسار قدامك خالص.
    بكون أدق وأسرع من تخمين على شكل المشهد.
    """
    pipe = _get_vision()
    if pipe is None:
        return JSONResponse({"ok": False, "error": "الرؤية مش شغالة لسه"},
                            status_code=200)
    if not pipe.frame() and not pipe.detections():
        return JSONResponse(
            {"ok": False,
             "error": "مفيش فريم لسه — في صورة جاية من /cam؟"},
            status_code=200)
    out = vision.describe(pipe.detections())
    out["ok"] = True
    return out


@app.get("/api/vision/annotated.mjpeg", dependencies=[Depends(_vision_auth)])
async def api_vision_stream():
    """
    بثّ MJPEG **واحد**: الصورة المحسّنة + مربعات YOLO مرسومة فوقها.

    ⚠️ **ليه MJPEG مش WebSocket؟** العميل ده `<img src=...>` وال-browser
    بيتعامل معاه بشكل طبيعي، وفيه reconnect تلقائي لو الشبكة قطعت. ده
    أخف من WebSocket هنا — البثّ one-way أصلًا.

    ⚠️ **صفر تأخير**: نفس مبدأ `camera` — بنقارن رقم الفريم وبنبعت
    الجديد بس. لوEnhancementتأخر، الـ`<img>` بيعرض آخر متاح ويتحدّث
    أول ما الجاي. **مفيش طابور**، فمفيش إطارات متأخرة تتراكم.
    """
    async def gen():
        last = -1
        while True:
            seq = vision_pipe.frame_seq()
            if seq != last:
                data = vision_pipe.frame()
                if data:
                    last = seq
                    yield (
                        b"--" + BOUNDARY.encode() + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(data)).encode()
                        + b"\r\n\r\n" + data + b"\r\n"
                    )
                    continue
            # ⚠️ `sleep(0)` مش `yield` فاضي: الـ yield الفاضي بيبعت
            # بايتات لـ StreamingResponse وهي بتعمل flush — ده حمل
            # فاضي على الشبكة. السleep هو اللي بيمنع استهلاك المعالج.
            await asyncio.sleep(0.03)
    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


# ================== اسأل الرؤية (Google Gemini — SDK رسمي) ==================
#
# ⚠️ ده **مش** نفس `/api/vision/describe`. الوصف اللي فوق بيقول "الحاجات
# اللي YOLO شافها" — أرقام وقواعد، مفيش فيه أي فهم. ده بيبعت **الصورة
# نفسها** لنموذج متعدد الوسائط، فبيقدر يجاوب على أي سؤال: "فيه حاجة في
# إيدّي؟" "الليارة دي واقفة ولا ماشية؟"
#
# موديلين في نفس الشاشة:
#   · `/api/vision/ask`     → آخر فريم من الكاميرا (JPEG مخزّن) → رد فوري
#   · `/api/vision/analyze` → ملف فيديو كامل → تحليل زمني (File API)
#
# ⚠️ **ليه بنبعت آخر فريم اللي عندنا أصلًا؟** `vision_pipe.frame()` بترجع
# JPEG **بعد التحسين** (ومرسوم عليه المربعات لو annotate شغّال). دي برضه
# الصورة اللي المشغّل بيشوفها على الشاشة، فالإجابة بتتكلم عن نفس اللي شايفه
# هو بالظبط. العكس (نصوّر تاني من `camera`) معناه نبعتله صورة تانية غير
# اللي على الشاشة، فممكن يقول "مفيش حاجة" والعربية واقف قدام القطة في
# الصورة اللي هو بيبص فيها.
#
# ⚠️ **ليه بنقرأ المفتاح من `os.environ` هنا مش من `settings`؟** `settings`
# بيعمل `os.environ.setdefault` لكل سطر في `.env` وقت الاستيراد، فلو قراناه
# من `settings.CFG` كان لازم نضيف اسم المفتاح بسطر جديد هناك. ده بيشتغل
# عشان الاستيراد حصل فوق.
#
# ⚠️ **`strip("'\"")` مش تجميل.** `settings.py` بيقرا `.env` بـ split على
# أول `=` من غير أي تنضيف، فـ `GEMINI_API_KEY="AQ.x"` بتوصلنا ومعاها
# علامتا تنصيص جوّه المفتاح — وجوجل بيرد `API_KEY_INVALID` على مفتاح
# صحيح شكله غلط. كل سطر تاني في `.env` بتوعنا مكتوب من غير علامات، فده
# مش تليين: ده اللي يخلّي الصيغة المقروءة (علامات) والصيغة المشغّلة
# (من غير) نفس الحاجة.
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip().strip("'\"")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip().strip("'\"")

# ⚠️ **ليه بديل أصلاً؟** `gemini-3.8-flash` — الموديل المطلوب — رجّع
# 503 UNAVAILABLE "high demand" وقت ما اتكتب ده (اتأكدتفعلاً: نفس
# المفتاح، نفس الطلب، `gemini-3.5-flash` رجّع إجابةصححة). الموديل
# موجود ومش غلط، بس مش متاح دلوقتي. فبنجربه الأول، وبعده البديل.
#
# ⚠️ **بنجرب البديل على 503/UNAVAILABLE بس.** 400/401 معناها الطلب أو
# المفتاح غلط — تغيير الموديل مش هينفع، وwenخسر الوقت ونخفي الحقيقة
# عن المستخدم. نفس قاعدة OpenRouter اللي كانت موجودة قبل.
GEMINI_FALLBACK_MODEL = os.environ.get(
    "GEMINI_FALLBACK_MODEL", "gemini-3.5-flash").strip().strip("'\"")
GEMINI_MODEL_FALLBACKS = list(dict.fromkeys(
    m for m in (GEMINI_MODEL, GEMINI_FALLBACK_MODEL) if m))

# كم محاولة لكل موديل، والفاصل بينهم بيزيد (٢/٤/٨ ثواني). ٥xx من جوجل
# («high demand») بيعدّي لوحده، بس مش دايماً في نص ثانية.
GEMINI_RETRIES = int(os.environ.get("GEMINI_RETRIES", "3"))
GEMINI_RETRY_BACKOFF = float(os.environ.get("GEMINI_RETRY_BACKOFF", "2"))

# سقف صارم على طول السؤال (base64 جوّه JSON ⇒ payload ضخم بلا فايدة).
ASK_MAX_PROMPT = 500

# سؤال الفيديو أطول بكتير: عايزين تاريخ زمني وتوصيفات، مش سطرين.
VIDEO_MAX_PROMPT = 2000

# سقف الرفع. الـ File API نفسه بيسحب لحد 2GB، بس ده **سيرفر على لابتوب**
# والـ bytes كلها بتتمشى في الذاكرة (نكتب tempfile ونمشي) — 200 ميجا سقف
# معقول: فيديو دقايق من كاميرا موبايل أصغر من كده بكتير.
VIDEO_MAX_BYTES = 200 * 1024 * 1024

# مهلة انتظار `state == ACTIVE`. الرفع نفسه بياخد ثواني، بس التحويل
# (transcode) لأصيغ التحليل بياخد وقت على الفيديوهات الطويلة.
VIDEO_ACTIVE_TIMEOUT = 180.0
VIDEO_POLL_INTERVAL = 2.0

# الامتدادات اللي الـ File API بيعرف يحوّلها. الرفض هنا **قبل** الرفع عشان
# ما نرفعش 200 ميجا عشان نكتشف بعدين إن الصيغة مش مدعومة.
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".mpeg", ".mpg", ".3gp"}

# ⚠️ `from google import genai` **مستنى لحد أول استخدام**، مش وقت الاستيراد.
# نفس مبدأ الـ psycopg2 والـ ultralytics فوق: لو الحزمة مش مثبتة، السيرفر
# لازم يفضل واقف ويشتغل — ويبقى فشل الرؤية رسالة للمستخدم مش موت للسيرفر.
_GEMINI_CLIENT = None


def _gemini():
    """عميل واحد لكل العملية — الإنشاء بيعمل TLS handshake."""
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        from google import genai
        _GEMINI_CLIENT = genai.Client(api_key=GEMINI_KEY)
    return _GEMINI_CLIENT


def _file_state(f) -> str:
    """
    `File.state` بيرجع قيمة enum في الـ SDK، فبيبقى `"FileState.ACTIVE"`
    أو `"ACTIVE"` حسب الإصدار. بنقصّ آخر نقطة عشان نضمن مقارنة واحدة.
    """
    return str(getattr(f, "state", "") or "").rsplit(".", 1)[-1].upper()


# نفس عُرف باقي مسارات `/api/vision/*`: لو مش متظبظ، رجّع 200 بـ
# `{"ok": false, "error": ...}` — الـ frontend بيقرا `ok` أصلاً ومش
# بيرمي استثناء على 401/500.
def _ask_fail(msg: str, code: int = 200) -> JSONResponse:
    return JSONResponse({"ok": False, "error": msg}, status_code=code)


# ⚠️ **`system_instruction` مش رسالة `role: system` جوّه `contents`.** الـ SDK
# بيفصلها: `contents` = اللي المستخدم بعتاه (الفيديو/الصورة + السؤال)،
# و`config.system_instruction` = التعليمات الثابتة. خلط الاتنين كان هيسلّم
# النموذج كتلتين نصوص بلا صورة.
ASK_SYSTEM = (
    "إنت مساعد رؤية حاسوبي في غرفة تحكم عربية ذكية (I-VOLT).\n"
    "- جاوب **باختصار** (سطرين بالكتير) وبأدق حاجة تشوفها.\n"
    "- ردّ **بنفس لغة السؤال**: لو السؤال عامية مصرية ردّ عامية مصرية، "
    "لو إنجليزي ردّ إنجليزي.\n"
    "- لو مش واضح في الصورة، قول كده بصراحة. **ماتخمنش** ولا تحاجي حاجة "
    "ملقتهاش.\n"
    "- لو فيه حاجات بعيدة أو مختفية ورا حاجة تانية، ماتقدّمش لها رأي.\n"
    "- ⚠️ دي **معلومة للمشغّل مش أمان**: العربية مالها حساس مسافة، فما فيش "
    "حاجة بتعرف تتفادى عن طريقها."
)

# ⚠️ تحليل الفيديو **مختلف جوهرياً** عن سؤال الفريم: هنا المطلوب زمني مش
# لوصف لقطة واحدة. من غير التعليمات دي، النموذج بيلخّص «عمل إيه في
# الفيديو» في نص سطرين وميضيع كل التفاصيل اللي المستخدم رافعه عشانها.
VIDEO_SYSTEM = (
    "إنت محلل فيديو محترف بتشتغل على تسجيلات كاميرا مراقبة، والمستخدم "
    "بيتكلم معاك **عربي مصري**.\n"
    "المطلوب **تحليل زمني مفصّل**، مش وصف عام:\n"
    "- امشي على الفيديو **بالترتيب الزمني**، واذكر **اللحظات المهمة مع "
    "وقتها بالثواني** (مثال: «في الثانية ١٢»). من غير توقيت، الرد بيبان "
    "كويس بس عمليتهوش حاجة.\n"
    "- **اقرا أي كتابة صغيرة** في الفيديو: نص على أي جسم، أرقام، كلام على "
    "لافتة، وأي حاجة **في إيد** حد. اكتبها حرفياً زي ما شايفها، ولو مش "
    "مقروءة قول «مش واضح» بدل ما تخمّن.\n"
    "- **ألوان الهدوم** بالتفصيل: اللون الأساسي وأي لون تاني فيه.\n"
    "- **تغيّر الحالة**: حاجة امتلأت أو فرغت (كوباية، علبة، قنينة)، باب "
    "اتفتح أو اتقفل، حد دخل أو خرج، حاجة اتحطت أو اتشالت. وقل **إمتى** "
    "الحاجة دي اتغيرت.\n"
    "- لو فيه أكتر من شخص متحرك، افصل بينهم واحد واحد.\n"
    "- **ما تخمّنش** أي حاجة مش شايفها بعينك. لو مش واضح، قول كده صريح.\n"
    "اكتب الرد **منظّم**: نقاط أو فقرات قصيرة، وتحت عنوان لكل جزء."
)


def _generate(parts: list, prompt: str, system: str):
    """
    نداء واحد للـ SDK، مشترك بين الفريم والفيديو. **متزامن (blocking)** —
    لازم يتنادى من `asyncio.to_thread` عشان ما يوقفش حلقة الأحداث.
    بيرجع `(answer, error)` — واحد فيهم لازم يتظبط دايماً.
    """
    try:
        from google.genai import types
    except Exception as exc:  # الحزمة مش مثبتة
        return None, f"مكتبة google-genai مش متاحة ({type(exc).__name__})"

    # ---- موديل ورا موديل، وكل موديل عدّة محاولات ----
    #
    # ⚠️ **ليه إعادة المحاولة أصلاً؟** ده **مش تخيّل** — جوجل بترجّع
    # `503 UNAVAILABLE / "high demand"` على نفس المفتاح ونفس الطلب في نص
    # إندكشة، وجوهرتها عادية بعد ثانية. من غير إعادة محاولة، أول ضربة 5xx
    # كانت هتخلّي المستخدم يشوف «النموذج رجّع خطأ» على حاجة كانت هتنجح.
    # ٣ محاولات بفترات ٢/٤/٨ ثواني بتغطي النافذة دي.
    client = _gemini()
    resp = None
    failure = None
    for model in GEMINI_MODEL_FALLBACKS:
        for attempt in range(GEMINI_RETRIES):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=parts + [types.Part.from_text(text=prompt)],
                    config=types.GenerateContentConfig(system_instruction=system),
                )
                break
            except Exception as exc:
                failure = exc
                # ⚠️ `type(exc).__name__` بس، مش `exc` نفسه: ممكن الـ str
                # يطلع فيه الـ URL وفيه المفتاح. زي باقي مسارات الرؤية.
                #
                # ⚠️ **`ServerError` وحده يستاهل إعادة المحاولة.** هو 5xx
                # من جوجل (الموديل مشغول). أي حاجة تانية = الطلب أو
                # المفتاح غلط، فإعادة المحاولة هتضيّع وقت المستخدم بس.
                if type(exc).__name__ != "ServerError":
                    return None, f"النموذج رجّع خطأ ({type(exc).__name__})"
                if attempt < GEMINI_RETRIES - 1:
                    time.sleep(GEMINI_RETRY_BACKOFF * (attempt + 1))
        if resp is not None:
            break

    if resp is None:
        return None, (
            f"النموذج مشغول دلوقتي — استنى شوية وجرب تاني "
            f"({type(failure).__name__})" if failure is not None
            else "النموذج مش راد")

    answer = (getattr(resp, "text", None) or "").strip()
    if not answer:
        # نموذج بيقفل من غير سبب بيرجع فاضي أو بيحط `None` مكان النص.
        try:
            why = (resp.candidates[0].finish_reason.name if resp.candidates else "no candidates")
        except Exception:
            why = "empty"
        return None, f"النموذج رجّع فاضي ({why})"
    return answer, None


@app.post("/api/vision/ask", dependencies=[Depends(_vision_auth)])
async def api_vision_ask(request: Request):
    """
    اسأل على **آخر صورة من الكاميرا**، واج الرد يجي من Gemini.

    body: `{"prompt": "…"}`

    ⚠️ زي باقي مسارات الرؤية: `_vision_auth` بيقبل كوكي الداشبورد **أو**
    مفتاح الكاميرا. مش حماية وحدها — كل اللي بتعمله بيه صورة العربية
    قدامك، ومفيش فيه حركة ولا أمر.
    """
    if not GEMINI_KEY:
        return _ask_fail("مفيش مفتاح Gemini في .env — اسأل الأدمن")

    # ---- الطلب نفسه ----
    try:
        data = await request.json() or {}
    except Exception:
        return _ask_fail("مفيش JSON")
    if not isinstance(data, dict):
        return _ask_fail("مفيش JSON object")
    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        return _ask_fail("اكتب السؤال الأول")
    if len(prompt) > ASK_MAX_PROMPT:
        prompt = prompt[:ASK_MAX_PROMPT]

    # ---- الصورة ----
    pipe = _get_vision()
    if pipe is None:
        return _ask_fail("الرؤية مش شغالة لسه")
    frame = pipe.frame()
    if not frame:
        return _ask_fail("مفيش فريم لسه — في صورة جاية من الكاميرا؟")

    # ⚠️ `frame` JPEG أصلاً (محسّن، ومرسوم عليه المربعات)، فمفيش encode
    # تاني — base64 على اللي عندنا وخلاص.
    #
    # ⚠️ صور الموبايل 1080p ≈ ١٥٠KB. سقف ٤٠٠KB (أكبر بكتير من اللي
    # بنبعته) عشان حد قال "hi" مايبعتش طلب ٥ ميجا والـ JSON parse في
    # الذاكرة.
    if len(frame) > 400_000:
        return _ask_fail("الصورة كبيرة أوي — استنى ثانية وجرب تاني")

    try:
        from google.genai import types
        parts = [types.Part.from_bytes(data=frame, mime_type="image/jpeg")]
    except Exception as exc:
        return _ask_fail(f"مكتبة google-genai مش متاحة ({type(exc).__name__})")

    answer, err = await asyncio.to_thread(_generate, parts, prompt, ASK_SYSTEM)
    if err:
        return _ask_fail(err)
    return {"ok": True, "answer": answer}


def _analyze_video(data: bytes, mime: str, ext: str, prompt: str):
    """
    رفع → انتظار `ACTIVE` → سؤال → حذف. **متزامن**، يتنادى بـ
    `asyncio.to_thread`.

    ⚠️ **ليه ملف tempfile والـ bytes مش مباشرة؟** `files.upload()` بياخد
    مسار أو `IOBase` — **مش** `bytes` — والـ `mime_type` بتروح في
    `config` مش named argument. وحتى لو قبل الـ bytes، الـ SDK بيشوف
    نوع الملف من الامتداد عشان يحدّد الـ mime. فالملف المؤقت بالامتداد
    الصح (`.mp4` / `.mov` / …) هو الشكل اللي الـ API متوقعه.

    ⚠️ **الحذف — البعيد والح المحلي — في `finally` مش اختياري.** أي ملف
    مرفوع على الـ File API بيفضل عند جوجل ٤٨ ساعة، والملف ده فيديو
    **من بيت العميل**. و`_generate` جوه نفس الـ `try` عشان الحذف ما
    يحصلش **قبل** ما النموذج يقرأ الفيديو.
    """
    try:
        from google.genai import types
    except Exception as exc:
        return None, f"مكتبة google-genai مش متاحة ({type(exc).__name__})"

    client = None
    uploaded = None
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as fh:
            fh.write(data)
            tmp = fh.name

        client = _gemini()
        uploaded = client.files.upload(
            file=tmp, config=types.UploadFileConfig(mime_type=mime))

        deadline = time.time() + VIDEO_ACTIVE_TIMEOUT
        while _file_state(uploaded) != "ACTIVE":
            if _file_state(uploaded) == "FAILED":
                return None, "جوجل رفض الفيديو ده — جرّب صيغة تانية (mp4)"
            if time.time() > deadline:
                return None, (
                    f"الفيديو اتحمّل بس مابقاش جاهز بعد "
                    f"{int(VIDEO_ACTIVE_TIMEOUT)} ثانية — جرّب فيديو أقصر"
                )
            time.sleep(VIDEO_POLL_INTERVAL)
            uploaded = client.files.get(name=uploaded.name)

        parts = [types.Part.from_uri(
            file_uri=uploaded.uri, mime_type=uploaded.mime_type or mime)]
        return _generate(parts, prompt, VIDEO_SYSTEM)
    except Exception as exc:
        return None, f"رفع الفيديو فشل ({type(exc).__name__})"
    finally:
        if client is not None and uploaded is not None:
            with contextlib.suppress(Exception):
                client.files.delete(name=uploaded.name)
        if tmp:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


@app.post("/api/vision/analyze", dependencies=[Depends(_vision_auth)])
async def api_vision_analyze(file: UploadFile = File(...),
                             prompt: str = Form("")):
    """
    حلّل **ملف فيديو** كامل، مش لقطة واحدة.

    form: `file` (الفيديو) + `prompt` (السؤال، اختياري).

    الرفع بياخد ثواني، والانتظار لحد `ACTIVE` بياخد أكتر، وطلب النموذج
    بياخد الباقي — فالمجموع **دقيقة لثلاث على الفيديو القصير**، وعايز
    استرجاع مريح في الواجهة (س핀ر مش شريط تقدم). الرجع نفس عُرف
    `/api/vision/ask`: ٢٠٠ بـ `{"ok": …, "answer" | "error"}`.
    """
    if not GEMINI_KEY:
        return _ask_fail("مفيش مفتاح Gemini في .env — اسأل الأدمن")

    name = (file.filename or "").strip()
    ext = Path(name).suffix.lower()
    if ext not in VIDEO_EXTS:
        return _ask_fail(
            "صيغة الفيديو مش مدعومة (" + ", ".join(sorted(VIDEO_EXTS)) + ")")

    prompt = (prompt or "").strip()
    if not prompt:
        prompt = "حلّل الفيديو ده بالتفصيل: إيه اللي حصل فيه، بالترتيب الزمني."
    if len(prompt) > VIDEO_MAX_PROMPT:
        prompt = prompt[:VIDEO_MAX_PROMPT]

    # `file.size` مش متاح دايماً (بتبع من الـ headers) — فالفحص الحقيقي هو
    # اللي بنقراه بالظبط. بنقرا بالبلوكات ونقفل أول ما نعدّي السقف، عشان
    # ما نبعتش ٥ جيجا في الذاكرة عشان نرفضها في النهاية.
    cap = VIDEO_MAX_BYTES
    buf = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > cap:
            return _ask_fail(
                f"الفيديو أكبر من {cap // (1024 * 1024)} ميجا — قصّره وجرب تاني")
    if not buf:
        return _ask_fail("الملف فاضي")

    mime = (file.content_type or "").split(";")[0].strip() or "video/mp4"
    if not mime.startswith("video/"):
        mime = f"video/{ext.lstrip('.') or 'mp4'}"

    answer, err = await asyncio.to_thread(
        _analyze_video, bytes(buf), mime, ext, prompt)
    if err:
        return _ask_fail(err)
    return {"ok": True, "answer": answer}


@app.get("/video")
async def video(request: Request):
    if not _cam_ok(request):
        raise HTTPException(status_code=401, detail="لازم تسجّل دخول")
    return StreamingResponse(
        _mjpeg(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


# ---------- WebSocket ----------
@app.websocket("/wscam")
async def wscam(ws: WebSocket):
    if not _cam_ok_ws(ws):
        await ws.close(code=1008)
        return
    await ws.accept()
    log_event("📱 كاميرا الموبايل اتصلت")
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data:
                camera.push_remote(data)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        log_event("📱 كاميرا الموبايل فصلت")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    if not _cam_ok_ws(ws):
        await ws.close(code=1008)
        return

    await manager.connect(ws)
    try:
        await ws.send_text(json.dumps({"type": "motion", **motion.status()}, default=str))
        await ws.send_text(json.dumps({"type": "checkpoint_run",
                                       **checkpoint_run.status()}, default=str))
        tele = STATE["telemetry"]
        if tele:
            await ws.send_text(json.dumps({"type": "telemetry", **_compat_telemetry(tele)},
                                          ensure_ascii=False, default=str))

        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except ValueError:
                continue

            kind = msg.get("type")

            if kind == "motion_start":
                # ⚠️ لو فيه أمر شغال من مصدر تاني (دورية من الباك إند،
                # أو جلسة توجيه) — المشغّل اليدوي مايخطفهمش.
                if motion.command_id and motion.kind not in (None, "manual"):
                    await ws.send_text(json.dumps(
                        {"type": "event",
                         "msg": f"⚠️ فيه {motion.kind} شغالة — "
                                f"مش هتخطفها بالتحكم اليدوي"}))
                    await manager.broadcast({"type": "motion", **motion.status()})
                elif not motion.command_id:
                    # ⚠️ بنمرّر الاتجاه **زي ما هو** (مش `or "F"`): لو
                    # غلط/فاضي، `start` بترفضه وبنقول للمشغّل. زمان كان
                    # `or "F"` بيطلّع **نبضة قدّم** من غير ما حد يقصد.
                    res = motion.start(
                        str(msg.get("kind") or "manual"),
                        str(msg.get("route_id") or "manual"),
                        msg.get("incident_id"),
                        DASH_USER,
                        str(msg.get("direction") or ""),
                    )
                    if res.get("error") == "bad_direction":
                        await ws.send_text(json.dumps(
                            {"type": "event",
                             "msg": "⛔ اتجاه غير مفهوم — مفيش حركة "
                                    "(استخدم أسهم التحكم)"}))
                    await manager.broadcast({"type": "motion", **motion.status()})
                else:
                    motion.keepalive()

            elif kind == "motion_keepalive":
                # المشغّل لسه دايس على الزرار
                motion.keepalive()

            elif kind == "motion_input":
                # ⚠️ ذراع الموبايل (الجايرو) — تعديل على أمر **قائم**،
                # مش أمر جديد (شوف `set_gyro_input` للسبب). بيتبعت
                # ٢٥ مرة في الثانية، فالحلقة نفسها ما بتعملش حاجة
                # غير بتجدد الـ lease الموجود.
                if not motion.command_id:
                    # لسه مافيش أمر يدوي شغال — أول قراءة جايرو بتبدأ
                    # واحد عادي بنفس مسار الأسهم بالظبط.
                    motion.start("manual", "manual", None, DASH_USER,
                                 str(msg.get("steer") or "F"))
                    if motion.command_id:
                        motion.set_gyro_input(msg.get("throttle"),
                                              msg.get("steer"))
                elif not motion.set_gyro_input(msg.get("throttle"),
                                                msg.get("steer")):
                    # رفض (أمر مش يدوي، أو نسبة/اتجاه مش مفهوم) —
                    # نقف فورًا بدل ما نكمّل على آخر قراءة سليمة.
                    motion.stop(reason="gyro_input_rejected")
                    await ws.send_text(json.dumps(
                        {"type": "event",
                         "msg": "⛔ قراءة جايرو مرفوضة — وقف العربية"}))
                # ⚠️ **مانبعثش** `motion` كل قراءة: ٢٥ رسالة/ثانية لكل
                # عميل متصل (والإلخ) هو بالظبط الطوفان اللي الطلب
                # بيحاول يتجنّبه. الأرقام الحية بتيجي مع
                # `telemetry`؛ والدورة (بدأ/وقف) بس هي اللي بتنشر.
                if not motion.command_id:
                    await manager.broadcast({"type": "motion",
                                             **motion.status()})


            elif kind == "motion_stop":
                # ⚠️ ده "سيبت الزرار" بتاع المشغّل اليدوي — **ماينفعش**
                # يوقف دورية معتمدة من الباك إند ولا جلسة توجيه.
                # ده كان باچ حقيقي: تاب داشبورد مفتوح في المتصفح بيفقد
                # التركيز → يبعت motion_stop → الدورية تقع في نصها.
                # عايز توقف دورية؟ ده ليه /api/patrol/stop.
                if motion.command_id and motion.kind not in (None, "manual"):
                    log_event(f"↩️ المشغّل ساب الزرار — "
                              f"الأمر {motion.command_id} ({motion.kind}) "
                              f"مكمل (مش أمره)")
                else:
                    motion.stop(reason="operator_release")
                await manager.broadcast({"type": "motion", **motion.status()})

            elif kind == "checkpoint":
                cp_id = str(msg.get("checkpoint_id") or "").strip()
                if cp_id:
                    publish_checkpoint(cp_id, motion.route_id or "route_a",
                                       motion.incident_id, 0, DASH_USER)
                    log_event(f"تأكيد نقطة تفتيش من المشغّل: {cp_id}")

            elif kind == "config":
                payload: dict[str, int] = {}
                if "speed" in msg:
                    # ⚠️⚠️ السرعة دي بقت **سرعة القيادة اليدوية** (نسبة من
                    # سقف التوجيه 170) وبتتخزن عند الأدابتر — و**مابنبعتهاش**
                    # للعربية كـ `motorSpeed` تاني.
                    #
                    # ليه؟ لأن `motorSpeed` في الفيرموير هو **السقف المطلق**
                    # (0–255)، ولو خلّيناه 255 تبقى أي دوسة انطلاق بالكامل
                    # (شكوى المشغّل: «بتشتغل على أعلى سرعة ومعرفش أتحكم
                    # فيها»). دلوقتي السرعة بتتحدد من الـ lease (نسبة مقيّدة)
                    # والفيرموير بيسقّفها لوحده كمان.
                    chosen = motion.set_manual_speed(msg["speed"])
                    log_event(f"🎚️ سرعة القيادة اليدوية → {chosen} / "
                              f"{motion.MANUAL_SPEED_MAX}")
                if "stop" in msg:
                    payload["stop"] = max(10, min(120, int(msg["stop"])))
                if "auto_avoid" in msg:
                    payload["auto_avoid"] = 1 if msg["auto_avoid"] else 0
                for g in ("gain_l", "gain_r"):
                    if g in msg and _clamp_gain(msg[g]) is not None:
                        payload[g] = _clamp_gain(msg[g])
                if payload:
                    # ⚠️ الإعدادات بتغيّر السرعة ومسافة الفرملة — يعني
                    # مسألة أمان، فبتتوقّع برضه. القيم الناقصة بتتوقّع
                    # كـ -1 (يعني "زي ما هو") والفيرموير بيفهمها كده.
                    payload["signature"] = cmd_sign.sign_config(
                        payload.get("speed", -1), payload.get("stop", -1),
                        payload.get("auto_avoid", -1),
                        payload.get("gain_l", -1), payload.get("gain_r", -1))
                    publish("config", json.dumps(payload))

            elif kind == "hello":
                await ws.send_text(json.dumps({"type": "motion", **motion.status()},
                                              default=str))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await manager.disconnect(ws)


# ================== التشغيل ==================
if __name__ == "__main__":
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import uvicorn

    certs = None
    if os.environ.get("HTTPS", "1") != "0":
        try:
            from make_cert import covers, generate, local_ips

            cert_file = HERE / "certs" / "cert.pem"
            key_file = HERE / "certs" / "key.pem"
            ips = local_ips()
            if not covers(cert_file, ips):
                generate(cert_file, key_file)
                print("  اتولّدت شهادة HTTPS جديدة")
            certs = (cert_file, key_file)
        except Exception as exc:  # noqa: BLE001
            print(f"  (مش قادر أعمل شهادة HTTPS: {exc}) — هشتغل HTTP")

    if certs:
        uvicorn.run(app, host=HOST, port=PORT, log_level="warning",
                    ssl_certfile=str(certs[0]), ssl_keyfile=str(certs[1]))
    else:
        uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
