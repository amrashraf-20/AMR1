"""
visual_servo.py  —  «الكاميرا تشوف العربية، وأنا أسوقها للحتة»
============================================================
دي الطريقة اللي فيها **صفر معايرة**:

  • مفيش ارتفاع، مفيش ميل، مفيش حساس موبايل، مفيش مسطرة، مفيش متر.
  • المشغّل بيدوس على **العربية** في الصورة، وبعدين على **الحتة**.
  • إحنا بنتبّع الاتنين كل لحظة، ونسوق العربية في **إطار الصورة**
    نفسه: نحرّكها لحد ما بقعتها توصل لبقعة الحتة.

المفاتيح اللي بتخلي ده يشتغل من غير أي معايرة:

  ١) **بنشوف الحركة مش بنحسبها**: اتجاه بوز العربية بنعرفه من
     إزاحتها الحقيقية بين صورتين (بعد ما نشيل حركة الموبايل).
  ٢) **حركة الموبايل بتتعوّض**: بقعة الحتة ثابتة على الأرض، فأي
     حركة بتظهر فيها = حركة الموبايل. بنشيلها من حركة العربية.
  ٣) **الوصول بالمقارنة مش بالمتر**: العربية وصلت لما بقعتها تبقى
     جنب بقعة الحتة — والنسبة دي مش محتاجة أي مقياس.
  ٤) **الموبايل بيتحرك كتير؟ بنستنى**: لو الإزاحة العامة كبيرة،
     بنوقف لحد ما المشغّل يثبّت إيده، وبعدين نكمّل.

الملف ده **مستقل**: مفيش شبكة ومفيش هاردوير. بيتّختبر بمحاكاة
كاميرا كاملة (test_visual_servo.py) قبل ما يتوصل بأي حاجة.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ================== الثوابت ==================

# بنشتغل على إطار مصغّر عشان السرعة (إحداثيات الكل هنا في الإطار ده)
WORK_W = 480

# مقاس العينة اللي بنتبّعها (نص الضلع بالبكسل)
PATCH_HALF = 22
# المقاسات اللي بنجرّبها (عشان العربية تقرب وتبعد)
SCALES = (0.4, 0.5, 0.62, 0.78, 1.0, 1.25, 1.55, 1.9, 2.3)

# أقل درجة تطابق نقبلها
MIN_SCORE = 0.42
# بنحدّث العينة بس لو التطابق قوي (عشان ما نسرحش ورا الغلط)
UPDATE_SCORE = 0.72

# منطقة البحث حوالين آخر مكان (بكسل)
SEARCH = 70

# وصلنا لما الباقي يبقى أقل من **خطوة واحدة** من خطوات العربية نفسها.
# ⚠️ ده مش مقياس ثابت: بنقيس الخطوة في نفس الصورة وفي نفس اللحظة،
# فمهما كان البعد أو الزاوية، «خطوة» بتفضل خطوة. وده اللي بيخلي
# الدقة ثابتة من غير أي معايرة — وأدق بكتير من نسبة حجم البقعة.
ARRIVE_STEP_RATIO = 0.8
ARRIVE_MIN_PX = 6.0
# ⚠️ سقف للعتبة: لو تقدير الخطوة طلع كبير غلط (ضجيج/حركة إيد)،
# مابنسمحش للعتبة تكبر لدرجة إنها تعلن الوصول وهي بعيدة.
ARRIVE_MAX_PX = 20.0
# لازم شرط الوصول يتحقق كام دورة ورا بعض (عشان ما نصدّقش قراءة واحدة)
ARRIVE_CONFIRM = 2
# لو بعدنا كده عن أقرب نقطة وصلنالها → شاردة → وقف
DIVERGE_FACTOR = 2.5
DIVERGE_CONFIRM = 50
# لسه مش عارفين حجم الخطوة؟ نستخدم ده مؤقتًا
ARRIVE_FALLBACK_PX = 12.0
# قربنا أوي؟ نبضات أنعم (اقتراب نهائي)
FINE_ZONE = 2.5
PULSE_FINE = 0.30
SPEED_FINE = 200
# عدد الخطوات اللي بناخد وسيطها
STEP_MEDIAN_N = 5
STEP_MIN_PX = 2.0
STEP_MAX_PX = 60.0

# زاوية أكبر من كده → نلف بدل ما نمشي
TURN_ANGLE_DEG = 22.0

# لو الموبايل اتحرك أكتر من كده في اللحظة → نستنى
CAMERA_MOVE_LIMIT = 26.0

# إزاحة أقل من كده مابنحسبهاش حركة (ضجيج)
MIN_MOTION_PX = 2.0

# إزاحة كبيرة أوي معناها إن الاتجاه اتغيّر (لفّة) — بنتجاهلها في التقدير
MAX_MOTION_PX = 120.0

# البطولات (ثواني)
PULSE_FORWARD = 2.0
PULSE_TURN = 0.50
PULSE_SENSE = 0.50

# السرعات (نفس سقف الفيرموير)
SPEED_FORWARD = 255
SPEED_TURN = 255

# حد أقصى للمشوار كله (أمان) — بالثواني واللفّات
MAX_SECONDS = 90.0
MAX_CYCLES = 400

# ===== حاجز ضد «اللف حوالين نفسها» =====
# ⚠️ اتشاف على الأرض (العربية الحقيقية): لو تقدير بوز العربية باظ،
# السيرفو يفضل يقلب اتجاه اللف **يمين/شمال/يمين/شمال** بسرعة —
# يعني العربية بتتهز في مكانها لحد الـ ٩٠ ثانية.
# العلامة المميزة مش عدد اللفّات ولا الوقت (المشوار السليم بياخد وقت
# ولفّات كتير) — العلامة هي **قلب الاتجاه**: مشوار سليم بيلف ناحية
# واحدة (أو قلبة واحدة)، واللفة الحوالين نفسها بتقلب كل مرة.
TURN_REVERSALS_MAX = 20
PROGRESS_MIN_PX = 3.0       # «تقدّم حقيقي» = قربنا كده عن آخر نقطة سجّلناها
# نفس المعنى بس بالمتر — ودي **المرجع** لو معانا موديل الكاميرا
PROGRESS_MIN_M = 0.05
# عتبة الوصول الحقيقية بالمتر (زي التوجيه بالظبط: 0.35م)
ARRIVE_M = 0.10
# ⚠️ أقصى مسافة **حقيقية** نقبل عندها إن البكسل يقول «وصلنا». البكسل
# لوحده بيكدب وهو بعيد (منظور)، وده الحاجز اللي بيمنع «وصلت» كدب.
ARRIVE_SANITY_M = 1.0
# ⚠️ تنعيم مسافة القرار: اهتزاز الموبايل بيخلي القراءة تنط، والقرار
# على قراءة واحدة بيبقى متقلّب (وصلنا/لسه). بناخد وسيط آخر كام قراءة.
DIST_MEDIAN_N = 3
# ⚠️ حاجز «مفيش تقدّم»: لو مشينا كده دورة **وفيهم** كده نبضة قدّم من غير
# أي تقدّم حقيقي → بنقف. ده اللي بيمنع العربية إنها تشرد وهي فاكرة إنها
# لسه بتقرب (المسافة بالبكسل بتفضل شبه ثابتة في المنظور).
NO_PROGRESS_CYCLES = 100
NO_PROGRESS_FORWARDS = 50


def gray_from_jpeg(data: bytes):
    """صورة JPEG (من الموبايل) → إطار رمادي. بترجّع None لو الصورة باظت."""
    if not data:
        return None
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    except Exception:  # noqa: BLE001
        return None
    if img is None or img.size == 0:
        return None
    return img


@dataclass
class Patch:
    """بقعة في الصورة."""

    u: float
    v: float
    w: float
    h: float
    score: float
    ok: bool = True

    @property
    def size(self) -> float:
        return max(self.w, self.h)


@dataclass
class Cmd:
    """الأمر اللي هنبعته للعربية."""

    direction: str          # F / L / R / S
    speed: int
    seconds: float
    why: str
    done: bool = False
    reason: str = ""
    # ⚠️ نبضة «استطلاع» (بنمشي شوية عشان نعرف البوز) — **مش** مشي
    # ناحية الهدف. بتتعلّم عشان ما تتحسبش «تقدّم» (شوف بند ٣‑ج).
    sensing: bool = False


class Lost(Exception):
    """مش شايف حاجة — بنقف فورًا."""


class Tracker:
    """
    تتبّع بقعة بطريقتين مع بعض:

      ١) **نقاط مميزة + تدفّق بصري (KLT) + تحويل أفاين** — دي الأساس.
         بتصمد مع دوران العربية وتغيّر المنظور وتغيّر الحجم، لأنها
         بتتبّع تفاصيل جوه البقعة مش شكلها كله.
      ٢) **مطابقة عينة (template) بمقاسات** — بترجع تشتغل لو النقط
         ضاعت، وبتبلّغ إن البقعة اتلغبطت لو الاتنين فشلوا.

    الأولانية هي اللي بتخلي التتبّع يعيش، والتانية هي شبكة الأمان.
    """

    def __init__(self, gray, u: float, v: float, half: int = PATCH_HALF) -> None:
        h, w = gray.shape[:2]
        self.u = float(max(half + 1, min(w - half - 2, u)))
        self.v = float(max(half + 1, min(h - half - 2, v)))
        self.half = half
        self.size = float(2 * half)
        self.lost_frames = 0
        self.rotation = 0.0            # دوران البقعة التراكمي (للتشخيص)
        self.prev = gray
        self.pts = None
        self.templates: list = []
        self._seed(gray)

    # ---------- التقاط النقاط ----------

    def _seed(self, gray) -> None:
        half = self.half
        x0 = int(round(self.u - half))
        y0 = int(round(self.v - half))
        H, W = gray.shape[:2]
        x0 = max(0, min(W - 2 * half - 1, x0))
        y0 = max(0, min(H - 2 * half - 1, y0))
        roi = gray[y0:y0 + 2 * half, x0:x0 + 2 * half]
        if roi.size == 0:
            self.pts = None
            return
        self.templates = [roi.copy()] + self.templates[:1]
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=40, qualityLevel=0.04, minDistance=4)
        if pts is None or len(pts) < 4:
            self.pts = None
            return
        self.pts = (pts + np.array([[x0, y0]], dtype=np.float32)).astype(np.float32)
        self._box0 = (x0, y0)

    # ---------- البحث ----------

    def find(self, gray, search: int = SEARCH) -> Patch:
        """
        يلاقي البقعة في الإطار.

        `search` = نصف منطقة البحث للقالب (بكسل). الافتراضي `SEARCH`،
        وبنكبّرها في «إعادة الالتقاط» بعد لفة (شوف `tap_nav`): اللف
        بتحرّك الهدف بسرعة في الصورة، فالبحث الواسع بيرجّعه بدل ما
        نعلن إنه ضاع والعربية تقف.
        """
        p = self._by_flow(gray)
        if p is not None:
            return p
        p = self._by_template(gray, search=search)
        if p is not None:
            return p
        self.lost_frames += 1
        return Patch(self.u, self.v, self.size, self.size, 0.0, ok=False)

    def _by_flow(self, gray) -> Optional[Patch]:
        if self.pts is None or len(self.pts) < 4:
            return None
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev, gray, self.pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        if nxt is None or status is None:
            return None
        ok = status.ravel() == 1
        if int(ok.sum()) < 4:
            return None

        src = self.pts[ok].reshape(-1, 2)
        dst = nxt[ok].reshape(-1, 2)
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is None:
            return None

        # مقياس ودوران البقعة من التحويل
        scale = math.hypot(M[0, 0], M[1, 0])
        if scale <= 1e-3 or not math.isfinite(scale):
            return None
        self.rotation += math.degrees(math.atan2(M[1, 0], M[0, 0]))

        c = M @ np.array([self.u, self.v, 1.0], dtype=np.float64)
        if not (math.isfinite(c[0]) and math.isfinite(c[1])):
            return None
        self.u, self.v = float(c[0]), float(c[1])
        self.size = max(10.0, self.size * scale)
        self.half = int(max(10, min(60, self.size / 2)))
        self.pts = nxt[ok].reshape(-1, 1, 2)
        self.prev = gray
        self.lost_frames = 0

        # النقط قلّت؟ نلتقط نقط جديدة جوه المربع الحالي
        if int(ok.sum()) < 8:
            self._seed(gray)
        score = max(0.05, min(1.0, ok.sum() / 40.0 + 0.5))
        return Patch(self.u, self.v, self.size, self.size, score, ok=True)

    def _by_template(self, gray, search: int = SEARCH) -> Optional[Patch]:
        if not self.templates:
            return None
        h, w = gray.shape[:2]
        best = None
        for tpl in self.templates[:2]:
            if tpl.size == 0:
                continue
            for s in SCALES:
                tw = max(8, int(tpl.shape[1] * s))
                th = max(8, int(tpl.shape[0] * s))
                if tw >= w or th >= h:
                    continue
                t = tpl if (tw == tpl.shape[1] and th == tpl.shape[0]) else \
                    cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA)
                cx, cy = int(self.u), int(self.v)
                x0 = max(0, cx - search - tw // 2)
                y0 = max(0, cy - search - th // 2)
                x1 = min(w, cx + search + tw // 2)
                y1 = min(h, cy + search + th // 2)
                roi = gray[y0:y1, x0:x1]
                if roi.shape[0] < th or roi.shape[1] < tw:
                    continue
                res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
                if res.size == 0:
                    continue
                _, score, _, loc = cv2.minMaxLoc(res)
                if not math.isfinite(score):
                    continue                 # صورة سادة → النتيجة مش معنى
                if best is None or score > best[0]:
                    best = (score, x0 + loc[0] + tw / 2.0, y0 + loc[1] + th / 2.0, tw, th)

        if best is None or best[0] < MIN_SCORE:
            return None

        score, u, v, tw, th = best
        self.u, self.v = u, v
        self.size = float(max(tw, th))
        self.half = int(max(10, min(60, self.size / 2)))
        self.prev = gray
        self.lost_frames = 0
        self._seed(gray)
        return Patch(u, v, float(tw), float(th), score, ok=True)


@dataclass
class ServoState:
    """حالة السيرفو — بنحتاجها للعرض على الشاشة وللتشخيص."""

    phase: str = "idle"          # idle / sensing / driving / arrived / lost / waiting / done
    reason: str = ""
    cycles: int = 0
    elapsed: float = 0.0
    car: Optional[Patch] = None
    target: Optional[Patch] = None
    heading: Optional[tuple[float, float]] = None
    camera_moving: bool = False
    distance_px: float = 0.0
    # ⚠️ مسافة أرضية حقيقية (لو معانا موديل الكاميرا) — دي المرجع للقرار
    distance_m: Optional[float] = None
    log: list[str] = field(default_factory=list)


class VisualServo:
    """
    السيرفو كامل: عينتين + قرار + حالة.

    الاستخدام: تدّيه إطار رمادي كل دورة، وهو يرجّع `Cmd`
    (اتجاه + سرعة + مدة). الشغل كله هنا — مفيش MQTT ولا أزرار.
    """

    def __init__(self, gray, car_uv, target_uv, *, work_w: int = WORK_W,
                 calib=None) -> None:
        self.work_w = work_w
        scale = work_w / float(gray.shape[1])
        self.scale = scale
        small = cv2.resize(gray, (work_w, max(1, int(gray.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)

        # ⚠️ **مسافة حقيقية بالمتر** (لو معانا موديل الكاميرا).
        # ليه ده مهم؟ لأن المسافة **بالبكسل** مش مقياس مسافة: لما العربية
        # تبعد، صورة الهدف بتقرب من الأفق والفرق بالبكسل بيصغّر — يعني
        # البكسل بيقول «بتقرب» وهي بتشرد. ده كان بيخلي السيرفو يمشي
        # ١١ متر في الاتجاه الغلط (شوف test_visual_servo السيناريو ٣).
        self.calib = self._calib_for(small, calib)

        self.car = Tracker(small, car_uv[0] * scale, car_uv[1] * scale)
        self.target = Tracker(small, target_uv[0] * scale, target_uv[1] * scale)
        self.state = ServoState()
        self.state.car = self.last_car = Patch(self.car.u, self.car.v, 0, 0, 1.0)
        self.state.target = self.last_target = Patch(self.target.u, self.target.v, 0, 0, 1.0)

        self.heading: Optional[tuple[float, float]] = None
        self.senses = 0
        self.turns_since_forward = 0
        self.last_progress_dist: Optional[float] = None
        self.last_turn_side: Optional[str] = None
        self.turn_reversals = 0         # قلبات اتجاه اللف من غير تقدّم
        self.steps: list[float] = []
        self.best_dist: Optional[float] = None
        # ⚠️ «أحسن نقطة» بالمقياس الحقيقي لو معانا موديل كاميرا (وإلا بالبكسل)
        self.best_metric: Optional[float] = None
        self.diverge_cycles = 0
        self.arrive_cycles = 0
        # ⚠️ حاجز «مفيش تقدّم» (شوف _tick بند ٣‑ج)
        self.no_progress_cycles = 0
        self.forwards_since_progress = 0
        self.dist_hist: list[float] = []
        self.last_cmd_forward = False
        self.start = None

    @staticmethod
    def _calib_for(small, calib):
        """
        بيظبّط موديل الكاميرا على **مقاس الشغل** (الصورة بتتصغّر جوّه).

        ⚠️ التصغير مابيغيّرش زاوية الرؤية ولا الميل — بس العرض والطول
        اللي بيتحسب بيهم مركز الصورة. فبنبني `Calib` جديد بنفس الأرقام
        الفيزيائية ومقاسات الصورة المصغّرة.
        """
        if calib is None:
            return None
        try:
            import point_nav as pn
        except Exception:            # noqa: BLE001
            return None
        h = max(1, int(small.shape[0]))
        w = max(1, int(small.shape[1]))
        try:
            return pn.Calib(height_m=calib.height_m, pitch_deg=calib.pitch_deg,
                            hfov_deg=calib.hfov_deg, width=w, height=h)
        except Exception:            # noqa: BLE001
            return None

    def distance_to_target_m(self):
        """
        المسافة **الحقيقية** بين العربية والهدف على الأرض (بالمتر) — أو
        `None` لو مفيش موديل كاميرا (`calib`).

        ⚠️ ليه مش `pixel_to_ground(الهدف).distance_m`؟ لأن دي المسافة من
        **الكاميرا** للهدف. لو الكاميرا على العربية (الموبايل مربوط عليها)
        بتساوي المطلوب، لكن لو الموبايل في إيد حد واقف (والعربية هي اللي
        بتتحرك في الكادر) تبقى **ثابتة** ومالهاش معنى — والاختبار أثبت ده:
        المسافة من الكاميرا فضلت ٣.٣م والعربية بتقرب من ١.٤م لـ ٠.١م.
        فبنحوّل **الاتنين** (العربية والهدف) لنقط أرض ونقيس المسافة بينهم.
        """
        if self.calib is None:
            return None
        car = self.last_car or self.state.car
        tgt = self.last_target or self.state.target
        if car is None or tgt is None:
            return None
        try:
            import point_nav as pn
            cg = pn.pixel_to_ground(car.u, car.v, self.calib)
            tg = pn.pixel_to_ground(tgt.u, tgt.v, self.calib)
        except Exception:            # noqa: BLE001
            return None
        if cg is None or tg is None:
            return None
        return float(math.hypot(cg.x_m - tg.x_m, cg.y_m - tg.y_m))

    @property
    def step_px(self) -> float:
        """وسيط آخر الخطوات — أثبت بكتير من قراءة واحدة."""
        if not self.steps:
            return ARRIVE_FALLBACK_PX
        s = sorted(self.steps)
        return s[len(s) // 2]

    # ---------- الأدوات ----------

    def _to_full(self, p: Patch) -> tuple[float, float]:
        return p.u / self.scale, p.v / self.scale

    def public_state(self) -> dict:
        """الحالة اللي بتتبعت للصفحة (بكسل الإطار الكامل)."""
        car = self.state.car
        tgt = self.state.target
        cu, cv_ = self._to_full(car) if car else (None, None)
        tu, tv = self._to_full(tgt) if tgt else (None, None)
        return {
            "phase": self.state.phase,
            "reason": self.state.reason,
            "cycles": self.state.cycles,
            "elapsed": round(self.state.elapsed, 1),
            "distance_px": round(self.state.distance_px, 1),
            # ⚠️ المسافة الحقيقية بالمتر (None = مفيش موديل كاميرا)
            "distance_m": (round(self.state.distance_m, 2)
                           if self.state.distance_m is not None else None),
            "camera_moving": self.state.camera_moving,
            "car": None if car is None else {"u": round(cu, 1), "v": round(cv_, 1),
                                             "w": round(car.w / self.scale, 1),
                                             "h": round(car.h / self.scale, 1),
                                             "score": round(car.score, 2), "ok": car.ok},
            "target": None if tgt is None else {"u": round(tu, 1), "v": round(tv, 1),
                                                "w": round(tgt.w / self.scale, 1),
                                                "h": round(tgt.h / self.scale, 1),
                                                "score": round(tgt.score, 2), "ok": tgt.ok},
            "heading": self.heading,
            "log": self.state.log[-8:],
        }

    def _say(self, msg: str) -> None:
        self.state.log.append(msg)
        self.state.reason = msg

    # ---------- الدورة ----------

    def tick(self, gray, dt: float, dry: bool = False) -> Cmd:
        """
        دورة واحدة: شوف → قرّر → رجّع أمر.

        `dt` = الوقت اللي فات من الدورة اللي قبلها (تقريبي).
        """
        cmd = self._tick(gray, dt)
        # بنسجّل نوع آخر أمر عشان الدورة الجاية تعرف تفسّر الحركة:
        # الحركة اللي جاية من نبضة «لقدام» هي اللي بتقيس الخطوة.
        self.last_cmd_forward = (cmd.direction == "F")
        # ⚠️ بنعدّ نبضات القدّم من غير تقدّم حقيقي (شوف بند ٣‑ج في _tick).
        # العدّ ده في المكان الواحد اللي كل الأوامر بتعدي منه.
        if self.last_cmd_forward and not cmd.done and not cmd.sensing:
            self.forwards_since_progress += 1
        return cmd

    def _tick(self, gray, dt: float) -> Cmd:
        self.state.cycles += 1
        self.state.elapsed += dt

        if self.state.elapsed > MAX_SECONDS or self.state.cycles > MAX_CYCLES:
            self.state.phase = "done"
            self._say("خلص الوقت المسموح — وقف")
            return Cmd("S", 0, 0.05, "حد أقصى", done=True, reason="timeout")

        small = cv2.resize(gray, (self.work_w, max(1, int(gray.shape[0] * self.scale))),
                           interpolation=cv2.INTER_AREA)

        car = self.car.find(small)
        tgt = self.target.find(small)
        prev_car, prev_tgt = self.last_car, self.last_target
        self.last_car, self.last_target = car, tgt
        self.state.car, self.state.target = car, tgt

        # ---------- ١) شايفين الاتنين؟ ----------
        if not car.ok or not tgt.ok:
            self.state.phase = "lost"
            who = "العربية" if not car.ok else "الحتة"
            self._say(f"مش شايف {who} — وقفت")
            return Cmd("S", 0, 0.05, "فقدت التتبع", done=True, reason="lost")

        # ---------- ٢) الموبايل بيتحرك؟ ----------
        # بقعة الحتة ثابتة على الأرض، فأي حركة فيها = حركة الموبايل.
        cam_u = tgt.u - prev_tgt.u
        cam_v = tgt.v - prev_tgt.v
        cam_move = math.hypot(cam_u, cam_v)
        self.state.camera_moving = cam_move > CAMERA_MOVE_LIMIT
        if self.state.camera_moving:
            self.state.phase = "waiting"
            self._say(f"الموبايل بيتحرك ({cam_move:.0f}px) — مستني يثبت")
            return Cmd("S", 0, 0.12, "الموبايل بيتحرك")

        # ---------- ٣) وصلنا؟ ----------
        dx = tgt.u - car.u
        dy = tgt.v - car.v
        dist = math.hypot(dx, dy)
        self.state.distance_px = dist

        # ⚠️ **مسافة أرضية حقيقية** لو معانا موديل الكاميرا (`calib`).
        # البكسل مش مقياس مسافة: لما العربية تبعد، الهدف بيقرب من الأفق
        # والفرق بالبكسل بيصغّر — يعني البكسل يقول «بتقرب» وهي بتشرد
        # (اتشاف: مشيت ١١ متر وهي فاكرة إنها على ٨ بكسل من الهدف).
        dist_m = self.distance_to_target_m()
        if dist_m is not None:
            self.dist_hist.append(dist_m)
            if len(self.dist_hist) > DIST_MEDIAN_N:
                self.dist_hist.pop(0)
            dist_m = sorted(self.dist_hist)[len(self.dist_hist) // 2]
        self.state.distance_m = dist_m
        self.state.distance_px = dist

        # عتبة الوصول = نسبة من **خطوة العربية** المقيسة في نفس اللحظة،
        # مقيّدة من فوق ومن تحت، ولازم تتحقق كام دورة ورا بعض.
        #
        # ⚠️ **ولو معانا مسافة حقيقية (متر) فهي المرجع** — لأن البكسل
        # بيكدب وهو بعيد: الهدف البعيد بيبان برضه على ٨-١١ بكسل من
        # العربية (منظور)، وده اللي خلّى السيرفو يمشي ١١ متر وهو فاكر
        # إنه وصل. (قرب الهدف بالعكس: المتر بيتحيّز شوية بسبب اهتزاز
        # الموبايل — وساعتها بنوقف «قريب» وبنقولها بصدق، شوف تحت.)
        step = min(STEP_MAX_PX, max(STEP_MIN_PX, self.step_px))
        arrive_at = min(ARRIVE_MAX_PX, max(ARRIVE_MIN_PX, ARRIVE_STEP_RATIO * step))
        close = (dist_m <= ARRIVE_M) if dist_m is not None else (dist <= arrive_at)
        if close:
            self.arrive_cycles += 1
            if self.arrive_cycles >= ARRIVE_CONFIRM:
                self.state.phase = "arrived"
                where = (f"باقي {dist_m:.2f}م" if dist_m is not None
                         else f"باقي {dist:.0f}px · الخطوة {step:.0f}px")
                self._say(f"وصلت ({where})")
                return Cmd("S", 0, 0.05, "وصلت", done=True, reason="arrived")
        else:
            self.arrive_cycles = 0

        # ---------- ٣‑ب) أمان: شاردة؟ ----------
        # تقدّم حقيقي؟ (قربنا عن آخر نقطة سجّلناها)
        # ⚠️ المقارنة بـ `last_progress_dist` مش بـ `best_dist` — لأن
        # `best_dist` بيتحدّث كل دورة، فتقدّم صغير متكرر ماكانش بيتحسب.
        # وبنقيس بالمتر لو معانا (المرجع الصح)، وإلا بالبكسل.
        metric = dist_m if dist_m is not None else dist
        metric_min = PROGRESS_MIN_M if dist_m is not None else PROGRESS_MIN_PX
        if (self.last_progress_dist is None
                or metric < self.last_progress_dist - metric_min):
            self.last_progress_dist = metric
            self.turn_reversals = 0      # تقدمنا → مش «لفة حوالين نفسها»
            self.no_progress_cycles = 0
            self.forwards_since_progress = 0
        else:
            self.no_progress_cycles += 1

        # ---------- ٣‑ج) أمان: بنمشي من غير أي تقدّم؟ ----------
        # ⚠️ باچ حقيقي (اتشاف بالاختبار): لما العربية خلاص عند الهدف أو
        # عدّته، المسافة **بالبكسل** بتفضل شبه ثابتة (٨-١١px) وهي بتعَدّي
        # وتشرد **٧ متر** — لأن البكسل مش مقياس مسافة حقيقي (منظور).
        # وحاجز «الشرود» مامسكش الحالة دي: `dist` بيتذبذب حوالين العتبة
        # فعدّاد الشرود بيتصفّر كل شوية.
        #
        # فبنحسب حاجة مالهاش علاقة بالعتبة: **كام نبضة قدّم من غير أي
        # تقدّم حقيقي**. لو مشينا كتير من غير ما نقرب، بنقف.
        if not dry and (self.no_progress_cycles >= NO_PROGRESS_CYCLES
                and self.forwards_since_progress >= NO_PROGRESS_FORWARDS):
            # ⚠️ لو وقفنا وإحنا **أصلاً عند الهدف** (جوّه عتبة الوصول)،
            # فده وصول مش فشل: السيرفو مش قادر يقرب أكتر (اهتزاز الصورة
            # أو دقة التتبّع) وهو واقف في المكان الصح. بنقول «وصلت» بصدق.
            # ⚠️ بنقارن بـ **أحسن** مسافة وصلها مش اللحظة دي، لأن
            # اهتزاز الكاميرا بيخلي القراءة الحالية تنط فوق العتبة والقراءة
            # اللي قبلها كانت جوّه. (لو وصلنا العتبة في أي لحظة، يبقى وصلنا.)
            reached = close
            if (not reached and dist_m is not None
                    and self.best_metric is not None
                    and self.best_metric <= ARRIVE_M):
                reached = True
            if reached:
                self.state.phase = "arrived"
                self._say("مش قادر أقرب أكتر — وأنا عند الهدف خلاص")
                return Cmd("S", 0, 0.05, "وصلت", done=True, reason="arrived")
            self.state.phase = "done"
            shown = (f"{dist_m:.2f}م" if dist_m is not None else f"{dist:.0f}px")
            self._say(f"مشيت {self.forwards_since_progress} نبضة من غير ما أقرب "
                      f"({shown} ثابتة) — وقفت")
            return Cmd("S", 0, 0.05, "مفيش تقدّم", done=True,
                       reason="no_progress")

        # الشرود: بنقارن بالمقياس الحقيقي لو معانا، وإلا بالبكسل
        if self.best_metric is None or metric < self.best_metric:
            self.best_metric = metric
            self.best_dist = dist
            self.diverge_cycles = 0
        elif not dry and metric > self.best_metric * DIVERGE_FACTOR:
            self.diverge_cycles += 1
            if self.diverge_cycles >= DIVERGE_CONFIRM:
                self.state.phase = "done"
                self._say(f"شاردت (أحسن نقطة {self.best_metric:.2f} · "
                          f"دلوقتي {metric:.2f}) — وقفت")
                return Cmd("S", 0, 0.05, "شاردت", done=True, reason="diverged")
        else:
            self.diverge_cycles = 0

        # ---------- ٤) اتجاه العربية من حركتها الحقيقية ----------
        # حركة العربية في الكادر ناقص حركة الموبايل = الحركة الحقيقية
        mu = (car.u - prev_car.u) - cam_u
        mv = (car.v - prev_car.v) - cam_v
        moved = math.hypot(mu, mv)
        if MIN_MOTION_PX <= moved <= MAX_MOTION_PX:
            # «خطوة العربية» = إزاحتها الحقيقية في آخر نبضة لقدام.
            # بناخد وسيط كام قراءة عشان قراءة واحدة غلط ما تلخبطش العتبة.
            if self.last_cmd_forward:
                self.steps.append(moved)
                if len(self.steps) > STEP_MEDIAN_N:
                    self.steps.pop(0)
            n = moved
            mx, my = mu / n, mv / n
            if self.heading is None:
                self.heading = (mx, my)
            else:
                hx, hy = self.heading
                # تمليس: الاتجاه الجديد بوزن أكبر لو الحركة واضحة
                wx = 0.45
                hx2, hy2 = hx * (1 - wx) + mx * wx, hy * (1 - wx) + my * wx
                nn = math.hypot(hx2, hy2) or 1.0
                self.heading = (hx2 / nn, hy2 / nn)
            self.state.heading = self.heading

        # ---------- ٥) القرار ----------
        if self.heading is None:
            # لسه مش عارفين بوزها → حركة استطلاع قصيرة
            self.state.phase = "sensing"
            self._say("بتحرك شوية أشوف بوزها")
            return Cmd("F", SPEED_FORWARD, PULSE_SENSE, "استطلاع", sensing=True)

        hx, hy = self.heading
        # الزاوية بين اتجاه حركتها واتجاه الحتة (إحداثيات شاشة: y لتحت)
        cross = hy * dx - hx * dy
        dot = hx * dx + hy * dy
        angle = math.degrees(math.atan2(cross, dot))   # موجب = شمال

        if abs(angle) > TURN_ANGLE_DEG:
            # ⚠️ اللف في المكان مابيحرّكش مركز العربية، يعني مابنشوفش
            # نتيجته في الصورة. عشان كده بعد لفّتين بالكتير بنعمل
            # **نبضة استطلاع لقدام** نتأكد بيها من البوز الجديد.
            if self.turns_since_forward >= 2:
                self.turns_since_forward = 0
                self.state.phase = "sensing"
                self._say("بشوف بوزها تاني")
                return Cmd("F", SPEED_FORWARD, PULSE_SENSE, "استطلاع بعد اللف", sensing=True)
            side = "L" if angle > 0 else "R"
            # ---- حاجز اللف حوالين نفسها: قلب الاتجاه من غير تقدّم ----
            if self.last_turn_side is not None and side != self.last_turn_side:
                self.turn_reversals += 1
                if self.turn_reversals > TURN_REVERSALS_MAX:
                    self.state.phase = "done"
                    self._say("بتلف حوالين نفسها (بتقلب يمين/شمال) — وقفت")
                    return Cmd("S", 0, 0.05, "لفّة حوالين نفسها", done=True,
                               reason="spin")
            self.last_turn_side = side
            self.turns_since_forward += 1
            self.state.phase = "turning"
            self._say(f"بلف {'شمال' if side == 'L' else 'يمين'} {abs(angle):.0f}°")
            return Cmd(side, SPEED_TURN, PULSE_TURN, f"فرق {angle:.0f}°")

        self.turns_since_forward = 0
        self.state.phase = "driving"
        if dist <= FINE_ZONE * arrive_at:
            # اقتراب نهائي بنبضات أنعم — عشان ما تعديش الهدف
            self._say(f"بقرّب — باقي {dist:.0f}px")
            return Cmd("F", SPEED_FINE, PULSE_FINE, f"قدّم ناعم {dist:.0f}px")
        self._say(f"ماشية — باقي {dist:.0f}px")
        return Cmd("F", SPEED_FORWARD, PULSE_FORWARD, f"قدّم {dist:.0f}px")
