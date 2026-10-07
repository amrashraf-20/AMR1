"""
vision.py  —  الرؤية بالحاسوب: تحسين الصورة + كشف الأجسام
========================================================================

ثلاث قطع صغيرة، كل واحدة شغلتها واحدة:

    FrameEnhancer  — تحسين صورة (CLAHE). دوال صافية.
    Detector       — YOLOv8n على CPU. تحميل كسول + استدلال في ثريد.
    VisionPipeline —  4) the pipeline thread that ties it together。

⚠️⚠️ **«latest-frame-only» — السرعة أهم من الكمال:**

كل قطعة بتخزّن **آخر نتيجة** في خانة واحدة، والقارئ بيمسحها ويكتب
فوقها. ** طابور.** ده معناه: لو استغرق استدلال YOLO ١٠٠ مللي وال
فريم الجديد وصل في الـ ٦٠ مللي، الفريم القديم ** بيتّرك** — مش بيتخزّن
لحد ما تلحق. النتيجة: **صفر تأخير (zero lag)** مهما ثقل الحاسوب.

ده سبب إن `/video` (البثّ القديم) **مش** بيعدّي من هنا خالص — بياخد من
`camera` مباشرة. لو YOLO علِق، البثّ مايتأثرش أبدًا.

⚠️ **الاعتماديات اختيارية كل واحدة لوحدها:**
    • `cv2`    → من غيره  تحسين ولا رسم
    • `ultralytics` → من غيره  كشف (الوصف لسه شغال)
الاستيراد متأخر (lazy) جوّه الدوال، عشان السيرفر **يقوم** حتى لو الحزم
مش موجودة — نفس مبدأ `db.py` مع psycopg2.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent

# ⚠️ الأوزان مخزّنة **جوه المشروع** (مش في كاش Ultralytics في الـ home).
# السبب: كاش الـ home بيتمسح، ووقتها السيرفر هيعتمد على شبكة وقت التشغيل
# ويفشل لو الجهاز أوفلاين. مجلد `weights/` معانا في الريبو.
WEIGHTS_DIR = HERE / "weights"
WEIGHTS_FILE = WEIGHTS_DIR / "yolov8n.pt"

# نلاقي الأوزان في ٣ أماكن عشان متهتمّش بمكان التثبيت exactly
WEIGHTS_CANDIDATES = (
    WEIGHTS_FILE,
    HERE / "yolov8n.pt",
    Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "yolov8n.pt",
)

# ================= إعدادات الكشف =================
# ⚠️ `skip=2` (كان 5 — 2026-09-29). الاستدلال ≈ ٤٣ms على CPU، فـ:
#     skip=5 → استدلال كل ٥ فريم (~٤Hz عند ٢٠fps) — المربعات بترتعش
#     skip=2 → ~١٠Hz — ناعم، و≈٢٦% معالج
#     skip=1 → ~٢٠Hz — ≈٥٢% معالج، وبتنافس مع البثّ نفسه
# فـ ٢ هو التوازن.型和 ومع الاحتمال (شوف الحلقة) الـ skipمش بيسبب
# رعشة تاني.
DEFAULT_SKIP = 2
# ⚠️ `conf=0.25` (كان 0.35). الحاجات الحقيقية (كرسي، زجاجة، شخص بعيد)
# طالعة بح confidences متوسطة، وعتبة 0.35 كانت بتخليها تختفي وتظهر.
# العتبة الأقل بتدي مربعات **أكثر ثباتًا** لأن الحاجات الحقيقية متوسط
# ثقتها أعلى بكتير من 0.25.
DEFAULT_CONF = 0.25          # أقل من 0.5 عشان نلقط أهداف بعيدة/مظلمة

# ⚠️ أقصى عمر للمربع قبل ما ينمسح (ثواني). الاحتمال مالوش = خطر:
# لو الاستدلال وقف، المربعات هتفضل عالقة على مشهد مختلف تمامًا.
BOX_MAX_AGE_S = 2.0

# استدلال torch takes every core by default - so it would fight the rest of the server.
# بwe cap it: we left أنوية لـ MQTT والفريم نفسه.
TORCH_THREADS = 2

# أقل عتبة ثقة نعتبر بيها «الحاجز أمامنا» (قسم المسار Middle)
NEAR_CONF = 0.45


def find_weights() -> Optional[Path]:
    """أول ملف أوزان موجود فعلًا (أو None)."""
    for p in WEIGHTS_CANDIDATES:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# =====================================================================
#  ١) تحسين الصورة — دوال صافية، مالها غير numpy/cv2
# =====================================================================

def apply_clahe(bgr, *, clip_limit: float = 2.0, grid: int = 8):
    """
    CLAHE على قناة الـ L في فضاء LAB.

    ليه LAB مش RGB: CLAHE بيزوّد الـ contrast محليًا، وفي RGB بيعمل
    تشبّع ودرجات غريبة. في LAB إحنا بنعدّل **الإضاءة بس** (L) واللون
    (a/b) بيفضل زي ما هو.

    دالة صافية: بترجّع صورة جديدة، ومدخلها **مش بيتغير**.
    """
    import cv2
    import numpy as np

    if bgr is None or getattr(bgr, "size", 0) == 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(grid, grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # ⚠️ نسخ صريح: `cvtColor` بيرجّع مصفوفة جديدة بس  داعي نخلي
    # القارئ يفسّر إن入力 اتغيّر.
    return np.ascontiguousarray(out)


def unsharp(bgr, *, amount: float = 0.6, radius: int = 3):
    """
    Unsharp masking: `صورة + amount × (صورة − ضبابها)`.

    ⚠️ ده **اختياري** ومش مفعّل افتراضيًا: CLAHE لوحده بيغطي أغلب
    الحاجة، و unsharp بيعamplify الضوضاء مع الحواف. we leftها متاحة
    من `/api/vision/config` لو حد حب يجرّب.
    """
    import cv2
    import numpy as np

    if bgr is None or getattr(bgr, "size", 0) == 0:
        return bgr
    blur = cv2.GaussianBlur(bgr, (radius * 2 + 1, radius * 2 + 1), 0)
    return cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)


def enhance_frame(bgr, *, enhance: bool = True, unsharp_mask: bool = False,
                  unsharp_amount: float = 0.6):
    """
    السلسلة الكاملة بالترتيب الصح:

        ١) **CLAHE** — يوحّد الإضاءة ويوضّح.
        ٢) **Unsharp** — آخر حاجة.

    ⚠️ **GaussianBlur اتشال بالكامل** (2026-09-29، طلب المشغّل). سببين:
    الضباب بيبلّع حوافّ الحاجات، وYOLOv8 بياخد قراراته من الحوافّ —
    فالضباب **بيقلّل دقة الكشف**، مش بس بيبعّد. وكمان وسّع الداشبورد
    كان بيبان حلو في الفيديو وبيلخبط في الأرقام. فاحنا عايزين صورة
    **أوضح**، مش أنعم.
    """
    out = bgr
    if enhance:
        out = apply_clahe(out)
    if unsharp_mask:
        out = unsharp(out, amount=unsharp_amount)
    return out


# =====================================================================
#  ٢) الكشف — YOLOv8n
# =====================================================================

@dataclass
class Detection:
    """حاجة واحدة اتشحصت. الإحداثيات نسب مئوية (0..100) مش بكسل."""
    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float          # مركز الصندوق أفقيًا (نسبة) — ده موقعه قدام العربية
    cy: float          # رأسه رأسيًا (نسبة) — كل ما زاد = أقرب

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "conf": round(self.conf, 3),
            "box": [round(self.x1, 1), round(self.y1, 1),
                    round(self.x2, 1), round(self.y2, 1)],
            "cx": round(self.cx, 1),
            "cy": round(self.cy, 1),
        }


def detections_from_result(result) -> list[Detection]:
    """
    YOLO Result → قائمة `Detection` نسبية.

    **نسب مش بكسل** عشان الـ endpoint يبقى مستقل عن حجم الصورة: نفس
    الإحداثيات هتطلع ٦٤٠×٤٨٠ أو ١٠٨٠×١٠٨٠.
    """
    out: list[Detection] = []
    try:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out
        names = getattr(getattr(result, "names", None), "__getitem__", None)
        for b in boxes:
            conf = float(b.conf[0])
            xy = b.xyxy[0].tolist()
            h_img = float(getattr(result, "orig_shape")[0]) or 1.0
            w_img = float(getattr(result, "orig_shape")[1]) or 1.0
            x1, y1, x2, y2 = xy
            cls = int(b.cls[0])
            label = result.names.get(cls, str(cls)) if names else str(cls)
            out.append(Detection(
                label=label, conf=conf,
                x1=100.0 * x1 / w_img, y1=100.0 * y1 / h_img,
                x2=100.0 * x2 / w_img, y2=100.0 * y2 / h_img,
                cx=100.0 * ((x1 + x2) / 2.0) / w_img,
                cy=100.0 * ((y1 + y2) / 2.0) / h_img,
            ))
    except Exception:                      # noqa: BLE001
        # ⚠️ شكل الـ result بت Ultralytics بيتغيّر بين الإصدارات. أحسن
        # نرجع فاضي (الكشف بيرجع يشتغل) من إن السلسلة Located تقع.
        return out
    return out


class Detector:
    """
    YOLOv8n على CPU، **بتحميل كسول**: الموديل بيتحمّل أول استدلال،
    مش وقت الإقلاع.

    ليه كده؟ تحميل ultralytics + torch بياخد **٧ ثواني**. لو ده حصل
    وقت الإقلاع، السيرفر كله هيتأخر ٧ ثواني — من غير أي داعي، لأن
    `/api/tap/start` (القيادة) محتاج يشتغل فورًا.
    """

    def __init__(self, weights: Optional[Path] = None,
                 conf: float = DEFAULT_CONF) -> None:
        self.weights = weights or find_weights()
        self.conf = conf
        self._model: Any = None
        self._lock = threading.Lock()
        self.available = False
        self.error: Optional[str] = None
        self.infer_ms: Optional[float] = None
        self.loads_ms: Optional[float] = None

    def load(self) -> bool:
        """حمّل الموديل لو مش محمّل. بيرجّع True لو جاهز."""
        if self._model is not None:
            return True
        with self._lock:
            if self._model is not None:       # حد تانيحمّله في الأثناء
                return True
            if self.weights is None:
                self.error = ("yolov8n.pt مش موجود — حطه في "
                              f"{WEIGHTS_DIR}")
                return False
            t0 = time.time()
            try:
                import torch
                from ultralytics import YOLO
                try:
                    torch.set_num_threads(TORCH_THREADS)
                except Exception:             # noqa: BLE001
                    pass
                self._model = YOLO(str(self.weights))
                self.loads_ms = (time.time() - t0) * 1000
                self.available = True
                self.error = None
                print(f"[vision] YOLOv8n اتحمّل في {self.loads_ms/1000:.1f}s "
                      f"({self.weights.name})", flush=True)
                return True
            except Exception as exc:          # noqa: BLE001
                self.error = str(exc)
                self.available = False
                print(f"[vision] ✘ حمل YOLO فشل: {exc}", flush=True)
                return False

    def detect(self, bgr) -> list[Detection]:
        """استدلال على صورة واحدة. بيرجّع [] لو الموديل مش جاهز."""
        if bgr is None or getattr(bgr, "size", 0) == 0:
            return []
        if not self.load():
            return []
        t0 = time.time()
        try:
            with self._lock:                  # 🔒 MUTEX
                result = self._model(bgr, verbose=False, conf=self.conf)[0]
        except Exception as exc:              # noqa: BLE001
            self.error = str(exc)
            return []
        finally:
            self.infer_ms = (time.time() - t0) * 1000
        return detections_from_result(result)


# ================== رسم المربعات ==================

# ألوان مميّزة لكل صنف — نختار بالـ hash عشان نفس الصنف يطلع بنفس اللون
# كل مرة (المستخدم بيتعلّم ألوانCLASS ويفرّق بسرعة).
def _color_for(label: str):
    import colorsys
    h = (hash(label) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))   # OpenCV = BGR


def draw_detections(bgr, dets: list[Detection], *,
                    show_conf: bool = False,
                    thickness: int = 2,
                    path_bands: bool = True):
    """
    يرسم المربعات + الاسم **على الصورة نفسها** — عشان الداشبورد يعرض
    فريم واحد فيه كل حاجة (صورة محسّنة + مربعات) بدل ما يرسمها في
    المتصفح على صورة تانية.

    ⚠️ بيرجّع صورة **جديدة**؛ لو `dets` فاضية بيرجّع المدخل زي ما هو
    (من غير نسخة — عشان مانعملش copy كل فريم في الحالة الشائعة دي).
    """
    import cv2

    if bgr is None or getattr(bgr, "size", 0) == 0 or not dets:
        return bgr
    img = bgr.copy()
    h, w = img.shape[:2]

    # ---- خط «عمود الطريق» (شريط وسط) ----
    # ⚠️ دي **معلومة للمشغّل** مش指挥 للعربية. الخط بينصفّ المسار بس.
    if path_bands:
        cv2.line(img,
                 (int(w * PATH_LEFT_PCT / 100), 0),
                 (int(w * PATH_LEFT_PCT / 100), h),
                 (60, 60, 60), 1, cv2.LINE_AA)
        cv2.line(img,
                 (int(w * PATH_RIGHT_PCT / 100), 0),
                 (int(w * PATH_RIGHT_PCT / 100), h),
                 (60, 60, 60), 1, cv2.LINE_AA)

    scale = max(0.4, h / 480.0)                 # الخط بيكبر مع الصورة
    fs = 0.45 * scale
    th = max(1, int(round(thickness * scale)))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for d in dets:
        x1 = int(d.x1 * w / 100.0)
        y1 = int(d.y1 * h / 100.0)
        x2 = int(d.x2 * w / 100.0)
        y2 = int(d.y2 * h / 100.0)
        col = _color_for(d.label)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, th, cv2.LINE_AA)

        # ⚠️ حاجز في عمود الطريق = خط **أسمك**، عشان يبقى واضح من بعيد
        is_in_path = PATH_LEFT_PCT <= d.cx <= PATH_RIGHT_PCT
        if is_in_path and d.label in OBSTACLE_LABELS and d.conf >= NEAR_CONF:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255),
                          th + 2, cv2.LINE_AA)

        text = d.label if not show_conf else f"{d.label} {d.conf:.0%}"
        (tw, tht), _ = cv2.getTextSize(text, font, fs, th)
        ty = max(tht + 4, y1)
        cv2.rectangle(img, (x1, ty - tht - 4), (x1 + tw + 6, ty + 2),
                      col, -1)
        cv2.putText(img, text, (x1 + 3, ty - 2), font, fs,
                    (0, 0, 0), max(1, th - 1), cv2.LINE_AA)
    return img


# =====================================================================
#  ٣) الوصف الهندسي — من المربعات لجملة مفيدة
# =====================================================================

# ⚠️ «عمود الطريق»: نص عرض الصورة (25%..75%) — أي حاجة برّه ده مش في
# طريقنا. واللي تحط جوه السطر = **حاجز**.
PATH_LEFT_PCT = 25.0
PATH_RIGHT_PCT = 75.0

# the ratios الرأسية اللي بتحكم «قريب/بعيد»:
# ⚠️ منطقه: في صورة لكاميرا **مثبّتة على الأرض وشايفة قدامها**، هدف بعيد
# بيبقى **قريب من الأفق (لفوق)**، وقريب بيبقى **في نص الصورة أو تحتها**.
# فـ `cy` الصغيرة = قريب. ده عكس الإحساس اللي عندنا من صور الشبكة.
NEAR_CY = 45.0            # cy تحت دي = حاجه في النطاق القريب
FAR_CY = 28.0             # cy تحت دي = بعيد

# намشي قدام ك حاجة (可为 obejcts) — تصنيف تقريبي
OBSTACLE_LABELS = frozenset({
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "chair",
    "couch", "potted plant", "dining table", "bed", "toilet", "tv",
    "bench", "suitcase", "bottle", "backpack", "handbag", "book",
    "cell phone", "traffic light", "fire hydrant", "stop sign",
})


def classify_path(dets: list[Detection]) -> dict:
    """
    يقسّم الكشف لـ «في طريقنا» و«برّه»، ويطلّع أقرب حاجز.

    ⚠️ **مالوش علاقة بالسلامة** — ده كلام للمشغّل بس. العربية
    مالهاش حساس (الـ HC-SR04 اتشال)، ف فرملة تلقائية. اقراه
    كـ«إيه قدامي» مش كـ«ممكن أستخدمه».
    """
    in_path, beside = [], []
    for d in dets:
        (in_path if PATH_LEFT_PCT <= d.cx <= PATH_RIGHT_PCT else beside).append(d)

    obstacles = [d for d in in_path
                 if d.label in OBSTACLE_LABELS and d.conf >= NEAR_CONF]
    clear = not obstacles

    nearest = min(obstacles, key=lambda d: d.cy, default=None)
    return {
        "clear": clear,
        "in_path": in_path,
        "beside": beside,
        "obstacles": obstacles,
        "nearest": nearest,
    }


def describe(dets: list[Detection]) -> dict:
    """
    يحوّل قائمة Detection لجملة عربية قصيرة + ملخص JSON.

    ⚠️ **دي مش «كابشن» بالمعنى الكامل.**  VLM هنا، ف فهم للمشهد
    («غرفة فيها ترابيزة عليها قهوة»). اللي بنقوله هو **حقائق هندسية**:
    إيهDetected، أنهي جزء من الصورة، وقربه. ده اللي ينفع للعربية، وكتير
    أدق من تخمين على «المشهد شكله إيه».
    """
    dets = list(dets)
    info = classify_path(dets)
    nearest: Optional[Detection] = info["nearest"]
    beside: list[Detection] = info["beside"]

    if not dets:
        text = "مفيش حاجة واضحة قدام العربية — الطريق باين خالص."
    elif info["clear"]:
        side = "، و" + "، ".join(sorted({d.label for d in beside})) \
            if beside else ""
        text = f"الطريق قدامك خالص. شايف {len(dets)} حاجة بعيدة{side}."
    else:
        # `clear=False` معناها إن `obstacles` مش فاضية، فـ `nearest` مضمون
        assert nearest is not None
        near = ("قريب جدًا" if nearest.cy < NEAR_CY
                else "على مسافة")
        text = (f"فيه {nearest.label} في طريقك {near}. "
                f"عدد العوائق في المسار: {len(info['obstacles'])}.")
        if beside:
            text += f" وبرّه الطريق: {', '.join(sorted({d.label for d in beside}))}."

    return {
        "description": text,
        "path_clear": bool(info["clear"]),
        "object_count": len(dets),
        "objects_in_path": [d.label for d in info["in_path"]],
        "objects_beside": [d.label for d in beside],
        "nearest_obstacle": nearest.as_dict() if nearest else None,
        "detections": [d.as_dict() for d in dets],
    }


# =====================================================================
#  ٤) الـ Pipeline — الثريد اللي بيجمع الكل
# =====================================================================

class VisionPipeline:
    """
    ثريد واحد بيعمل: خد آخر فريم → حسّنه → خزّنه → (كل `skip` فريم) اشتغل
    YOLO.

    ⚠️ **latest-frame-only**: `camera.push_remote` بيكتب فوق أي فريم قديم
    على طول. إحنا we do exactly the same thing — lock, read, unlock. **
    طابور في أي مكان**، ف تأخير متراكم مهما ثقل الحاسوب.
    """

    def __init__(self, camera, *, skip: int = DEFAULT_SKIP,
                 conf: float = DEFAULT_CONF,
                 enhance: bool = True, detect: bool = True,
                 annotate: bool = True, show_conf: bool = False,
                 unsharp_mask: bool = False,
                 jpeg_quality: int = 80) -> None:
        self.camera = camera
        self.skip = max(1, int(skip))
        self.conf = float(conf)
        self.enhance = bool(enhance)
        self.unsharp_mask = bool(unsharp_mask)
        # ⚠️ `detect` = نشغّل YOLO. `annotate` = نرسم المربعات على
        # الفريم. منفصلين عمدًا: ممكن تشغّل كشف بس من غير رسم (لو
        # بتقرا الأرقام من `/api/vision/detections` بس).
        self.detect = bool(detect)
        self.annotate = bool(annotate)
        self.show_conf = bool(show_conf)
        self.jpeg_quality = jpeg_quality

        self.detector = Detector(conf=self.conf)

        # آخر نتائج (كل واحد في خانته)
        self._frame: Optional[bytes] = None       # jpeg محسّن
        # ⚠️ عدّاد خاص لينا (مش `camera.seq()`): البثّ لازم يبعت كل
        # فريم **بنشر**، وننشر مرتين في اللفة الواحدة (من غير مربعات
        # + بالمربعات). لو استعملنا رقم الكاميرا كانت الزيادة مرة واحدة
        # والـ `<img>` هيعرض التانية بس أو مايفرقش.
        self._frame_seq: int = 0
        self._dets: list[Detection] = []
        self._dets_at: float = 0.0
        self._lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.enhanced_frames = 0
        self.detect_runs = 0
        self.last_error: Optional[str] = None

    # ---------- الإعدادات ----------
    def configure(self, **kw) -> None:
        """تبديل الإعدادات من غير ما نعيد تشغيل الثريد."""
        for key in ("enhance", "unsharp_mask", "detect", "annotate",
                    "show_conf"):
            if key in kw and kw[key] is not None:
                setattr(self, key, bool(kw[key]))
        if kw.get("skip"):
            self.skip = max(1, int(kw["skip"]))
        if kw.get("conf"):
            self.detector.conf = float(kw["conf"])

    def status(self) -> dict:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "enhance": self.enhance,
            "detect": self.detect,
            "annotate": self.annotate,
            "show_conf": self.show_conf,
            "unsharp_mask": self.unsharp_mask,
            "skip": self.skip,
            "conf": self.detector.conf,
            "model_loaded": self.detector._model is not None,
            "model_available": self.detector.available,
            "model_error": self.detector.error,
            "weights": str(self.detector.weights) if self.detector.weights else None,
            "infer_ms": (round(self.detector.infer_ms, 1)
                         if self.detector.infer_ms else None),
            "load_ms": (round(self.detector.loads_ms, 1)
                        if self.detector.loads_ms else None),
            "enhanced_frames": self.enhanced_frames,
            "detect_runs": self.detect_runs,
            "has_enhanced_frame": self._frame is not None,
            "detections": [d.as_dict() for d in self.detections()],
            "last_error": self.last_error,
        }

    # ---------- القراءة ----------
    def frame(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    def frame_seq(self) -> int:
        """
        رقم آخر فريم محسّن (مش رقم الكاميرا).

        ⚠️ مهم للبثّ MJPEG: لو الداشبورد قارن ده برقم الكاميرا، التزا
        봄gardiyاد بين الحسّن والكاميرا، وبيعمل تكرار. ده الرقم بتاع
        الفريم **بعد** المعالجة — اللي البثّ بيقدّمه فعلًا.
        """
        with self._lock:
            return self._frame_seq

    def detections(self) -> list[Detection]:
        with self._lock:
            return list(self._dets)

    # ---------- التشغيل ----------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="vision")
        self._thread.start()
        w = self.detector.weights
        print(f"[vision] بدأ (enhance={self.enhance} detect={self.detect} "
              f"annotate={self.annotate} skip={self.skip}) · الأوزان: "
              f"{w.name if w else 'مش موجودة!'}", flush=True)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._thread = None

    def _publish(self, img, cv2) -> None:
        """يرمّز صورة ويخزّنها كـ «آخر فريم» (يكتب فوق القديم دايمًا)."""
        ok, buf = cv2.imencode(
            ".jpg", img,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if ok:
            with self._lock:
                self._frame = buf.tobytes()
                self._frame_seq += 1
            self.enhanced_frames += 1

    # ---------- الحلقة ----------
    def _loop(self) -> None:
        import cv2
        import numpy as np

        counter = 0
        while not self._stop.is_set():
            raw = self.camera.jpeg()
            if not raw:
                # ⚠️  فريم (لسه الموبايل مبعتش). بننام شوية بدل ما
                # ندور في دايرة بتاكل معالج.
                time.sleep(0.1)
                continue
            try:
                arr = np.frombuffer(raw, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue

                out = enhance_frame(
                    bgr, enhance=self.enhance,
                    unsharp_mask=self.unsharp_mask)

                counter += 1
                want_det = self.detect and counter % self.skip == 0

                # ---- ١) الكشف: كل `skip` فريم ----
                if want_det:
                    dets = self.detector.detect(out)
                    with self._lock:
                        self._dets = dets
                        self._dets_at = time.time()
                    self.detect_runs += 1

                # ---- ٢) انشر **واحد** — دايمًا بالمربعات ----
                # ⚠️⚠️ **دي كانت باعث رعشة (اتمسك 2026-09-29).**
                # المنطق القديم كان بينشر الصورة **من غير مربعات** الأول،
                # وبعدين لو ده فريم كشف بينشر تانية **عليها مربعات**. يعني
                # ٤ فريم من كل ٥ (اللي بيتخطّوا) كانوا بيتبعتوا **من غير
                # أي مربع** → المربعات بتظهر وتختفي ٥ مرات في الثانية.
                #
                # الحل: بنقرأ **آخر كشف محفوظ** (اللي فوق) وبنرسمه على **كل**
                # فريم، مش بس فريم الكشف. فالمربعات بتفضل ثابتة للحين ما
                # الاستدلال الجاي يجيب undetش.
                #
                # ⚠️ وكمان بقى **نشر واحد** لكل فريم مكان 1+1/skip — أقل
                # ترميز، يعني معالج أهدى.
                with self._lock:
                    shown = list(self._dets)
                    age = time.time() - self._dets_at

                # ⚠️ الـ persistence **محدود بوقت**: لو الاستدلال وقف (كاميرا
                # واقفة / الموديل وقع) من غير ما يرجّع فاضي، المربعات
                # القديمة هتفضل معلّقة على مشهد اتغيّر تمامًا — وده أسوأ
                # من رعشة. بعد `BOX_MAX_AGE` بنمسحها.
                if age > BOX_MAX_AGE_S:
                    shown = []

                if self.annotate and self.detect:
                    # ⚠️ الرسم **قبل** الترميز: لو ر码منا على الـ JPEG بعد ما
                    # يترمز، النص والخطوط بتبان مقطّعة/ناعمة. على الـ BGR
                    # مباشرة الحدة كاملة. الفرق ملحوظ.
                    self._publish(draw_detections(
                        out, shown, show_conf=self.show_conf), cv2)
                else:
                    self._publish(out, cv2)
            except Exception as exc:          # noqa: BLE001
                self.last_error = str(exc)
                time.sleep(0.2)
