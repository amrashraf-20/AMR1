# AMR1 — AI & Vision Subsystem

Standalone YOLO object detection, proactive scene narration, and vision Q&A for a
camera-equipped robot. Extracted from a larger car-control project; this repo
contains **only** the vision modules — no motor control, no database, no web UI.

---

## What's in here

| File | What it does |
|---|---|
| `vision.py` | **Core.** YOLO detection, frame enhancement (CLAHE), scene classification, natural-language narration, and the background `VisionPipeline` worker. |
| `vision_autopilot.py` | Turns detections + telemetry into a driving **decision** (go / slow / stop / reverse). Pure logic — no hardware. |
| `visual_servo.py` | Visual servoing: tracks a target across frames and steers toward it (`Tracker`, `VisualServo`). |

---

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

**GPU (optional).** The default `torch` from PyPI is a CPU build, which is fine
for inference at low frame rates. For CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`VisionPipeline` picks up CUDA automatically via ultralytics. It runs fine on CPU.

On first run ultralytics downloads the YOLO weights (`yolov8n.pt`, ~6 MB) into
its cache dir. To use a different model, pass `weights=Path("yolov8s.pt")` to
`Detector`.

---

## Run it

`VisionPipeline` needs *something* that can hand it frames. The smallest useful
adapter is an object exposing `latest()` (returns JPEG bytes), `seq()`, and
`push_remote(bytes)`. If you already have OpenCV, two lines:

```python
import cv2, threading
from vision import VisionPipeline

class CvCamera:
    """Minimal adapter over OpenCV."""
    def __init__(self, src=0):
        self._cap = cv2.VideoCapture(src)
        self._latest = None
        self._seq = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            ok, frame = self._cap.read()
            if not ok:
                continue
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                self._latest = buf.tobytes()
                self._seq += 1

    def latest(self):        return self._latest
    def seq(self):           return self._seq
    def push_remote(self, b): pass

pipe = VisionPipeline(CvCamera(), conf=0.4, enhance=True)
pipe.start()

import time
while True:
    for d in pipe.detections():
        print(f"{d.label:12} {d.conf:.2f}  bbox={d.bbox}")
    time.sleep(0.2)
```

Also useful:

```python
pipe.frame()             # latest JPEG bytes (enhanced + annotated)
pipe.status()            # dict: ready / fps / seq / detector status
pipe.stop()
```

---

## Narration (proactive scene description)

`describe()` turns a detection list into a plain-English sentence about what the
robot is looking at. It's pure and synchronous — no model call, no network:

```python
from vision import describe, classify_path

dets = pipe.detections()
print(describe(dets)["text"])
# -> "Path blocked. A person 1.2 m ahead."

classify_path(dets)
# -> {"clear": False, "nearest": Detection(...), "obstacles": ["person"], ...}
```

`classify_path` is the raw signal (is the path clear? what's nearest? what's
beside me?), and `describe` is the prose wrapper around it.

---

## Vision Q&A

Ask a question about the current frame. The pipeline's own narration answers
"what do you see?" style queries; `vision_autopilot.decide()` answers
"what should I do?" from detections + telemetry:

```python
from vision_autopilot import decide

action = decide(dets, tele, vision_age_s=0.3)
# -> {"action": "stop", "reason": "obstacle ahead", "speed": 0.0, ...}
```

`vision_age_s` matters: if the camera is stale, the autopilot refuses to move.
Pass `tele` as `None` if you have no telemetry — the decision degrades to
vision-only.

To wire the narration model in as a backend, `vision_autopilot.attach(module)`
binds a vision-language model so it can answer free-form questions about frames.
Without a model attached it stays deterministic and offline.

---

## Integrating into your own project

The only real dependency is the camera adapter interface:

```python
class MyCamera:
    def latest(self) -> bytes | None: ...   # JPEG/encoded frame
    def seq(self) -> int: ...               # monotonically increasing
    def push_remote(self, data: bytes) -> None: ...   # inbound frames
```

Swap the adapter, keep the pipeline:

```python
# --- obstacle avoidance ---
from vision import VisionPipeline
from vision_autopilot import decide

pipe = VisionPipeline(MyCamera(), conf=0.35)
pipe.start()

while driving:
    act = decide(pipe.detections(), telemetry, vision_age_s=age())
    if act["action"] == "stop":
        brake()
```

```python
# --- track and follow a person ---
from visual_servo import VisualServo

servo = VisualServo(pipe.detections(), target_label="person")
cmd = servo.update()
# cmd.steer in [-1, 1], cmd.forward is a distance estimate
```

`visual_servo` imports `point_nav` lazily, only inside the motion-planning code
paths. Detection and tracking work without it; if you call the planner helpers,
drop a `point_nav.py` beside these files.

---

## Notes

- **Frame skipping.** `VisionPipeline` processes every Nth frame (`skip=`) and
  always keeps the *latest* frame — it never lags behind reality.
- **Enhancement.** `enhance=True` applies CLAHE, which measurably helps YOLO in
  low light and high-contrast scenes. Turn it off for a straight passthrough.
- **CPU vs GPU.** `yolov8n` on CPU is roughly 5–15 FPS for a 640px stream. Use a
  smaller `conf` threshold or `skip` if you're CPU-bound.

## License

Internal / academic use. The YOLO weights carry Ultralytics' AGPL-3.0 license —
review that before commercial use.
