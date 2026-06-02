import os
import cv2
import time
import random
import threading
import numpy as np
from pathlib import Path
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import anthropic

# ── Startup: LLM client ───────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# ── Startup: load LibreYOLO model once ────────────────────────────────────────
# Model weights — try a repo-local copy first, then known local paths,
# then fall through to "LibreYOLO9t.pt" which libreyolo auto-downloads.
_HERE = Path(__file__).parent
_MODEL_CANDIDATES = [
    str(_HERE / "weights" / "LibreYOLO9t.pt"),
    str(Path.home() / "hackathon-cursor/libreyolo/weights/LibreYOLO9t.pt"),
    str(Path.home() / "el-juego-de-la-sepia/server/weights/LibreYOLO9t.pt"),
    str(Path.home() / "claude/vision-hackathon/libreyolo/weights/LibreYOLO9t.pt"),
    "LibreYOLO9t.pt",  # falls through to libreyolo auto-download
]

try:
    from libreyolo import LibreYOLO
    _model_path = next((p for p in _MODEL_CANDIDATES if Path(p).exists()), "LibreYOLO9t.pt")
    print(f"[startup] loading LibreYOLO from {_model_path} …")
    _yolo = LibreYOLO(_model_path)
    YOLO_LOADED = True
    print("[startup] LibreYOLO model loaded ✓")
except Exception as e:
    print(f"[startup] libreyolo load failed: {e}. Will use fallback objects.")
    _yolo = None
    YOLO_LOADED = False

# ── Startup: webcam ───────────────────────────────────────────────────────────
_cap = cv2.VideoCapture(0)
_cam_lock = threading.Lock()
WEBCAM_OK = _cap.isOpened()
if not WEBCAM_OK:
    print("[startup] WARNING: webcam not found. /video_feed will show error frame.")

# ── Constants ─────────────────────────────────────────────────────────────────
GENRES = [
    "Greek Tragedy",
    "Telenovela",
    "Film Noir",
    "Nordic Existential Drama",
    "Spaghetti Western",
    "Shakespearean",
    "Epic Fantasy",
    "Daytime Soap Opera",
]

FALLBACK_OBJECTS = [
    "a pineapple",
    "a suspicious cat",
    "a rubber duck",
    "a single lonely sock",
    "an existential houseplant",
    "a half-eaten sandwich",
    "a traffic cone",
]

NARRATOR_SYSTEM = (
    "You are a grandiloquent, omniscient theater narrator delivering a live voice-over. "
    "You treat mundane everyday objects as profound, fate-laden dramatic elements. "
    "Commit FULLY to the chosen genre. Write in English. "
    "The humor comes from over-the-top seriousness, never from breaking character. "
    "Keep each response to 2-3 sentences, maximum 45 words. "
    "Maintain continuity with the story so far. Keep it PG."
)

CANNED_FALLBACK = (
    "And so fate intervenes — a presence so catastrophic, so magnificently absurd, "
    "that even the gods themselves dare not speak its name aloud. The drama continues."
)

app = FastAPI()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _error_frame() -> bytes:
    """Return a JPEG frame with an error message."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "WEBCAM NOT FOUND", (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 3)
    cv2.putText(img, "Check your camera connection", (90, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 200), 2)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _generate_mjpeg():
    while True:
        if not WEBCAM_OK:
            frame_bytes = _error_frame()
        else:
            with _cam_lock:
                ret, frame = _cap.read()
            if not ret:
                frame_bytes = _error_frame()
            else:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_bytes = buf.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.04)  # ~25 fps


def _np(x):
    try:
        return x.cpu().numpy()
    except AttributeError:
        return np.asarray(x)


def _detect_objects(frame: np.ndarray) -> list[str]:
    """Run LibreYOLO on frame, return unique class names with conf > 0.5."""
    if not YOLO_LOADED or _yolo is None:
        return []
    try:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = _yolo(pil_img)
        r = results[0] if isinstance(results, (list, tuple)) else results
        conf_arr = _np(r.boxes.conf)
        cls_arr  = _np(r.boxes.cls).astype(int)
        names    = r.names
        seen = set()
        for c, k in zip(conf_arr, cls_arr):
            if float(c) > 0.5:
                seen.add(names.get(int(k), str(k)))
        return list(seen)
    except Exception as e:
        print(f"[detect] error: {e}")
        return []


def _llm(prompt: str, system: str = NARRATOR_SYSTEM) -> str:
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=120,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/new_show")
def new_show():
    genre = random.choice(GENRES)
    try:
        theme = _llm(
            f"Generate a dramatic play title for a {genre} show about mundane everyday objects. "
            "Return ONLY the title, no extra text. Max 10 words.",
            system="You are a pompous theater impresario. Titles only, maximum 10 words.",
        )
    except Exception:
        theme = f"The Magnificent Objects of {genre}"
    return {"genre": genre, "theme": theme}


@app.post("/capture")
def capture():
    if not WEBCAM_OK:
        objects = [random.choice(FALLBACK_OBJECTS)]
        return {"objects": objects}
    with _cam_lock:
        ret, frame = _cap.read()
    if not ret:
        return {"objects": [random.choice(FALLBACK_OBJECTS)]}
    detected = _detect_objects(frame)
    if not detected:
        detected = [random.choice(FALLBACK_OBJECTS)]
    return {"objects": detected}


class NarrateRequest(BaseModel):
    genre: str
    theme: str
    story_so_far: str
    new_object: str
    is_ending: bool = False


@app.post("/narrate")
def narrate(req: NarrateRequest):
    try:
        if req.is_ending:
            prompt = (
                f"Genre: {req.genre}. Play: \"{req.theme}\".\n"
                f"Story so far: {req.story_so_far}\n"
                "Deliver a FINAL 2-3 sentence dramatic conclusion. "
                "The story ends with operatic grandeur and perhaps a hint of absurdity. "
                "Max 45 words."
            )
        else:
            prompt = (
                f"Genre: {req.genre}. Play: \"{req.theme}\".\n"
                f"Story so far: {req.story_so_far}\n"
                f"A new object has entered the stage: {req.new_object}.\n"
                "Write 2-3 sentences treating this object as deeply fate-laden and dramatically significant. "
                "Escalate the tension. Max 45 words."
            )
        line = _llm(prompt)
    except Exception as e:
        print(f"[narrate] LLM error: {e}")
        line = CANNED_FALLBACK
    return {"line": line}
