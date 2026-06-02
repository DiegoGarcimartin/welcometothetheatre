import os
import re
import base64
import random
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import anthropic

# Load a local .env if present (so each teammate just drops their key there).
try:
    from dotenv import load_dotenv
    # override=True so a populated .env wins over an empty/stale shell var.
    load_dotenv(override=True)
except ImportError:
    pass

# ── Startup: LLM client ───────────────────────────────────────────────────────
_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not _API_KEY:
    print("[startup] WARNING: ANTHROPIC_API_KEY not set — narrator will use "
          "canned fallback lines. Copy .env.example to .env and add your key.")
client = anthropic.Anthropic(api_key=_API_KEY)

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

# Note: the webcam is owned by the BROWSER (getUserMedia), not the backend.
# The frontend grabs frames and POSTs them to /capture for detection. This
# means it works for anyone who opens the page — no server-side camera
# permission needed.

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
    "Maintain continuity with the story so far. Keep it PG. "
    "Output ONLY the spoken narration as plain prose — no stage directions, "
    "no sound effects, no asterisks, no markdown, no line breaks, no labels."
)

CANNED_FALLBACK = (
    "And so fate intervenes — a presence so catastrophic, so magnificently absurd, "
    "that even the gods themselves dare not speak its name aloud. The drama continues."
)

app = FastAPI()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_data_url(data_url: str) -> Optional[np.ndarray]:
    """Decode a browser canvas data URL (or raw base64 jpeg) into a BGR frame."""
    try:
        data = data_url.split(",", 1)[1] if "," in data_url else data_url
        raw = base64.b64decode(data)
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[capture] decode error: {e}")
        return None


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


_STAGE_DIR = re.compile(r"\*[^*]*\*")  # *harmonica wails* style asides


def _clean(text: str) -> str:
    """Strip stage directions, markdown and line breaks so TTS reads clean prose."""
    text = _STAGE_DIR.sub("", text)
    text = text.replace("*", "").replace("#", "")
    text = re.sub(r"\s+", " ", text)  # collapse newlines/extra spaces
    return text.strip()


def _llm(prompt: str, system: str = NARRATOR_SYSTEM) -> str:
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=120,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return _clean(msg.content[0].text)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


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


class CaptureRequest(BaseModel):
    image: Optional[str] = None  # data URL grabbed from the browser webcam


@app.post("/capture")
def capture(req: CaptureRequest):
    detected: list[str] = []
    if req.image:
        frame = _decode_data_url(req.image)
        if frame is not None:
            detected = _detect_objects(frame)
    # If no frame / nothing detected, the show goes on with a funny fallback.
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
