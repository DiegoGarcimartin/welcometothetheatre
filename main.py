import os
import re
import base64
import random
import numpy as np
import cv2
import requests
from pathlib import Path
from typing import Optional
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
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

# ── Cloud TTS — optional, swappable ───────────────────────────────────────────
# Two providers supported; whichever has a key configured is used (fish.audio
# first). Otherwise the browser's built-in speechSynthesis voices the show.
#
# fish.audio: set FISH_API_KEY. FISH_MODEL_ID picks the voice — defaults to the
# Chiquito de la Calzada community model for the bit. (Parody of a public figure;
# use your own fish.audio account.)
FISH_KEY = os.environ.get("FISH_API_KEY", "")
FISH_MODEL_ID = os.environ.get("FISH_MODEL_ID", "1f747932d1b947359760ad8b431cdf31")
FISH_BACKBONE = os.environ.get("FISH_BACKBONE", "s1")

# ElevenLabs: set ELEVENLABS_API_KEY + ELEVENLABS_VOICE_ID.
ELEVEN_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVEN_VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "")
ELEVEN_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")

if FISH_KEY:
    print(f"[startup] Cloud TTS: fish.audio (model={FISH_MODEL_ID}).")
elif ELEVEN_KEY:
    print(f"[startup] Cloud TTS: ElevenLabs (voice={ELEVEN_VOICE or 'default'}).")
else:
    print("[startup] No cloud TTS configured — using browser voice. "
          "Set FISH_API_KEY (or ELEVENLABS_API_KEY) in .env to enable.")

# ── Startup: load LibreYOLO model once ────────────────────────────────────────
# Model weights — try a repo-local copy first, then known local paths,
# then fall through to "LibreYOLO9t.pt" which libreyolo auto-downloads.
_HERE = Path(__file__).parent
# LibreYOLO9m: best object recall of the YOLO9 sizes we tested (skateboard 0.90
# vs 0.55 for the tiny model), still ~100ms/frame. Falls back to t if missing.
_MODEL_CANDIDATES = [
    str(_HERE / "weights" / "LibreYOLO9m.pt"),
    str(_HERE / "weights" / "LibreYOLO9t.pt"),
    "LibreYOLO9m.pt",  # falls through to libreyolo auto-download from HF
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
# Géneros disparatados que el público puede elegir.
GENRES = [
    "Tragedia griega",
    "Telenovela venezolana",
    "Cine negro de detectives",
    "Drama existencial nórdico",
    "Western espagueti",
    "Teatro shakespeariano",
    "Fantasía épica medieval",
    "Documental épico de naturaleza",
    "Ópera espacial",
    "Telediario catastrofista",
    "Culebrón de sobremesa",
    "Épica vikinga",
]

# Objetos de reserva, absurdos y divertidos, si no se detecta nada.
FALLBACK_OBJECTS = [
    "una piña con secretos",
    "un gato sospechoso",
    "un patito de goma",
    "un calcetín solitario",
    "una planta de interior deprimida",
    "un bocadillo a medio comer",
    "un cono de tráfico fugitivo",
    "una fregona con ambiciones",
]

# Traducción de las clases COCO (inglés) al español, para narrar y mostrar.
COCO_ES = {
    "person": "una persona", "bicycle": "una bicicleta", "car": "un coche",
    "motorcycle": "una moto", "airplane": "un avión", "bus": "un autobús",
    "train": "un tren", "truck": "un camión", "boat": "un barco",
    "traffic light": "un semáforo", "fire hydrant": "una boca de incendios",
    "stop sign": "una señal de stop", "parking meter": "un parquímetro",
    "bench": "un banco", "bird": "un pájaro", "cat": "un gato", "dog": "un perro",
    "horse": "un caballo", "sheep": "una oveja", "cow": "una vaca",
    "elephant": "un elefante", "bear": "un oso", "zebra": "una cebra",
    "giraffe": "una jirafa", "backpack": "una mochila", "umbrella": "un paraguas",
    "handbag": "un bolso", "tie": "una corbata", "suitcase": "una maleta",
    "frisbee": "un frisbee", "skis": "unos esquís", "snowboard": "una tabla de snow",
    "sports ball": "una pelota", "kite": "una cometa", "baseball bat": "un bate",
    "baseball glove": "un guante de béisbol", "skateboard": "un monopatín",
    "surfboard": "una tabla de surf", "tennis racket": "una raqueta",
    "bottle": "una botella", "wine glass": "una copa de vino", "cup": "una taza",
    "fork": "un tenedor", "knife": "un cuchillo", "spoon": "una cuchara",
    "bowl": "un cuenco", "banana": "un plátano", "apple": "una manzana",
    "sandwich": "un bocadillo", "orange": "una naranja", "broccoli": "un brócoli",
    "carrot": "una zanahoria", "hot dog": "un perrito caliente", "pizza": "una pizza",
    "donut": "un dónut", "cake": "un pastel", "chair": "una silla", "couch": "un sofá",
    "potted plant": "una maceta", "bed": "una cama", "dining table": "una mesa",
    "toilet": "un váter", "tv": "una tele", "laptop": "un portátil", "mouse": "un ratón",
    "remote": "un mando", "keyboard": "un teclado", "cell phone": "un móvil",
    "microwave": "un microondas", "oven": "un horno", "toaster": "una tostadora",
    "sink": "un fregadero", "refrigerator": "una nevera", "book": "un libro",
    "clock": "un reloj", "vase": "un jarrón", "scissors": "unas tijeras",
    "teddy bear": "un osito de peluche", "hair drier": "un secador",
    "toothbrush": "un cepillo de dientes",
}

NARRATOR_SYSTEM = (
    "Eres un narrador de teatro cómico que hace una voz en off en directo. Tu único objetivo es DAR RISA. "
    "La técnica es el BATHOS: montas algo épico y solemne y lo revientas con una realidad cutre y española "
    "(el Mercadona, la ITV, la suegra, un Cola Cao, el cuñado). Frases CORTAS, con ritmo y remate al final. "
    "Nada de prosa florida: si suena elegante pero no hace gracia, has fallado. Humor blanco y luminoso, "
    "NUNCA oscuro ni triste; nada de muerte ni sufrimiento. Español de España, coloquial y natural. "
    "NO uses coletillas ni frases hechas de humoristas (nada de 'fistro', 'no puedorrr', 'te das cuén', "
    "'pecador', etc.): escribe español normal y gracioso, sin muletillas postizas. "
    "Máximo 2 frases, 35 palabras. Continúa la historia metiendo el objeto. Apto para todos los públicos. "
    "Devuelve SOLO la narración hablada: sin acotaciones, sin efectos de sonido, sin asteriscos, sin markdown, "
    "sin saltos de línea, sin etiquetas."
)

CANNED_FALLBACK = (
    "¡Y entonces, contra todo pronóstico, no pasó absolutamente nada! "
    "El público aguantó la respiración; el apuntador miró el reloj. La función, queridos míos, continúa."
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


CONF_THRESHOLD = 0.25  # tiny COCO model — keep it low so it catches more


def _detect_objects(frame: np.ndarray) -> list[str]:
    """Run LibreYOLO and return the single most prominent object, in Spanish.

    Ignores 'person' — the user is always in frame, so we want the object
    they're holding up, not them. Returns the highest-confidence non-person
    class (translated to Spanish), or [] if nothing clears the threshold.
    """
    if not YOLO_LOADED or _yolo is None:
        return []
    try:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = _yolo(pil_img, conf=CONF_THRESHOLD)
        r = results[0] if isinstance(results, (list, tuple)) else results
        conf_arr = _np(r.boxes.conf)
        cls_arr  = _np(r.boxes.cls).astype(int)
        names    = r.names
        best_conf, best_name = 0.0, None
        for c, k in zip(conf_arr, cls_arr):
            name = names.get(int(k), str(k))
            if name != "person" and float(c) >= CONF_THRESHOLD and float(c) > best_conf:
                best_conf, best_name = float(c), name
        if best_name is None:
            return []
        return [COCO_ES.get(best_name, best_name)]
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


@app.get("/risas.mp3")
def laugh_track():
    return FileResponse(_HERE / "assets" / "risas.mp3", media_type="audio/mpeg")


# Serve the Chiquito soundboard clips (and any other audio) from /assets/*
app.mount("/assets", StaticFiles(directory=str(_HERE / "assets")), name="assets")


@app.get("/config")
def config():
    # Tells the frontend whether to use cloud TTS or the browser voice.
    return {"cloud_tts": bool(FISH_KEY or (ELEVEN_KEY and ELEVEN_VOICE))}


class TTSRequest(BaseModel):
    text: str


def _tts_fish(text: str) -> bytes:
    r = requests.post(
        "https://api.fish.audio/v1/tts",
        headers={
            "Authorization": f"Bearer {FISH_KEY}",
            "Content-Type": "application/json",
            "model": FISH_BACKBONE,
        },
        json={"text": text, "reference_id": FISH_MODEL_ID, "format": "mp3"},
        timeout=30,
    )
    r.raise_for_status()
    return r.content


def _tts_eleven(text: str) -> bytes:
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}",
        headers={
            "xi-api-key": ELEVEN_KEY,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        },
        json={
            "text": text,
            "model_id": ELEVEN_MODEL,
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.8, "style": 0.6},
        },
        timeout=20,
    )
    r.raise_for_status()
    return r.content


@app.post("/tts")
def tts(req: TTSRequest):
    """Voice a line via cloud TTS and return MP3. 503 if not configured so the
    frontend can fall back to the browser voice."""
    if FISH_KEY:
        provider = _tts_fish
    elif ELEVEN_KEY and ELEVEN_VOICE:
        provider = _tts_eleven
    else:
        raise HTTPException(status_code=503, detail="cloud TTS not configured")
    try:
        return Response(content=provider(req.text), media_type="audio/mpeg")
    except Exception as e:
        print(f"[tts] error: {e}")
        raise HTTPException(status_code=502, detail="TTS failed")


@app.get("/genres")
def genres():
    """List of selectable genres for the landing picker."""
    return {"genres": GENRES}


@app.get("/new_show")
def new_show(genre: Optional[str] = None):
    # Use the genre the audience picked; fall back to random if none/invalid.
    if not genre or genre not in GENRES:
        genre = random.choice(GENRES)
    try:
        theme = _llm(
            f"Inventa un título de obra de teatro disparatado y divertidísimo para un espectáculo "
            f"de género «{genre}» que va sobre objetos cotidianos y aburridos. "
            "Devuelve SOLO el título, sin comillas ni texto extra. Máximo 8 palabras.",
            system=("Eres un empresario teatral pomposo y con muchísima labia. "
                    "Solo títulos en español de España, máximo 8 palabras, cuanto más absurdos mejor."),
        )
    except Exception:
        theme = f"Los magníficos objetos de la {genre}"
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
    if detected:
        return {"objects": detected, "detected": True}
    # Nothing real this frame — the frontend keeps scanning, or uses the
    # funny fallback once the reading window is over.
    return {"objects": [random.choice(FALLBACK_OBJECTS)], "detected": False}


class NarrateRequest(BaseModel):
    genre: str
    theme: str
    story_so_far: str
    new_object: str = ""
    mode: str = "weave"           # "intro" | "weave" | "ending"
    is_ending: bool = False       # back-compat; equivalent to mode="ending"


_WEAVE_FALLBACK = "Y, como era de esperar, aquello lo cambió todo. Más o menos."


@app.post("/narrate")
def narrate(req: NarrateRequest):
    mode = "ending" if req.is_ending else req.mode
    try:
        if mode == "intro":
            prompt = (
                f"Género: {req.genre}. Obra: «{req.theme}».\n"
                "Abre con UNA sola frase tipo cuento ('Érase una vez…') que presente a un "
                "protagonista sencillo, en clave de comedia. Español de España. Máximo 18 palabras."
            )
        elif mode == "ending":
            prompt = (
                f"Género: {req.genre}. Obra: «{req.theme}».\n"
                f"Historia hasta ahora: {req.story_so_far}\n"
                "Cierra con UNA frase que ate los objetos que han ido apareciendo, "
                "con remate cómico por bathos. Español de España. Máximo 25 palabras."
            )
        else:  # weave — incorporate the new object into the running story
            prompt = (
                f"Género: {req.genre}. Obra: «{req.theme}».\n"
                f"Historia hasta ahora: {req.story_so_far}\n"
                f"Acaba de aparecer en escena este objeto real: {req.new_object}.\n"
                f"Continúa la historia en UNA frase corta metiendo «{req.new_object}» de forma "
                "natural pero absurda (bathos, detalle castizo). Español de España. Máximo 22 palabras."
            )
        line = _llm(prompt)
    except Exception as e:
        print(f"[narrate] LLM error: {e}")
        line = CANNED_FALLBACK if mode != "weave" else _WEAVE_FALLBACK
    return {"line": line}
