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
    "La técnica es el BATHOS: montas una frase épica y solemne y la revientas de golpe con una realidad cutre, "
    "doméstica y española. Ejemplo del mecanismo: 'Los dioses del Olimpo contuvieron el aliento ante el héroe… "
    "que resultó ser una fregona del Mercadona con dos años de antigüedad.' Ese choque entre lo grandioso y lo "
    "ridículamente normal es TODA la gracia. "
    "Reglas de comedia: frases CORTAS y con ritmo; remate cómico al final (lo más fuerte, lo último); "
    "detalles concretos y absurdos (marcas, precios, cuñados, el Lidl, la suegra, la ITV, un Cola Cao) mejor que "
    "metáforas bonitas; sé inesperado, anticlímax, exageración tonta. Nada de prosa florida ni poética: si suena "
    "elegante pero no hace gracia, has fallado. "
    "Mantente en el género elegido, pero la comedia manda sobre el género. Humor blanco y luminoso, "
    "NUNCA oscuro, triste ni dramático de verdad; nada de muerte, sangre ni sufrimiento. "
    "Español de España, coloquial y castizo (vale 'madre mía', 'menudo percal', 'ni tan mal'). "
    "Máximo 2-3 frases, 40 palabras. Continúa la historia. Apto para todos los públicos. "
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


CONF_THRESHOLD = 0.35  # tiny COCO model — keep it generous so it catches objects


def _detect_objects(frame: np.ndarray) -> list[str]:
    """Run LibreYOLO and return detected objects in Spanish, best first.

    Ignores 'person' — the user is always in frame, so we want the object
    they're holding up, not them. Returns unique class names sorted by
    confidence (most prominent object first), translated to Spanish.
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
        # collect (confidence, english_name), skip 'person', best first
        dets = []
        for c, k in zip(conf_arr, cls_arr):
            name = names.get(int(k), str(k))
            if float(c) >= CONF_THRESHOLD and name != "person":
                dets.append((float(c), name))
        dets.sort(reverse=True)
        seen, out = set(), []
        for _, name in dets:
            es = COCO_ES.get(name, name)
            if es not in seen:
                seen.add(es)
                out.append(es)
        return out[:3]  # at most 3 objects, the most prominent ones
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
    # If no frame / nothing detected, the show goes on with a funny fallback.
    if not detected:
        detected = [random.choice(FALLBACK_OBJECTS)]
    return {"objects": detected}


class NarrateRequest(BaseModel):
    genre: str
    theme: str
    story_so_far: str
    new_object: str = ""
    mode: str = "reveal"          # "intro" | "setup" | "reveal" | "ending"
    is_ending: bool = False       # back-compat; equivalent to mode="ending"


_SETUP_FALLBACKS = [
    "Y entonces, en mitad de la escena, el protagonista sacó muy despacio su…",
    "El silencio era absoluto cuando, de pronto, apareció sobre las tablas…",
    "Nadie esperaba lo que el héroe llevaba escondido bajo la capa:…",
]
_REVEAL_FALLBACK = "…¡y resultó ser justo eso! Madre mía, qué giro. El público no daba crédito."


@app.post("/narrate")
def narrate(req: NarrateRequest):
    mode = "ending" if req.is_ending else req.mode
    try:
        if mode == "ending":
            prompt = (
                f"Género: {req.genre}. Obra: «{req.theme}».\n"
                f"Historia hasta ahora: {req.story_so_far}\n"
                "Cierra la función con un FINAL de 2 frases: resuelve el cuento de forma sencilla y "
                "ridícula, con remate cómico por bathos. Español de España. Máximo 35 palabras."
            )
        elif mode == "intro":
            prompt = (
                f"Género: {req.genre}. Obra: «{req.theme}».\n"
                "Abre la función con UNA frase tipo cuento ('Érase una vez…') que presente a un "
                "protagonista sencillo y la situación, en clave de comedia. Español de España. Máximo 25 palabras."
            )
        elif mode == "setup":
            prompt = (
                f"Género: {req.genre}. Obra: «{req.theme}».\n"
                f"Historia hasta ahora: {req.story_so_far}\n"
                "Continúa la historia con UNA frase corta y sencilla que termine JUSTO antes de un objeto, "
                "dejándola a medias con puntos suspensivos, para que el público traiga un objeto real. "
                "Ejemplos de formato: 'El príncipe salió al galope montado en su…', "
                "'La princesa abrió el cofre del tesoro y dentro había…', "
                "'El malvado hechicero alzó por encima de su cabeza su temible…'. "
                "NO nombres tú ningún objeto. Termina obligatoriamente en '…'. Español de España. Máximo 18 palabras."
            )
        else:  # reveal
            prompt = (
                f"Género: {req.genre}. Obra: «{req.theme}».\n"
                f"La frase quedó a medias así: «{req.story_so_far}»\n"
                f"El público ha traído este objeto real: {req.new_object}.\n"
                f"Completa la frase metiendo «{req.new_object}» como si fuera lo más normal del mundo, "
                "y añade un remate cómico por lo absurdo del objeto en ese cuento (bathos/anticlímax). "
                "Empieza enlazando con la frase a medias. Español de España. Máximo 30 palabras."
            )
        line = _llm(prompt)
    except Exception as e:
        print(f"[narrate] LLM error: {e}")
        line = (random.choice(_SETUP_FALLBACKS) if mode == "setup"
                else _REVEAL_FALLBACK if mode == "reveal"
                else CANNED_FALLBACK)
    return {"line": line}
