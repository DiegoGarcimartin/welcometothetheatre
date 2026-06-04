"""El Teatro de los Cuentos — backend.

Cuentacuentos infantil (4-9 años). El niño elige un cuento clásico y va enseñando
objetos a la cámara que SUSTITUYEN elementos del cuento (la casa del cerdito, lo que
lleva Caperucita en la cesta…). El backend detecta el objeto (LibreYOLO, COCO) y un
narrador (Claude) lo teje en la historia, fiel al cuento pero con chispa, en español
de España y con final feliz.
"""
import os
import re
import base64
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import anthropic

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

_HERE = Path(__file__).parent

# ── LLM ───────────────────────────────────────────────────────────────────────
_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not _API_KEY:
    print("[startup] WARNING: ANTHROPIC_API_KEY no configurada — se usarán frases de "
          "reserva. Copia .env.example a .env y pon tu clave.")
client = anthropic.Anthropic(api_key=_API_KEY)

# ── Modelo de visión (LibreYOLO9m — el más fiable según benchmark) ─────────────
_MODEL_CANDIDATES = [
    str(_HERE / "weights" / "LibreYOLO9m.pt"),
    "LibreYOLO9m.pt",  # auto-descarga de HF si falta
]
try:
    from libreyolo import LibreYOLO
    _mp = next((p for p in _MODEL_CANDIDATES if Path(p).exists()), "LibreYOLO9m.pt")
    print(f"[startup] cargando LibreYOLO desde {_mp} …")
    _yolo = LibreYOLO(_mp)
    YOLO_LOADED = True
    print("[startup] modelo cargado ✓")
except Exception as e:
    print(f"[startup] fallo al cargar libreyolo: {e}. Se usarán los elementos clásicos.")
    _yolo = None
    YOLO_LOADED = False

CONF_THRESHOLD = 0.25

# COCO (80 clases) → español, para nombrar el objeto en el cuento.
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
    "baseball glove": "un guante", "skateboard": "un monopatín",
    "surfboard": "una tabla de surf", "tennis racket": "una raqueta",
    "bottle": "una botella", "wine glass": "una copa", "cup": "una taza",
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

# ── Cuentos ───────────────────────────────────────────────────────────────────
# Cada escena: emoji (decorado), bg (color), prompt (lo que se pide al niño),
# ctx (qué pasa, para el narrador) y default (el elemento clásico si no se detecta).
TALES = {
    "cerditos": {
        "title": "Los tres cerditos",
        "emoji": "🐷",
        "cover": "🐷🐷🐷",
        "intro_ctx": "Tres cerditos hermanos se despiden de su mamá y se van al bosque a construir, cada uno, su propia casa. Cerca ronda un lobo grande pero bastante torpe.",
        "ending_ctx": "Los tres cerditos quedan sanos y salvos, juntos en la casa más fuerte, y el lobo se marcha para siempre sin hacer daño a nadie.",
        "scenes": [
            {"emoji": "🐷🌿", "bg": "#bfe3a3", "prompt": "¿Con qué construye su casa el primer cerdito?",
             "ctx": "El primer cerdito, el más juguetón, construye su casita a toda prisa usando ESTE objeto en lugar de paja.", "default": "paja"},
            {"emoji": "🐷🪵", "bg": "#a9d99a", "prompt": "¿Y la casa del segundo cerdito?",
             "ctx": "El segundo cerdito construye su casa con ESTE objeto en lugar de palos de madera.", "default": "palos de madera"},
            {"emoji": "🐷🧱", "bg": "#e6c79c", "prompt": "¿Y la del tercer cerdito, el más trabajador?",
             "ctx": "El tercer cerdito, muy trabajador, construye una casa fuerte y resistente con ESTE objeto en lugar de ladrillos.", "default": "ladrillos"},
            {"emoji": "🐺💨", "bg": "#a7c9ec", "prompt": "¡Llega el lobo soplando! ¿Con qué lo espantan?",
             "ctx": "El lobo sopla y resopla, pero los cerditos lo espantan de forma divertida usando ESTE objeto. El lobo se asusta y huye, sin que nadie se haga daño.", "default": "una escoba"},
            {"emoji": "🐷🎉", "bg": "#f3da8c", "prompt": "¡A celebrar! ¿Qué preparan para la fiesta?",
             "ctx": "A salvo y juntos, los tres cerditos hacen una fiesta y preparan ESTE objeto como si fuera el manjar más rico del mundo.", "default": "una rica sopa"},
        ],
    },
    "caperucita": {
        "title": "Caperucita Roja",
        "emoji": "🔴",
        "cover": "🔴🐺",
        "intro_ctx": "Caperucita, una niña con una capa roja, sale de casa para llevarle la merienda a su abuelita, que vive al otro lado del bosque. Un lobo astuto la observa entre los árboles.",
        "ending_ctx": "El lobo sale corriendo, la abuelita está bien, y Caperucita, la abuelita y el leñador meriendan felices todos juntos. Nadie sufre ningún daño.",
        "scenes": [
            {"emoji": "🔴🧺", "bg": "#f4b8c1", "prompt": "¿Qué lleva Caperucita en la cesta para la abuelita?",
             "ctx": "Caperucita prepara la cesta para su abuelita y, en lugar de pasteles, mete ESTE objeto con mucho cariño.", "default": "unos pasteles"},
            {"emoji": "🐺🎭", "bg": "#c9b8e8", "prompt": "El lobo se disfraza de abuelita. ¿Con qué se disfraza?",
             "ctx": "El lobo, muy pillo, se mete en la cama y se disfraza de abuelita usando ESTE objeto para que no lo reconozcan.", "default": "un gorro y unas gafas"},
            {"emoji": "🐺👀", "bg": "#b7d9ec", "prompt": "—Abuelita… ¡qué cosa tan grande! ¿Qué le ve Caperucita al lobo?",
             "ctx": "Caperucita mira al falso abuelito y, en lugar de '¡qué orejas tan grandes!', se sorprende de ESTE objeto enorme. Empieza a sospechar.", "default": "unas orejas"},
            {"emoji": "🪓🌲", "bg": "#bfe0a8", "prompt": "¡Llega el leñador a ayudar! ¿Con qué ayuda?",
             "ctx": "Un leñador bondadoso aparece y, en lugar de su hacha, usa ESTE objeto para espantar al lobo de forma amable. El lobo huye y la abuelita aparece sana y salva.", "default": "una cuerda"},
            {"emoji": "🍰🫖", "bg": "#f3da8c", "prompt": "¡A merendar todos juntos! ¿Qué meriendan?",
             "ctx": "Caperucita, la abuelita y el leñador se sientan felices a merendar ESTE objeto como si fuera lo más delicioso.", "default": "un rico bizcocho"},
        ],
    },
    "gato": {
        "title": "El gato con botas",
        "emoji": "🐱",
        "cover": "🐱👢",
        "intro_ctx": "Un molinero le deja a su hijo pequeño solo un gato. Pero no es un gato cualquiera: es muy listo y habla, y promete hacer rico y feliz a su amo con un poco de ingenio.",
        "ending_ctx": "Gracias al ingenio del gato, el joven se convierte en un señor importante, se casa con la princesa y todos viven felices. El gato con botas se vuelve el héroe del reino.",
        "scenes": [
            {"emoji": "🐱👢", "bg": "#f0c98a", "prompt": "El gato pide algo para sus patas. ¿Qué se pone?",
             "ctx": "El gato, muy elegante, pide ponerse en las patas ESTE objeto en lugar de unas botas, para verse importante.", "default": "unas botas"},
            {"emoji": "🐱👑", "bg": "#f4d27a", "prompt": "El gato lleva un regalo al rey. ¿Qué le regala?",
             "ctx": "El gato va al palacio y le lleva al rey, de parte de su amo, ESTE objeto como un regalo magnífico.", "default": "un conejo del campo"},
            {"emoji": "🐱✨", "bg": "#bcd9ec", "prompt": "Para impresionar a la princesa, ¿qué consigue el gato?",
             "ctx": "El gato, listísimo, consigue ESTE objeto para que su amo deslumbre a la princesa y parezca un gran señor.", "default": "un traje precioso"},
            {"emoji": "🐱👹", "bg": "#d8b8e0", "prompt": "El gato engaña al ogro. ¿Con qué lo despista?",
             "ctx": "El gato se enfrenta a un ogro presumido y lo despista con astucia usando ESTE objeto. El ogro cae en la trampa (sin que nadie salga herido).", "default": "un ovillo de lana"},
            {"emoji": "🐱🎉", "bg": "#f3da8c", "prompt": "¡Gran banquete de boda! ¿Qué sirven?",
             "ctx": "En el gran banquete de la boda sirven ESTE objeto como el plato estrella, y todo el reino lo celebra.", "default": "un pastel enorme"},
        ],
    },
}

NARRATOR_SYSTEM = (
    "Eres un cuentacuentos cálido y cariñoso que narra cuentos clásicos a niños de 4 a 9 años. "
    "Hablas en español de España, con frases CORTAS, sencillas y claras, fáciles de entender para los más pequeños. "
    "Eres fiel al cuento clásico, pero con una chispa de fantasía y ternura. "
    "Cuando aparece un objeto, lo integras en la historia como si fuera lo más natural y mágico del mundo, "
    "con asombro y alegría. Tono dulce, luminoso y seguro: NUNCA das miedo, ni hay daño, sangre, muerte ni tristeza; "
    "los malos solo se asustan o se van. Nada de humor adulto, ironía ni coletillas: lenguaje de cuento infantil. "
    "Devuelve SOLO la narración hablada en prosa: sin acotaciones, sin efectos de sonido, sin asteriscos, "
    "sin markdown, sin saltos de línea, sin emojis, sin etiquetas. Máximo 2 frases, 35 palabras."
)

_FALLBACK = {
    "intro": "Érase una vez, en un lugar muy bonito, una historia a punto de empezar.",
    "weave": "Y entonces, como por arte de magia, apareció justo lo que la historia necesitaba.",
    "ending": "Y colorín colorado, este precioso cuento se ha acabado.",
}

app = FastAPI()


# ── Helpers ───────────────────────────────────────────────────────────────────
_MD = re.compile(r"[*#_`]")


def _clean(text: str) -> str:
    text = _MD.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _np(x):
    try:
        return x.cpu().numpy()
    except AttributeError:
        return np.asarray(x)


def _decode_data_url(data_url: str) -> Optional[np.ndarray]:
    try:
        data = data_url.split(",", 1)[1] if "," in data_url else data_url
        raw = base64.b64decode(data)
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[capture] decode error: {e}")
        return None


def _detect_object(frame: np.ndarray) -> Optional[str]:
    """Return the most prominent non-person object (Spanish), or None."""
    if not YOLO_LOADED or _yolo is None:
        return None
    try:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = _yolo(pil, conf=CONF_THRESHOLD)
        r = results[0] if isinstance(results, (list, tuple)) else results
        conf_arr = _np(r.boxes.conf)
        cls_arr = _np(r.boxes.cls).astype(int)
        names = r.names
        best_c, best_n = 0.0, None
        for c, k in zip(conf_arr, cls_arr):
            name = names.get(int(k), str(k))
            if name != "person" and float(c) >= CONF_THRESHOLD and float(c) > best_c:
                best_c, best_n = float(c), name
        return COCO_ES.get(best_n, best_n) if best_n else None
    except Exception as e:
        print(f"[detect] error: {e}")
        return None


def _llm(prompt: str) -> str:
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=120,
        system=NARRATOR_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return _clean(msg.content[0].text)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTMLResponse((_HERE / "index.html").read_text(encoding="utf-8"))


@app.get("/tales")
def tales():
    return {"tales": [{"id": tid, "title": t["title"], "emoji": t["emoji"], "cover": t["cover"]}
                      for tid, t in TALES.items()]}


@app.get("/tale/{tale_id}")
def tale(tale_id: str):
    t = TALES.get(tale_id)
    if not t:
        return {"error": "not found"}
    return {
        "id": tale_id, "title": t["title"], "emoji": t["emoji"], "cover": t["cover"],
        "scenes": [{"emoji": s["emoji"], "bg": s["bg"], "prompt": s["prompt"], "default": s["default"]}
                   for s in t["scenes"]],
    }


class CaptureRequest(BaseModel):
    image: Optional[str] = None


@app.post("/capture")
def capture(req: CaptureRequest):
    obj = None
    if req.image:
        frame = _decode_data_url(req.image)
        if frame is not None:
            obj = _detect_object(frame)
    return {"object": obj, "detected": obj is not None}


class NarrateRequest(BaseModel):
    tale_id: str
    scene_index: int = 0
    new_object: str = ""
    story_so_far: str = ""
    mode: str = "weave"   # "intro" | "weave" | "ending"


@app.post("/narrate")
def narrate(req: NarrateRequest):
    t = TALES.get(req.tale_id)
    if not t:
        return {"line": _FALLBACK.get(req.mode, _FALLBACK["weave"])}
    try:
        if req.mode == "intro":
            prompt = (f"Cuento: «{t['title']}». Situación inicial: {t['intro_ctx']}\n"
                      "Cuenta el COMIENZO del cuento en 1-2 frases ('Érase una vez…'), presentando a los "
                      "personajes con ternura. Para niños pequeños.")
        elif req.mode == "ending":
            prompt = (f"Cuento: «{t['title']}». Historia hasta ahora: {req.story_so_far}\n"
                      f"Final feliz: {t['ending_ctx']}\n"
                      "Cierra el cuento en 1-2 frases, con un final feliz y tierno y un 'colorín colorado'.")
        else:  # weave
            scene = t["scenes"][max(0, min(req.scene_index, len(t["scenes"]) - 1))]
            obj = req.new_object or scene["default"]
            prompt = (f"Cuento: «{t['title']}». Historia hasta ahora: {req.story_so_far}\n"
                      f"Lo que pasa ahora: {scene['ctx']}\n"
                      f"El objeto que aparece es: {obj}.\n"
                      f"Cuenta esta parte en 1-2 frases, metiendo «{obj}» en la historia con asombro y "
                      "naturalidad, fiel al cuento. Para niños pequeños.")
        return {"line": _llm(prompt)}
    except Exception as e:
        print(f"[narrate] error: {e}")
        return {"line": _FALLBACK.get(req.mode, _FALLBACK["weave"])}
