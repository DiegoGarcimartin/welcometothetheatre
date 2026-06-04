# 🎭 El Teatro de los Cuentos

Un **cuentacuentos interactivo para niños (4–9 años)**. El niño elige un cuento
clásico y va **enseñando objetos a la cámara** que **se convierten en parte del
cuento**: la casa del cerdito, lo que lleva Caperucita en la cesta, las botas del
gato… El objeto detectado aparece en la escena y el narrador lo teje en la
historia, fiel al cuento pero con chispa, en español de España y con final feliz.

- **Cuentos:** Los tres cerditos · Caperucita Roja · El gato con botas
- **Backend:** Python + FastAPI (`main.py`)
- **Visión:** [libreyolo](https://pypi.org/project/libreyolo/) (modelo `LibreYOLO9m`), la cámara la abre el **navegador**
- **Narración:** Anthropic Claude (`claude-sonnet-4-5`), tono dulce para niños
- **Voz:** Web Speech API del navegador (es-ES, lenta y clara)
- **Frontend:** un solo `index.html` (teatro luminoso, escenas con emoji + la foto del objeto del niño)

---

## Cómo funciona

1. El niño elige uno de los 3 cuentos.
2. Se abren las cortinas y empieza el cuento ("Érase una vez…").
3. En **5 momentos**, el cuento le pide algo ("¿Con qué construye su casa el
   cerdito?"). El niño **enseña un objeto a la cámara**.
4. El objeto se reconoce, su **foto entra en la escena** con destellos, y el
   narrador lo mete en la historia.
5. Final feliz, telón y un **librito** con todo su cuento y sus fotos.

> Si la cámara no reconoce el objeto (o no hay cámara), el cuento sigue con el
> **elemento clásico** de esa escena (paja, palos, ladrillos…), así nunca se corta.

> ⚠️ El modelo reconoce ~80 objetos comunes de casa (peluche-oso, pelota, libro,
> taza, botella, cuchara, plátano…), **no** la mayoría de juguetes. Para que
> acierte, enseña objetos cotidianos.

## Instalación

Requiere **Python 3.10+** y **webcam**.

```bash
git clone https://github.com/DiegoGarcimartin/welcometothetheatre
cd welcometothetheatre

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # baja torch vía libreyolo (unos cientos de MB)

cp .env.example .env               # pega tu clave de Anthropic dentro
```

> El modelo (`weights/LibreYOLO9m.pt`) viene en el repo: sin descargas al
> arrancar. Sin la clave de Anthropic el cuento se cuenta con frases de reserva.

## Ejecutar

```bash
./start.sh                          # o: uvicorn main:app --port 8010
```

Abre **http://localhost:8010** en **Chrome**, elige un cuento y permite la cámara.
Sube el volumen (la voz es el hilo principal del cuento).

## Endpoints

| Endpoint | Para qué |
|---|---|
| `GET /` | sirve el teatro (HTML) |
| `GET /tales` | lista de cuentos |
| `GET /tale/{id}` | escenas de un cuento (emoji, fondo, pregunta, elemento clásico) |
| `POST /capture` | detecta el objeto del frame (en español; ignora personas) |
| `POST /narrate` | narra una parte (`intro` / `weave` / `ending`) metiendo el objeto |

La voz (`speak()`) está aislada por si se quiere cambiar a una TTS en la nube.
