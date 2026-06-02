# 🎭 The Absurd Theater

A webcam-based interactive theater. An AI narrator improvises a dramatic story
in a random genre (Greek tragedy, film noir, telenovela…) and weaves in whatever
real objects you hold up to the camera — narrating a coffee mug or a banana with
full operatic gravity. One continuous ~90s performance.

- **Backend:** Python + FastAPI (`main.py`)
- **Vision:** [libreyolo](https://pypi.org/project/libreyolo/) object detection on the webcam frame
- **Narration:** Anthropic Claude (`claude-sonnet-4-5`)
- **Voice:** browser Web Speech API (no cloud TTS needed)
- **Frontend:** a single `index.html` (velvet curtains, gold title cards, subtitles)

---

## Setup

Requires **Python 3.10+** and a **webcam**.

```bash
git clone https://github.com/DiegoGarcimartin/welcometothetheatre
cd welcometothetheatre

# create + activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install deps (this pulls torch via libreyolo — a few hundred MB)
pip install -r requirements.txt

# set your Anthropic key (get the team key from Diego — Slack / 1Password)
cp .env.example .env
# then edit .env and paste the key after ANTHROPIC_API_KEY=
```

> The model weights (`weights/LibreYOLO9t.pt`) ship in the repo, so there's
> **no download** on first run. The narrator needs the key in `.env`; without
> it the show still runs but uses canned fallback lines.

## Run

```bash
./start.sh
# or: uvicorn main:app --port 8010
```

Open **http://localhost:8010** and click **Start the Show**.

> Your browser will ask for **camera permission** — the *backend* owns the
> webcam via OpenCV, so make sure no other app is using it. Use **Chrome** for
> the best Web Speech API voices.

## How it works

| Endpoint | Purpose |
|---|---|
| `GET /` | serves the HTML stage |
| `GET /video_feed` | MJPEG stream of the webcam |
| `GET /new_show` | picks a random genre + an LLM-generated play title |
| `POST /capture` | grabs a frame, runs detection, returns objects (conf > 0.5; funny fallback if none) |
| `POST /narrate` | returns a 2-3 sentence dramatic continuation in genre |

If `ANTHROPIC_API_KEY` is unset or the API fails, the narrator falls back to
canned dramatic lines so the show never stalls. If the webcam can't open, the
stage shows a clear error frame.

The `narrate()` and `speak()` functions are isolated so they can be swapped for
a different LLM or a cloud TTS later.
