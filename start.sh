#!/bin/bash
exec /Users/diegogarcimartinrey/el-juego-de-la-sepia/.venv/bin/uvicorn main:app --port "${PORT:-8010}"
