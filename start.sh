#!/bin/bash
# Start the Absurd Theater server.
# Activate your venv first (where libreyolo is installed), then run ./start.sh
# Reads $PORT if set, otherwise defaults to 8010.
exec uvicorn main:app --port "${PORT:-8010}"
