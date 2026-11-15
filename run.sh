#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "Setting up virtual environment (first time only)..."
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
  mkdir -p models
  if [ ! -f "models/hand_landmarker.task" ]; then
    echo "Downloading hand tracking model..."
    curl -L -o models/hand_landmarker.task \
      "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
  fi
else
  .venv/bin/python -m pip install -q -r requirements.txt
fi

if ! .venv/bin/python -c "from src.devices import find_loopback; raise SystemExit(0 if find_loopback() is not None else 1)"; then
  echo "Installing BlackHole so Cmd+Shift+5 can record the synth sound..."
  if command -v brew >/dev/null 2>&1; then
    brew install --cask blackhole-2ch || echo "Could not install BlackHole automatically."
  else
    echo "Install Homebrew, then: brew install --cask blackhole-2ch"
  fi
  echo "If BlackHole is newly installed, restart the Mac once, then run ./run.sh again."
fi

exec .venv/bin/python main.py "$@"
