#!/usr/bin/env bash
# ==========================================
# Face Puppet Desktop Pet Launcher
# Dependencies: PySide6, opencv-python, mediapipe, numpy, scipy
# ==========================================
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if command -v python3 &>/dev/null; then
    exec python3 face_puppet_pet.py "$@"
else
    echo "Error: Python 3 not found. Please install Python 3."
    exit 1
fi
