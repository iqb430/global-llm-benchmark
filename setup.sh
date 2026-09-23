#!/bin/bash
echo "[*] Setting up local environment pakai uv..."
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
echo "[*] Siap nguli. Jangan lupa pull datasetnya."
