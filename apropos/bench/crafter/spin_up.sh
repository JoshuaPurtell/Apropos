#!/bin/bash

pip install wheel==0.38.4 setuptools==66.0.0
pip install gym==0.21.0
pip install minedojo
pip install crafter

REPO_URL="https://github.com/microsoft/SmartPlay"
REPO_NAME=$(basename "$REPO_URL" .git)

git clone https://github.com/microsoft/SmartPlay datasets/crafter/SmartPlay

pip install -e datasets/crafter/SmartPlay
pip install "imageio[ffmpeg]"
