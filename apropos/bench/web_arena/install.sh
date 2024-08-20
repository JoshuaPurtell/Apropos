#!/bin/bash

# Clone the WebArena repository
cd apropos/bench/web_arena/
git clone https://github.com/web-arena-x/webarena.git
cd webarena

# Install dependencies
pip install -r requirements.txt
playwright install

# Install the package in editable mode
pip install -e .