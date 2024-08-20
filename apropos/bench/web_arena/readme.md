conda create -n webarena python=3.10; conda activate webarena
pip install -r requirements.txt
playwright install
pip install -e .