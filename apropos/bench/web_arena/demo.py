import random
import os
import sys
sys.path.append("/Users/jp/Documents/GitHub/Apropos/apropos/bench/web_arena/webarena")
#apropos.bench.web_arena.webarena.
from browser_env import ScriptBrowserEnv, create_id_based_action


# init the environment
env = ScriptBrowserEnv(
    headless=False,
    observation_type="accessibility_tree",
    current_viewport_only=True,
    viewport_size={"width": 1280, "height": 720},
)
# prepare the environment for a configuration defined in a json file
config_file = "apropos/bench/web_arena/webarena/config_files/examples/1.json"
# TODO: Do once I get connection ! 
# playwright._impl._api_types.Error: Executable doesn't exist at /Users/jp/Library/Caches/ms-playwright/chromium-1055/chrome-mac/Chromium.app/Contents/MacOS/Chromium
obs, info = env.reset(options={"config_file": config_file})
# get the text observation (e.g., html, accessibility tree) through obs["text"]

# create a random action
id = random.randint(0, 1000)
action = create_id_based_action(f"click [id]")

# take the action
obs, _, terminated, _, info = env.step(action)