import gym
import re
from apropos.bench.crafter.recording_utils import Recorder

if __name__ == "__main__":
    
    tmp_env = gym.make("smartplay:Crafter-v0")
    env = Recorder(
        tmp_env,
        "apropos/bench/crafter/artifacts",
        save_stats=False,
        save_video=True,
        save_episode=True,
    )
    seed = 42
    env.seed(seed)
    env.action_space.seed(seed)
    _, info = env.reset()
    print(parse_observations(info["obs"]))