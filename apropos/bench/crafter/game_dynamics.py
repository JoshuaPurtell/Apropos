import gym
import re
from typing import List, Any
from apropos.bench.crafter.recording_utils import Recorder
from apropos.bench.crafter.game_information import action_dict


# Make this more parametric
def parse_observations(data):
    print(data)
    observations = []
    # Split by "Player Observation Step"
    steps = re.split(r"Player Observation Step \d+:", data)

    for step in steps[1:]:  # skip the first split as it's header
        current_obs = {}
        # Extract action taken
        action_match = re.search(r"You took action (\w+).", step)
        if action_match:
            current_obs["action"] = action_match.group(1)
        else:
            current_obs["action"] = "noop"  # Default if no action mentioned

        # Extract observations
        current_obs["observations"] = {}
        seen_items = re.findall(r"- (\w+) (\d+) steps to your (\w+[-\w+]*)", step)
        for item, distance, direction in seen_items:
            current_obs["observations"][direction] = f"{item} {distance} steps"

        # Face direction
        face_match = re.search(r"You face (\w+) at your front.", step)
        current_obs["front"] = face_match.group(1) if face_match else "unknown"

        # Extract status
        status_matches = re.findall(r"- (\w+): (\d+)/(\d+)", step)
        current_obs["status"] = {status: f"{current}/{max_val}" for status, current, max_val in status_matches}

        if "You have nothing in your inventory." in step:
            current_obs["inventory"] = {}
        else:
            pass

        observations.append(current_obs)

    return observations

# Agent Computer Interface
class CrafterACI:
    env: gym.Env

    def __init__(self, artifacts_path="apropos/bench/crafter/artifacts", filename="v0"):
        self.filename = filename
        tmp_env = gym.make("smartplay:Crafter-v0")
        env = Recorder(
            tmp_env,
            artifacts_path,
            save_stats=False,
            save_video=True,
            save_episode=True,
        )
        seed = 42
        env.seed(seed)
        env.action_space.seed(seed)
        _, info = env.reset()
        self.env = env

    def render_observations(self, data):
        return parse_observations(data)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        step_info = {
            "obs": self.render_observations(info["obs"]),
            "reward": reward,
        }
        return step_info, reward, done
    
    def terminate(self):
        self.env.force_save(self.filename)
        self.env.close()

class DummyAgent:

    async def add_observation(self, observation):
        print(observation)
        pass

    async def get_actions(self):
        import random
        return [random.choice(list(action_dict.keys()))]
    
async def harness(agent, k_steps,filename="v0"):
    aci = CrafterACI(filename=filename)
    done = False
    step_info = None
    while not done and k_steps > 0:
        await agent.add_observation(step_info)
        actions = await agent.get_actions()
        for action in actions:
            step_info, reward, done = aci.step(action)
            if done:
                break
            else:
                k_steps -= 1
    aci.terminate()


if __name__ == "__main__":
    import asyncio
    agent = DummyAgent()
    k_steps = 10
    filename = "v0"
    asyncio.run(harness(agent, k_steps, filename=filename))