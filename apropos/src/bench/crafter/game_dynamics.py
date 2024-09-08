from abc import abstractmethod
import gym
import re
from typing import List, Any, Dict, Tuple
from apropos.src.bench.crafter.recording_utils import Recorder
from apropos.src.bench.crafter.game_information import action_dict
from PIL import Image
from typing import Optional
import imageio


class StepInfo:
    prev_action_taken: str
    surroundings: Dict[str, str]
    front: str
    inventory: Dict[str, int]
    status: Dict[str, str]
    new_achievements: List[str]

    def __init__(self, prev_action_taken, surroundings, front, inventory, status):
        self.prev_action_taken = prev_action_taken
        self.surroundings = surroundings
        self.front = front
        self.inventory = inventory
        self.status = status
        self.new_achievements = []

    def visual_readout(self):
        readout = f"""
<action_taken>
{self.prev_action_taken}
</action_taken>
"""
        if self.new_achievements:
            readout += f"""
<new_achievements>
{self.new_achievements}
</new_achievements>
"""
        return readout

    def default_readout(self):
        readout = f"""
<action_taken>
{self.prev_action_taken}
</action_taken>
<surroundings>
{self.surroundings}
</surroundings>
<front>
{self.front}
</front>
<inventory>
{self.inventory}
</inventory>
<status>
{self.status}
</status>
"""
        if self.new_achievements:
            readout += f"""
<new_achievements>
{self.new_achievements}
</new_achievements>
"""
        return readout


def parse_observations(raw_obs: str) -> List[StepInfo]:
    steps = re.split(r"Player Observation Step \d+:", raw_obs)
    step_infos = []
    for step in steps:
        prev_action_taken = re.search(r"You took action (\w+).", step)
        prev_action_taken = prev_action_taken.group(1) if prev_action_taken else "noop"
        surroundings = {
            direction: f"{item} {distance} steps"
            for item, distance, direction in re.findall(
                r"- (\w+) (\d+) steps to your (\w+[-\w+]*)", step
            )
        }
        front = re.search(r"You face (\w+) at your front.", step)
        front = front.group(1) if front else "unknown"
        status = {
            status: f"{current}/{max_val}"
            for status, current, max_val in re.findall(r"- (\w+): (\d+)/(\d+)", step)
        }
        inventory = (
            {
                item: int(count)
                for item, count in re.findall(r"- (\w+): (\d+)", step)
                if item not in ["health", "food", "drink", "energy"]
            }
            if "You have nothing in your inventory." not in step
            else {}
        )
        step_infos.append(
            StepInfo(prev_action_taken, surroundings, front, inventory, status)
        )
    return step_infos


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
        obs, info = env.reset()
        self.starting_obs = {
            "vis_obs": Image.fromarray(obs).tobytes(),
            "text_obs": self.render_observations(info["obs"]),
            "reward": 0,
        }
        self.env = env

    def render_observations(self, data):
        return parse_observations(data)

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        step_info = {
            "vis_obs": Image.fromarray(obs).tobytes(),
            "text_obs": self.render_observations(info["obs"]),
            "achievements": info["achievements"],
            "reward": reward,
        }
        return step_info, reward, done

    @abstractmethod
    def step(self, action):
        pass

    def _terminate(self):
        self.env.force_save(self.filename)
        self.env.close()

    @abstractmethod
    def terminate(self):
        pass


class StatelessCrafterACI(CrafterACI):
    def multistep(self, actions: List[int]) -> Tuple[List[StepInfo], List[int], bool]:
        done = False
        step_infos = []
        rewards = []
        for action in actions:
            step_info, reward, done = self._step(action)
            step_infos.append(step_info)
            rewards.append(reward)
            if done:
                break
        return step_infos, rewards, done

    def terminate(self):
        self._terminate()


class StatefulCrafterACI(CrafterACI):
    achievements: Dict[str, int]
    # validate actions, expanded territory, etc
    # keep track of achieved rewards

    def __init__(self, verbose=False, filename="v0"):
        super().__init__(filename=filename)
        self.achievements = {}
        self.verbose = verbose

    def _update_achievements(self, info):
        new_achievements = []
        for k, v in info["achievements"].items():
            if v > 0 and (k not in self.achievements or self.achievements[k] == 0):
                new_achievements.append(k)
                if self.verbose:
                    print(f"Achievement Unlocked: {k}")
        info["text_obs"][0].new_achievements = new_achievements
        self.achievements = info["achievements"]
        return info

    def multistep(self, actions: List[int]) -> Tuple[List[StepInfo], List[int], bool]:
        done = False
        step_infos = []
        rewards = []
        for action in actions:
            step_info, reward, done = self._step(action)
            step_info = self._update_achievements(step_info)
            step_infos.append(step_info)
            rewards.append(reward)
            if done:
                break
        return step_infos, rewards, done

    def terminate(self) -> Dict[str, int]:
        self._terminate()
        return self.achievements

    def return_sparse_rewards(self):
        pass

    def return_shaped_rewards(self):
        pass

    pass


class DummyAgent:
    async def add_observation(self, observations: List[Any]):
        pass

    async def get_actions(self):
        import random

        return [random.choice(list(action_dict.keys()))]


async def harness(agent, k_steps, filename="v0", mode="stateful", verbose=False):
    if mode == "stateless":
        aci = StatelessCrafterACI(filename=filename)
    elif mode == "stateful":
        aci = StatefulCrafterACI(filename=filename, verbose=verbose)
    else:
        raise ValueError("Invalid Env mode")
    done = False
    step_infos = [aci.starting_obs]
    while not done and k_steps > 0:
        await agent.add_observations(step_infos)
        actions = await agent.get_actions()
        k_steps -= len(actions)
        step_infos, rewards, done = aci.multistep(actions)
        if done:
            break
    achievements = aci.terminate()
    return achievements


if __name__ == "__main__":
    import asyncio

    agent = DummyAgent()
    k_steps = 10
    filename = "v0"
    asyncio.run(harness(agent, k_steps, filename=filename))


#     Actions ['down', 'do']
# Action enum:  Action.DOWN
# Entering jdb:
# (jdb) state_before.player_position
# Array([32, 32], dtype=int32)
# (jdb) state.player_position
# # Array([33, 32], dtype=int32)

# Action enum:  Action.RIGHT
# Entering jdb:
# (jdb) state_before.player_position
# Array([35, 32], dtype=int32)
# (jdb) state.player_position
# Array([35, 33], dtype=int32)
