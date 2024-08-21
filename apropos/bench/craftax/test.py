import asyncio
from typing import Dict, List, Literal

import numpy as np
from pydantic import BaseModel

from apropos.bench.craftax.game_dynamics import harness
from apropos.bench.craftax.game_information import (
    craftax_classic_action_dict,
    craftax_classic_game_rules,
    craftax_full_action_dict,
    craftax_full_game_rules,
    craftax_game_tips,
)
from apropos.bench.crafter.game_information import (
    crafter_game_tips,
)
from apropos.src.lms.cost import CostMonitor
from apropos.src.lms.helpers import LLM


class ReAct(BaseModel):
    reasoning: str
    actions: List[str]

class SimpleReActLanguageAgent:
    obs_history: List[Dict]
    react_history: List[Dict]
    lm: LLM
    cost_monitor: CostMonitor

    def __init__(self, lm: LLM, cost_monitor: CostMonitor, mode: Literal["craftax_classic","craftax_full"] = "craftax_classic"):
        self.obs_history = []
        self.react_history = []
        self.lm = lm
        self.cost_monitor = cost_monitor
        self.mode = mode

    def render_history(self):
        def remove_empty_error_section(d):
            if "errors" in d and not len(d["errors"]["illegal_actions"])>0:
                d.pop("errors")
            return d
        react_history = [f"<{i} reasoning step(s) in the past>{remove_empty_error_section(item)}</{i} reasoning step(s) in the past>" for i, item in enumerate(reversed(self.react_history[-5:]), 1)]
        obs_history = [f"<{i} environment step(s) in the past>{remove_empty_error_section(item)}</{i} environment step(s) in the past>" for i, item in enumerate(reversed(self.obs_history[-5:]), 1)]
        return "\n".join(react_history), "\n".join(obs_history)

    async def get_actions(self):
        if self.mode == "craftax_classic":
            rules = craftax_classic_game_rules
            game_tips = crafter_game_tips
            actions = craftax_classic_action_dict
        elif self.mode == "craftax_full":
            rules = craftax_full_game_rules
            game_tips = craftax_game_tips
            actions = craftax_full_action_dict
        else:
            raise ValueError(f"Mode {self.mode} not recognized")
        system_message = f"""
# Premise
You're playing the game of Crafter.
Here is some information about this setting
<Crafter Information>
<Rules>
{rules}
</Rules>
<Tips>
{game_tips}
</Tips>
<Actions Available>
{[a for a in list(actions.keys()) if a.lower() not in ['noop']]}
</Actions Available>
You'll be given your past actions/thoughts, along with recent raw observations from the environment
The environment one step in the past is your current environment

# Objective
Please choose 1-3 actions to take next in the game, and provide a thought justifying the long-term end you'd like to achieve with these actions.
These actions will be executed sequentially in the game environment, and the results will be shared with you in the next round. 

# Constraints
- If your previous actions worked toward a valuable goal, continue in that direction unless you have a good reason to change course.
- If there's no important thought to share, don't share anything. "No update" is a valid and the most common response.
- Render your actions by name. All rendered names must match the names in the "Actions Available" section.
- If you attempt an action that requires certain conditions to be met without meeting those conditions, the action will fail.
- Never select 'Do' multiple times in a row unless fighting a mob or enemy. In other cases, it's a wasted action. 
    When you select 'Do', you will interact with whatever is one step in the direction you are facing (indicated under the Object_you_are_facing heading if there is anything) and zero steps away in every other direction. If there is nothing one step (e.g. one step up, one step down, one step right, or one step left) in the direction you are facing, your 'Do' action will fail.
        - If something is e.g. one step up and one step right, attempting to interact with it will fail. You must be only one step away in the direction you are facing.
- To change the direction you are facing, move in the direction you want to face.
"""
        react_history, obs_history = self.render_history()
        user_message = f"""
# Recent Actions / Thoughts
{react_history}
# Recent Observations
{obs_history}

Your next actions / thought: """
        
        react_step = await self.lm.async_respond(
            system_prompt = system_message,
            user_prompt = user_message,
            response_model = ReAct,
        )
        
        illegal_actions = [action for action in react_step.actions if action not in craftax_classic_action_dict.keys()]
        legal_actions = [action for action in react_step.actions if action in craftax_classic_action_dict.keys()]
        react_info = react_step.dict()
        react_info["errors"] = {
            "illegal_actions": illegal_actions,
        }
        
        self.react_history.append(react_info)
        self.cost_monitor.update_token_counts(system_message, user_message, str(react_step.dict()))
        
        return legal_actions

    async def add_observations(self, observations: List[Dict]):
        for observation in observations:
            self.obs_history.append(observation["state"])#[0].default_readout()

def hafner_scoring_function_modified(achievements_probs: Dict):
    probs = list(achievements_probs.values())
    return sum([np.log(1 + prob) for prob in probs])

# Log token count to get approx cost of rollout.
async def score_rollouts(k_steps=10, base_seed=0, n_rollouts=10, verbose=False, model_name="gpt-4o-mini", modality: Literal["text","vision"]="text", mode: Literal["craftax_classic","craftax_full"]="craftax_classic"):
    core_lm = LLM(model_name)
    cost_monitor = CostMonitor(
        model_name=core_lm.model_name
    )
    if modality == "text":
        agent = SimpleReActLanguageAgent(lm=core_lm, cost_monitor=cost_monitor, mode=mode)
    elif modality == "vision":
        raise NotImplementedError("Vision not implemented yet")
    async def single_rollout(seed):
        #try:
        achievements_log = await harness(agent, seed=base_seed+seed, k_steps=k_steps, verbose=verbose, mode=mode)
        # except Exception as e:
        #     print(f"Error in Rollout: {e}")
        #     return
        achievements = list({k: v for k, v in achievements_log.items() if v > 0}.keys())
        print("Achievements", achievements)
        action_density = len(agent.obs_history) / len(agent.react_history)
        price, n_tokens = cost_monitor.final_cost()
        if verbose and action_density < 2:
            print(f"Warning - Low Action Density of {action_density}")
        if verbose:
            print(f"Rollout price ${price:.2f} for {n_tokens} tokens")
        return achievements, price
    achivements_and_price_by_rollout = await asyncio.gather(*[single_rollout(seed=_) for _ in range(n_rollouts)])
    achivements_and_price_by_rollout = [a_and_p for a_and_p in achivements_and_price_by_rollout if a_and_p is not None]
    total_price = sum([price for _, price in achivements_and_price_by_rollout])
    achievements_probs = {k: len([achievements for achievements, _ in achivements_and_price_by_rollout if k in achievements]) / len(achivements_and_price_by_rollout) for k in set([achievement for achievements, _ in achivements_and_price_by_rollout for achievement in achievements])}
    return hafner_scoring_function_modified(achievements_probs), achievements_probs, total_price

if __name__ == "__main__":
    #gpt-4o-2024-08-06
    #gpt-4o-mini-2024-07-18
    model_name = "ft:gpt-4o-mini-2024-07-18:basis:fbc-full:9yQ7idkM"#"ft:gpt-4o-mini-2024-07-18:basis:fbc-0:9yMCGTnx"
    mode = "craftax_classic"
    hafner_score,achievement_probs, total_price = asyncio.run(
        score_rollouts(k_steps=300, n_rollouts=5, base_seed=1000, verbose=False, model_name=model_name,modality="text", mode =mode)#hermes-3-llama-3.1-405b-fp8-128k
    )
    print("Agent got a normalized Crafter score of", hafner_score, "on mode: ",mode)
    print("Achievement Probabilities:")
    for k, v in achievement_probs.items():
        print(f"{k}: {v:.3f}")
    print(f"Total price of experiment: ${total_price:.2f}")#TODO: add a way to price images


    # gpt-4o-mini-2024-07-18 @ classic - Aug 19, 2024 @ base seed 0
    # Achievements ['Collect Wood', 'Place Table']
    # Achievements ['Collect Wood', 'Collect Drink']
    # Achievements ['Collect Wood', 'Place Table', 'Collect Sapling', 'Make Wood Pickaxe', 'Collect Stone', 'Collect Coal']
    # Achievements ['Collect Wood', 'Place Table', 'Collect Drink']
    # Achievements ['Collect Wood', 'Place Table', 'Make Wood Pickaxe', 'Collect Stone', 'Collect Coal']
    # Agent got a normalized Crafter score of 2.8091443487408707/15.2492
    # Achievement Probabilities:
    # Collect Coal: 0.400
    # Make Wood Pickaxe: 0.400
    # Collect Stone: 0.400
    # Collect Sapling: 0.200
    # Place Table: 0.800
    # Collect Drink: 0.400
    # Collect Wood: 1.000


    # gpt-4o-mini-2024-07-18 @ classic - Aug 19, 2024 @ base seed 20 5 x 300
    # ['Collect Wood', 'Place Table', 'Make Wood Pickaxe', 'Collect Stone']
    # ['Collect Wood', 'Place Table', 'Collect Drink']
    # ['Collect Wood', 'Place Table', 'Collect Drink', 'Make Wood Pickaxe', 'Collect Stone']
    # ['Collect Wood', 'Place Table', 'Collect Drink', 'Make Wood Pickaxe']
    # ['Collect Wood', 'Place Table', 'Eat Cow', 'Make Wood Pickaxe']
    # Agent got a normalized Crafter score of 2.962878448682913 on mode:  craftax_classic
    # Achievement Probabilities:
    # Place Table: 1.000
    # Collect Stone: 0.400
    # Eat Cow: 0.200
    # Collect Wood: 1.000
    # Collect Drink: 0.600
    # Make Wood Pickaxe: 0.800

    # ft:gpt-4o-mini-2024-07-18:basis:fbc-0:9yMCGTnx @ classic - Aug 19, 2024 @ base seed 1000 5 x 300
    # FT on 100 filtered trajectories, 5 steps before reward (600 messages total)
    # Agent got a normalized Crafter score of 2.9628784486829125 on mode:  craftax_classic
    # Place Table: 1.000
    # Collect Wood: 1.000
    # Make Wood Pickaxe: 0.800
    # Collect Drink: 0.400
    # Collect Coal: 0.200
    # Collect Stone: 0.600

    # ft:gpt-4o-mini-2024-07-18:basis:fbc-full:9yQ7idkM @ classic - Aug 19, 2024 @ base seed 20 5 x 300

    # deepseek-chat @ classic - Aug 19, 2024
    # Agent got a normalized Crafter score of 2.662545258677573/15.2492
    # Achievement Probabilities:
    # Collect Stone: 0.200
    # Collect Drink: 0.800
    # Make Stone Pickaxe: 0.200
    # Wake Up: 0.600
    # Place Table: 0.200
    # Collect Wood: 1.000
    # Make Wood Pickaxe: 0.200
    # Collect Sapling: 0.200

    # gpt-4o-mini-2024-07-18 @ full - Aug 19, 2024
    # Agent got a normalized Crafter score of 1.3046503720793805 / 45.747
    # Achievement Probabilities:
    # collect_wood: 0.600
    # place_table: 0.200
    # collect_drink: 0.600
    # collect_sapling: 0.200

    # Agent got a normalized Crafter score of 1.6943671232194055
    # Achievement Probabilities:
    # collect_sapling: 0.200
    # collect_drink: 0.400
    # place_table: 0.800
    # collect_wood: 0.800

    # flash @ classic - Aug 19, 2024