import asyncio
from typing import Dict, List

import numpy as np
from pydantic import BaseModel

from apropos.bench.crafter.game_dynamics import harness
from apropos.bench.crafter.game_information import (
    action_dict,
    crafter_game_rules,
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

    def __init__(self, lm: LLM, cost_monitor: CostMonitor):
        self.obs_history = []
        self.react_history = []
        self.lm = lm
        self.cost_monitor = cost_monitor

    def render_history(self):
        react_history = [f"<{i} reasoning step(s) in the past>{item}</{i} reasoning step(s) in the past>" for i, item in enumerate(reversed(self.react_history[-5:]), 1)]
        obs_history = [f"<{i} environment step(s) in the past>{item}</{i} environment step(s) in the past>" for i, item in enumerate(reversed(self.obs_history[-5:]), 1)]
        return "\n".join(react_history), "\n".join(obs_history)

    async def get_actions(self):
        system_message = f"""
# Premise
You're playing the game of Crafter.
Here is some information about this setting
<Crafter Information>
<Rules>
{crafter_game_rules}
</Rules>
<Tips>
{crafter_game_tips}
</Tips>
<Actions Available>
{list(action_dict.values())}
</Actions Available>
You'll be given your past actions/thoughts, along with recent raw observations from the environment

# Objective
Please choose 4-7 actions to take next in the game, and provide a thought justifying the long-term end you'd like to achieve with these actions.

# Constraints
- If your previous actions worked toward a valuable goal, continue in that direction unless you have a good reason to change course.
- If there's no important thought to share, don't share anything. "No update" is a valid and the most common response.
- Render your actions by name. All rendered names must match the names in the "Actions Available" section.
- If you attempt an action that requires certain conditions to be met without meeting those conditions, the action will default to a no-op.
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
        
        illegal_actions = [action for action in react_step.actions if action not in action_dict.values()]
        legal_actions = [action for action in react_step.actions if action in action_dict.values()]
        react_info = react_step.dict()
        react_info["errors"] = {
            "illegal_actions": illegal_actions,
        }
        self.react_history.append(react_info)
        self.cost_monitor.update_token_counts(system_message, user_message, str(react_step.dict()))
        return [{v:k for k,v in action_dict.items()}[action_name] for action_name in legal_actions]

    async def add_observations(self, observations: List[Dict]):
        for observation in observations:
            self.obs_history.append(observation["text_obs"][0].default_readout())

def hafner_scoring_function_modified(achievements_probs: Dict):
    probs = list(achievements_probs.values())
    return sum([np.log(1 + prob) for prob in probs])

# Log token count to get approx cost of rollout.
async def score_rollouts(k_steps=10, n_rollouts=10, verbose=False, model_name="gpt-4o-mini"):
    core_lm = LLM(model_name)
    cost_monitor = CostMonitor(
        model_name=core_lm.model_name
    )
    agent = SimpleReActLanguageAgent(lm=core_lm, cost_monitor=cost_monitor)
    async def single_rollout():
        try:
            achievements_log = await harness(agent, k_steps=k_steps, verbose=verbose)
        except Exception as e:
            print(f"Error in Rollout: {e}")
            return
        achievements = list({k: v for k, v in achievements_log.items() if v > 0}.keys())
        action_density = len(agent.obs_history) / len(agent.react_history)
        price, n_tokens = cost_monitor.final_cost()
        if verbose and action_density < 2:
            print(f"Warning - Low Action Density of {action_density}")
        if verbose:
            print(f"Rollout price ${price:.2f} for {n_tokens} tokens")
        return achievements, price
    achivements_and_price_by_rollout = await asyncio.gather(*[single_rollout() for _ in range(n_rollouts)])
    achivements_and_price_by_rollout = [a_and_p for a_and_p in achivements_and_price_by_rollout if a_and_p is not None]
    total_price = sum([price for _, price in achivements_and_price_by_rollout])
    achievements_probs = {k: len([achievements for achievements, _ in achivements_and_price_by_rollout if k in achievements]) / len(achivements_and_price_by_rollout) for k in set([achievement for achievements, _ in achivements_and_price_by_rollout for achievement in achievements])}
    return hafner_scoring_function_modified(achievements_probs), achievements_probs, total_price

if __name__ == "__main__":
    hafner_score,achievement_probs, total_price = asyncio.run(
        score_rollouts(k_steps=300, n_rollouts=10, verbose=True, model_name="gemini-1.5-flash")#hermes-3-llama-3.1-405b-fp8-128k
    )
    print("Agent got a normalized Crafter score of", hafner_score)
    print("Achievement Probabilities:")
    for k, v in achievement_probs.items():
        print(f"{k}: {v:.3f}")
    print(f"Total price of experiment: ${total_price:.2f}")