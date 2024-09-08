from typing import Dict, List, Literal

from pydantic import BaseModel

from apropos.src.bench.craftax.game_information import (
    craftax_classic_action_dict,
    craftax_classic_game_rules,
    craftax_full_action_dict,
    craftax_full_game_rules,
    craftax_game_tips,
)
from apropos.src.bench.crafter.game_information import (
    crafter_game_tips,
)
from apropos.src.core.lms.cost import CostMonitor
from apropos.src.core.lms.helpers import LLM


class ReAct(BaseModel):
    reasoning: str
    actions: List[str]


class SimpleReActLanguageAgent:
    obs_history: List[Dict]
    react_history: List[Dict]
    lm: LLM
    cost_monitor: CostMonitor

    def __init__(
        self,
        lm: LLM,
        cost_monitor: CostMonitor,
        mode: Literal["craftax_classic", "craftax_full"] = "craftax_classic",
    ):
        self.obs_history = []
        self.react_history = []
        self.lm = lm
        self.cost_monitor = cost_monitor
        self.mode = mode

    def render_history(self):
        def remove_empty_error_section(d):
            if "errors" in d and not len(d["errors"]["illegal_actions"]) > 0:
                d.pop("errors")
            return d

        react_history = [
            f"<{i} reasoning step(s) in the past>{remove_empty_error_section(item)}</{i} reasoning step(s) in the past>"
            for i, item in enumerate(reversed(self.react_history[-5:]), 1)
        ]
        obs_history = [
            f"<{i} environment step(s) in the past>{remove_empty_error_section(item)}</{i} environment step(s) in the past>"
            for i, item in enumerate(reversed(self.obs_history[-5:]), 1)
        ]
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
            system_prompt=system_message,
            user_prompt=user_message,
            response_model=ReAct,
        )

        illegal_actions = [
            action
            for action in react_step.actions
            if action not in craftax_classic_action_dict.keys()
        ]
        legal_actions = [
            action
            for action in react_step.actions
            if action in craftax_classic_action_dict.keys()
        ]
        react_info = react_step.dict()
        react_info["errors"] = {
            "illegal_actions": illegal_actions,
        }

        self.react_history.append(react_info)
        self.cost_monitor.update_token_counts(
            system_message, user_message, str(react_step.dict())
        )

        return legal_actions

    async def add_observations(self, observations: List[Dict]):
        for observation in observations:
            self.obs_history.append(observation["state"])  # [0].default_readout()
