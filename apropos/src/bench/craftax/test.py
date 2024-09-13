import asyncio
from typing import Dict, Literal

import numpy as np

from apropos.src.bench.craftax.game_dynamics import harness
from apropos.src.bench.craftax.react import SimpleReActLanguageAgent
from apropos.src.core.lms.cost import CostMonitor
from apropos.src.core.lms.helpers import LLM


def hafner_scoring_function_modified(achievements_probs: Dict):
    probs = list(achievements_probs.values())
    return sum([np.log(1 + prob) for prob in probs])


# Log token count to get approx cost of rollout.
async def score_rollouts(
    k_steps=10,
    base_seed=0,
    n_rollouts=10,
    verbose=False,
    model_name="gpt-4o-mini",
    modality: Literal["text", "vision"] = "text",
    mode: Literal["craftax_classic", "craftax_full"] = "craftax_classic",
):
    core_lm = LLM(model_name)
    cost_monitor = CostMonitor(model_name=core_lm.model_name)
    if modality == "text":
        agent = SimpleReActLanguageAgent(
            lm=core_lm, cost_monitor=cost_monitor, mode=mode
        )
    elif modality == "vision":
        raise NotImplementedError("Vision not implemented yet")

    async def single_rollout(seed):
        achievements_log = await harness(
            agent, seed=base_seed + seed, k_steps=k_steps, verbose=verbose, mode=mode
        )
        achievements = list({k: v for k, v in achievements_log.items() if v > 0}.keys())
        print("Achievements", achievements)
        action_density = len(agent.obs_history) / len(agent.react_history)
        price, n_tokens = cost_monitor.final_cost()
        if verbose and action_density < 2:
            print(f"Warning - Low Action Density of {action_density}")
        if verbose:
            print(f"Rollout price ${price:.2f} for {n_tokens} tokens")
        return achievements, price

    achivements_and_price_by_rollout = await asyncio.gather(
        *[single_rollout(seed=_) for _ in range(n_rollouts)]
    )
    achivements_and_price_by_rollout = [
        a_and_p for a_and_p in achivements_and_price_by_rollout if a_and_p is not None
    ]
    total_price = sum([price for _, price in achivements_and_price_by_rollout])
    achievements_probs = {
        k: len(
            [
                achievements
                for achievements, _ in achivements_and_price_by_rollout
                if k in achievements
            ]
        )
        / len(achivements_and_price_by_rollout)
        for k in set(
            [
                achievement
                for achievements, _ in achivements_and_price_by_rollout
                for achievement in achievements
            ]
        )
    }
    return (
        hafner_scoring_function_modified(achievements_probs),
        achievements_probs,
        total_price,
    )


if __name__ == "__main__":
    # gpt-4o-2024-08-06
    # gpt-4o-mini-2024-07-18
    # meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
    model_name = "o1-mini"  # "ft:gpt-4o-mini-2024-07-18:basis:fbc-0:9yMCGTnx"
    mode = "craftax_classic"
    hafner_score, achievement_probs, total_price = asyncio.run(
        score_rollouts(
            k_steps=300,
            n_rollouts=1,
            base_seed=1000,
            verbose=False,
            model_name=model_name,
            modality="text",
            mode=mode,
        )  # hermes-3-llama-3.1-405b-fp8-128k
    )
    print("Agent got a normalized Crafter score of", hafner_score, "on mode: ", mode)
    print("Achievement Probabilities:")
    for k, v in achievement_probs.items():
        print(f"{k}: {v:.3f}")
    print(
        f"Total price of experiment: ${total_price:.2f}"
    )  # TODO: add a way to price images

    # Info for FT experiments
    # Agent got a normalized Crafter score of 2.824573777305732 on mode:  craftax_classic
    # Achievement Probabilities:
    # Collect Wood: 1.000
    # Make Wood Pickaxe: 0.400
    # Collect Drink: 0.900
    # Place Plant: 0.100
    # Place Table: 1.000
    # Collect Stone: 0.200
    # Collect Sapling: 0.200

    # claude-3-5-sonnet-20240620 @ classic - Aug 19, 2024 @ base seed 0
    # Josh note - a bit shocked, it typically does much better than gpt-4o. Going to use discretion and not add to the leaderboard.
    # Agent got a normalized Crafter score of 3.7285323200296063 on mode:  craftax_classic
    # Achievement Probabilities:
    # Make Wood Sword: 0.200
    # Make Wood Pickaxe: 0.600
    # Wake Up: 0.600
    # Collect Coal: 0.400
    # Collect Stone: 0.400
    # Place Table: 0.800
    # Collect Wood: 1.000
    # Collect Drink: 0.600
    # Make Stone Pickaxe: 0.200

    # gpt-4o-2024-08-06 @ classic - Aug 19, 2024 @ base seed 0
    # Achievements ['Collect Wood', 'Place Table', 'Collect Drink', 'Make Wood Pickaxe', 'Make Wood Sword', 'Collect Stone', 'Wake Up', 'Place Furnace', 'Collect Coal']
    # Achievements ['Collect Wood', 'Place Table', 'Collect Drink', 'Make Wood Sword', 'Wake Up']
    # Achievements ['Collect Wood', 'Place Table', 'Collect Drink', 'Make Wood Pickaxe', 'Collect Stone', 'Wake Up', 'Place Furnace', 'Collect Coal']
    # Achievements ['Collect Wood', 'Wake Up']
    # Achievements ['Collect Wood', 'Place Table', 'Make Wood Pickaxe', 'Make Wood Sword', 'Collect Stone', 'Wake Up']
    # Agent got a normalized Crafter score of 4.527040016247378/15.2492 on mode:  craftax_classic
    # Achievement Probabilities:
    # Collect Coal: 0.400
    # Collect Wood: 1.000
    # Make Wood Sword: 0.600
    # Make Wood Pickaxe: 0.600
    # Collect Drink: 0.600
    # Place Furnace: 0.400
    # Collect Stone: 0.600
    # Place Table: 0.800
    # Wake Up: 1.000

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

    # flash 8b 0.988/ 45.747
    # 0.6523
    # Collect Wood: 0.600
    # Collect Drink: 0.200

    # llama 3 8b 0.77010 / 45.747
