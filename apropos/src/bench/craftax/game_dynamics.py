from typing import Literal
from craftaxlm import CraftaxACI, CraftaxClassicACI


async def harness(
    agent,
    k_steps,
    seed=0,
    verbose=False,
    mode: Literal["craftax_classic", "craftax_full"] = "craftax_classic",
):
    if mode == "craftax_classic":
        aci = CraftaxClassicACI(seed=seed, verbose=verbose)
    elif mode == "craftax_full":
        aci = CraftaxACI(seed=seed, verbose=verbose)
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
    pass
