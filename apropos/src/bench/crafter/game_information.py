action_dict = {
    i: a
    for i, a in enumerate(
        [
            "Noop",
            "Move West",
            "Move East",
            "Move North",
            "Move South",
            "Do",
            "Sleep",
            "Place Stone",
            "Place Table",
            "Place Furnace",
            "Place Plant",
            "Make Wood Pickaxe",
            "Make Stone Pickaxe",
            "Make Iron Pickaxe",
            "Make Wood Sword",
            "Make Stone Sword",
            "Make Iron Sword",
        ]
    )
}

crafter_game_rules = {
    "World Information": {
        "Actions with Requirements": {
            "Movement": {
                "Move West": {"Requirements": "Flat ground west of the agent."},
                "Move East": {"Requirements": "Flat ground east of the agent."},
                "Move North": {"Requirements": "Flat ground north of the agent."},
                "Move South": {"Requirements": "Flat ground south of the agent."},
            },
            "Basic Actions": {
                "Do": {
                    "Requirements": "Facing creature or material; have necessary tool."
                },
                "Sleep": {"Requirements": "Energy level is below maximum."},
                "Noop": {"Requirements": "Always applicable. (Means no operation.)"},
            },
            "Crafting and Building": {
                "Place Stone": {"Requirements": "Stone in inventory."},
                "Place Table": {"Requirements": "Wood in inventory."},
                "Place Furnace": {"Requirements": "Stone in inventory."},
                "Place Plant": {"Requirements": "Sapling in inventory."},
                "Make Wood Pickaxe": {
                    "Requirements": "Nearby table; wood in inventory."
                },
                "Make Stone Pickaxe": {
                    "Requirements": "Nearby table; wood, stone in inventory."
                },
                "Make Iron Pickaxe": {
                    "Requirements": "Nearby table, furnace; wood, coal, iron in inventory."
                },
                "Make Wood Sword": {"Requirements": "Nearby table; wood in inventory."},
                "Make Stone Sword": {
                    "Requirements": "Nearby table; wood, stone in inventory."
                },
                "Make Iron Sword": {
                    "Requirements": "Nearby table, furnace; wood, coal, iron in inventory."
                },
            },
        },
        "Hierarchy of Goals": {
            "Collect Wood": {
                "Place Table": {
                    "Make Wood Pickaxe": {
                        "Collect Stone": {
                            "Place Stone": {},
                            "Make Stone Pickaxe": {
                                "Collect Iron": {
                                    "Make Iron Pickaxe": {
                                        "Collect Diamond": {},
                                        "Place Furnace": {
                                            "Make Iron Sword": {},
                                            "Make Iron Pickaxe": {},
                                        },
                                    }
                                },
                                "Collect Coal": {},
                            },
                        }
                    },
                    "Make Wood Sword": {},
                }
            },
            "Eat Cow": {},
            "Collect Sapling": {"Place Plant": {"Eat Plant": {}}},
            "Collect Drink": {},
            "Defeat Zombie": {},
            "Defeat Skeleton": {},
            "Wake Up": {},
        },
    }
}

crafter_game_tips = [
    "Don't take actions for which you don't have the necessary requirements. For example, don't try to place a table if you don't have wood in your inventory.",
    "Place safety and survival first, but when the coast is clear strive to make progress on unlocking the next goals in your progression.",
]
