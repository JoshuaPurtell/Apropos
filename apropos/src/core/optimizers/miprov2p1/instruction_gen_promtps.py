from typing import Dict, List

from pydantic import BaseModel

from apropos.src.core.programs.prompt import (
    PromptTemplate,
    SystemMessage,
    Topic,
    UserMessage,
)


class QuestionCurationResponse(BaseModel):
    bootstrapping_question_indices: List[int]
    train_question_indices: List[int]
    val_question_indices: List[int]


# We want to have train exemplify problems, val provide clear signal, and bootstrap demos to be "core"
question_curation_prompt = PromptTemplate(
    name="Question Curation",
    system=SystemMessage(
        premise=[
            Topic(
                topic_name="Premise",
                topic_template="# Premise\nYour task is to sort a list of questions into three groups: Bootstrapping Questions, Training Questions, and Validation Questions. You will be provided with information about an AI system's performance on these questions.",
                instructions_fields={},
                input_fields=[],
            )
        ],
        objective=[
            Topic(
                topic_name="Objective",
                topic_template="# Objective\nYou will be given the following information pertaining to an existing AI system:\n - A list of questions the AI system sometimes succeeds on / sometimes fails on, with respective success rates\n - For each such question, a sample of successful attempts/traces\n - For each such question, a sample of unsuccessful attempts/traces\nYour task is to sort these questions into three groups:\n - Bootstrapping Questions: these are questions where the successful attempts have reasoning of the highest quality, and so will be highly effective demonstrations of how to complete tasks like the questions shown\n - Training Questions: these are questions where the failed attempts best exemplify common failure modes / patterns, and so will be highly effective for training the AI system to avoid these failure modes\n - Validation Questions: these are questions that are representative of all learnable questions that you've been shown, with a preference for questions that have low success rates",
                instructions_fields={},
                input_fields=[],
            )
        ],
        constraints=[
            Topic(
                topic_name="Constraints",
                topic_template="# Constraints\nIf there are very few questions available, ensure that sufficiently many (10+ each) are used for demonstrations and training, however additional questions are likely best used for validation. For instance, if given 50 questions, use 10 for demonstrations, 10 for training, and 30 for validation. If given 200 questions, use 15 for demonstrations, 85 for training, and 100 for validation. No matter what, there must be at least 5 questions in each group. Ensure all indices are valid",
                instructions_fields={},
                input_fields=[],
            )
        ],
    ),
    user=UserMessage(
        user=[
            Topic(
                topic_name="User Input",
                topic_template="""# Input Data
## Questions with varying success rates
<<<HIGHLY_LEARNABLE_QUESTIONS>>>

## Success rates for each question
<<<SUCCESS_RATES>>>

## Samples of successful attempts
<<<SUCCESS_TRACES>>>

## Samples of unsuccessful attempts
<<<FAILURE_TRACES>>>
""",
                instructions_fields={},
                input_fields=[
                    "<<<HIGHLY_LEARNABLE_QUESTIONS>>>",
                    "<<<SUCCESS_RATES>>>",
                    "<<<SUCCESS_TRACES>>>",
                    "<<<FAILURE_TRACES>>>",
                ],
            )
        ]
    ),
    response_type="pydantic",
    response_model_scheme=QuestionCurationResponse.schema(),
    demonstrations=[],
)


class ProblemsResponse(BaseModel):
    problem_names: List[str]
    problem_descriptions: List[str]
    # failure_modes: List[str]
    nodes_responsible_by_name: List[List[str]]
    example_question_indices: List[List[int]]


# Keep it simple for now
# identify_problems_dag = None
identify_problems_prompt = PromptTemplate(
    name="Identifying Problems",
    system=SystemMessage(
        premise=[
            Topic(
                topic_name="Premise",
                topic_template="# Premise\nYou will be given the following information pertaining to an existing AI system:\n - A list of questions the AI system sometimes succeeds on / sometimes fails on, with respective success rates\n - For each such question, a sample of successful attempts/traces\n - For each such question, a sample of unsuccessful attempts/traces\nYour task is to identify the specific reasons why the AI system fails on these questions. For each failure mode, provide a name, description, associated nodes, and list of example where the AI's failures exhibit the failure mode.\n",
                instructions_fields={},
                input_fields=[],
            )
        ],
        objective=[
            Topic(
                topic_name="Objective",
                topic_template="# Objective\nPlease identify the specific reasons why the AI system fails on these questions, and suggest targeted interventions to improve its performance on these questions.\n",
                instructions_fields={},
                input_fields=[],
            )
        ],
        constraints=[
            Topic(
                topic_name="Constraints",
                topic_template="# Constraints\n - $SPECIFIC\n - $IMPACTFUL",
                instructions_fields={
                    "$SPECIFIC": "Ensure that each failure mode description is specific and unambiguous. When reviewing any failed trace, it should be crystal clear whether or not it exhibits the failure mode.",
                    "$IMPACTFUL": "Ensure that each failure mode could plausibly contribute to poor performance on many other questions. Failure modes pertaining to very specific subtopics or edge cases are not impactful.",
                },
                input_fields=[],
            )
        ],
    ),
    user=UserMessage(
        user=[
            Topic(
                topic_name="User Input",
                topic_template="""# Input Data
## Questions with varying success rates
<<<HIGHLY_LEARNABLE_QUESTIONS>>>

## Success rates for each question
<<<SUCCESS_RATES>>>

## Sample of successful attempts
<<<SUCCESS_TRACES>>>

## Sample of unsuccessful attempts
<<<FAILURE_TRACES>>>

## Nodes Involved
<<<NODE_NAMES>>>
""",
                instructions_fields={},
                input_fields=[
                    "<<<HIGHLY_LEARNABLE_QUESTIONS>>>",
                    "<<<SUCCESS_RATES>>>",
                    "<<<SUCCESS_TRACES>>>",
                    "<<<FAILURE_TRACES>>>",
                    "<<<NODE_NAMES>>>",
                ],
            )
        ]
    ),
    response_type="pydantic",
    response_model_scheme=ProblemsResponse.schema(),
    demonstrations=[],
)

# Have it sample multiple solutions, then filter out the best


class InterventionConceptsResponse(BaseModel):
    intervention_concepts_by_node: Dict[str, List[str]]


get_intervention_concepts_prompt = PromptTemplate(
    name="Get Intervention Concepts",
    system=SystemMessage(
        premise=[
            Topic(
                topic_name="Premise",
                topic_template="# Premise\nYou will be given the following information pertaining to an existing AI system:\n - A description of a problem the AI system is experiencing\n - A description of the failure mode the AI system is experiencing\n - A list of example questions that the AI system sometimes succeeds on / sometimes fails on\n - The nodes responsible for handling these questions",
                instructions_fields={},
                input_fields=[],
            )
        ],
        objective=[
            Topic(
                topic_name="Objective",
                topic_template="# Objective\nPlease identify possible hints, reminders or nudges that could help the AI system avoid the failure mode it is experiencing. These targeted interventions should be concise and to the point, and should be written in a way that is easy for a language model to understand and incorporate into its future responses.\n",
                instructions_fields={},
                input_fields=[],
            )
        ],
        constraints=[
            Topic(
                topic_name="Constraints",
                topic_template="# Constraints\nKeep each intervention very specific, up to 2 sentences each, and emphasize diversity in your generations",
                instructions_fields={},
                input_fields=[],
            )
        ],
    ),
    user=UserMessage(
        user=[
            Topic(
                topic_name="User Input",
                topic_template="""# Input Data
## Problem Description
<<<FAILURE_DESCRIPTION>>>

## Failure Mode
<<<FAILURE_MODE>>>

## Example Questions
<<<EXAMPLE_QUESTIONS>>>

## Nodes Responsible
<<<NODES_RESPONSIBLE>>>
""",
                instructions_fields={},
                input_fields=[
                    "<<<FAILURE_DESCRIPTION>>>",
                    "<<<FAILURE_MODE>>>",
                    "<<<EXAMPLE_QUESTIONS>>>",
                    "<<<NODES_RESPONSIBLE>>>",
                ],
            )
        ]
    ),
    response_type="pydantic",
    response_model_scheme=InterventionConceptsResponse.schema(),
    demonstrations=[],
)


class PromptDeltasResponse(BaseModel):
    node_names: List[str]
    descriptions: List[str]
    messages: List[str]  # Literal["system", "user"]
    subcomponents: List[str]  # Literal["premise", "objective", "constraints","user"]
    topic_names: List[str]
    instructions_fields_edits: List[Dict[str, str]]


get_simple_prompt_delta_prompt = PromptTemplate(
    name="Get Simple Prompt Delta",
    system=SystemMessage(
        premise=[
            Topic(
                topic_name="Premise",
                topic_template="# Premise\nYou will be given the following information pertaining to an existing AI system:\n$INFO \n$TASK_HIGH_LEVEL\n$PROGRAM_STRUCTURE_HINTS\n\n<<<VALID_TOPICS_BY_SUBCOMPONENT_BY_NODE>>>",
                instructions_fields={
                    "$INFO": "\n - A description of the current program \n - Traces where the AI system succeeded and failed\n",
                    "$TASK_HIGH_LEVEL": "Your task is to suggest a specific prompt delta for the AI system to improve its performance on the failure mode.",
                    "$PROGRAM_STRUCTURE_HINTS": "Every program is organized into nodes, some of which are shown here. Each node has two messages - system and user. System message is comprised of 3 subcomponents: premise, objective, constraints. The user message only has a 'user' subcomponent. Each subcomponent (premise, objective, constraints, user) is a list of topics. Each topic has a name, a template, instructions fields (denoted with $ALL_CAPS) and input fields (denoted with <<ALL_CAPS>>). Both instructions fields and input fields are interpolated into the topic template to generate the final message. Input fields are piped in automatically - don't change them.",
                },
                input_fields=["<<<VALID_TOPICS_BY_SUBCOMPONENT_BY_NODE>>>"],
            )
        ],
        objective=[
            Topic(
                topic_name="Objective",
                topic_template="# Objective\nPlease suggest a specific prompt delta for the AI system to improve its performance on the failure mode.",
                instructions_fields={},
                input_fields=[],
            )
        ],
        constraints=[
            Topic(
                topic_name="Constraints",
                topic_template="# Constraints\n - $SPECIFIC\n - $LITERALS\n - $EDITS\n - $TOPIC_NAMES\n - $SUBCOMPONENT_TOPIC_COMPATABILITY",
                instructions_fields={
                    "$SPECIFIC": "Ensure that the prompt delta is specific to the failure mode, and is not too vague or general.",
                    "$LITERALS": "Values for 'subcomponent' should be one of 'premise', 'objective', 'constraints', or 'user' and values for 'message' should be one of 'system', 'user'.",
                    "$EDITS": "All edits should be substitutions for instructions fields that already exist in the program. Keep the key, replace the value.",
                    "$TOPIC_NAMES": "The topic names *must* refer to the corresponding topic names in the current program for which the edits will be applied.",
                    "$SUBCOMPONENT_TOPIC_COMPATABILITY": "For each index, ensure the topic name provided is indeed present as a topic under the corresponding subcomponent.",
                    "$INFORMATION_TO_RETURN": "For each index, return a description for the prompt delta, the node name, the subcomponent, the topic name, the message, and the instructions fields for the prompt delta.",
                },
                input_fields=[],
            )
        ],
    ),
    user=UserMessage(
        user=[
            Topic(
                topic_name="User Input",
                topic_template="""# Input Data
## Current Program
<<<CURRENT_PROGRAM>>>

## Success Traces
<<<SUCCESS_TRACES>>>

## Failure Traces
<<<FAILURE_TRACES>>>
""",
                instructions_fields={},
                input_fields=[
                    "<<<CURRENT_PROGRAM>>>",
                    "<<<SUCCESS_TRACES>>>",
                    "<<<FAILURE_TRACES>>>",
                ],
            )
        ],
    ),
    response_type="pydantic",
    response_model_scheme=PromptDeltasResponse.schema(),
    demonstrations=[],
)

get_prompt_delta_prompt = PromptTemplate(
    name="Get Prompt Delta",
    system=SystemMessage(
        premise=[
            Topic(
                topic_name="Premise",
                topic_template="# Premise\nYou will be given the following information pertaining to an existing AI system:\n$INFO \n$TASK_HIGH_LEVEL\n$PROGRAM_STRUCTURE_HINTS\n\n<<<VALID_TOPICS_BY_SUBCOMPONENT_BY_NODE>>>",
                instructions_fields={
                    "$INFO": "\n - A description of the current program \n - A list of intervention concepts for a specific failure mode\n",
                    "$TASK_HIGH_LEVEL": "Your task is to suggest a specific prompt delta for the AI system to improve its performance on the failure mode.",
                    "$PROGRAM_STRUCTURE_HINTS": "Every program is organized into nodes, some of which are shown here. Each node has two messages - system and user. System message is comprised of 3 subcomponents: premise, objective, constraints. The user message only has a 'user' subcomponent. Each subcomponent (premise, objective, constraints, user) is a list of topics. Each topic has a name, a template, instructions fields (denoted with $ALL_CAPS) and input fields (denoted with <<ALL_CAPS>>). Both instructions fields and input fields are interpolated into the topic template to generate the final message. Input fields are piped in automatically - don't change them.",
                },
                input_fields=["<<<VALID_TOPICS_BY_SUBCOMPONENT_BY_NODE>>>"],
            )
        ],
        objective=[
            Topic(
                topic_name="Objective",
                topic_template="# Objective\nPlease suggest a specific prompt delta for the AI system to improve its performance on the failure mode.",
                instructions_fields={},
                input_fields=[],
            )
        ],
        constraints=[
            Topic(
                topic_name="Constraints",
                topic_template="# Constraints\n - $SPECIFIC\n - $LITERALS\n - $EDITS\n - $TOPIC_NAMES\n - $SUBCOMPONENT_TOPIC_COMPATABILITY",
                instructions_fields={
                    "$SPECIFIC": "Ensure that the prompt delta is specific to the failure mode, and is not too vague or general.",
                    "$LITERALS": "Values for 'subcomponent' should be one of 'premise', 'objective', 'constraints', or 'user' and values for 'message' should be one of 'system', 'user'.",
                    "$EDITS": "All edits should be substitutions for instructions fields that already exist in the program. Keep the key, replace the value.",
                    "$TOPIC_NAMES": "The topic names *must* refer to the corresponding topic names in the current program for which the edits will be applied.",
                    "$SUBCOMPONENT_TOPIC_COMPATABILITY": "For each index, ensure the topic name provided is indeed present as a topic under the corresponding subcomponent.",
                    "$INFORMATION_TO_RETURN": "For each index, return a description for the prompt delta, the node name, the subcomponent, the topic name, the message, and the instructions fields for the prompt delta.",
                },
                input_fields=[],
            )
        ],
    ),
    user=UserMessage(
        user=[
            Topic(
                topic_name="User Input",
                topic_template="""# Input Data
## Current Program
<<<CURRENT_PROGRAM>>>

## Intervention Concepts
<<<INTERVENTION_CONCEPTS>>>

## Problem Description
<<<FAILURE_DESCRIPTION>>>
""",
                instructions_fields={},
                input_fields=[
                    "<<<INTERVENTION_CONCEPTS>>>",
                    "<<<CURRENT_PROGRAM>>>",
                    "<<<FAILURE_DESCRIPTION>>>",
                ],
            )
        ],
    ),
    response_type="pydantic",
    response_model_scheme=PromptDeltasResponse.schema(),
    demonstrations=[],
)


class PromptDeltasVariationsResponse(BaseModel):
    descriptions: List[str]
    messages: List[str]
    subcomponents: List[str]
    topic_names: List[str]
    instructions_fields_edits: List[Dict[str, str]]


get_prompt_delta_variations_prompt = PromptTemplate(
    name="Get Prompt Delta Variations",
    system=SystemMessage(
        premise=[
            Topic(
                topic_name="Premise",
                topic_template="# Premise\nYou will be given the following information pertaining to an existing AI system: \n$INFO\n$TASK_HIGH_LEVEL\n$PROGRAM_STRUCTURE_HINTS",
                instructions_fields={
                    "$INFO": "\n - A description of the current program \n - A list of intervention concepts for a specific failure mode\n - A change to make to the program to address the failure mode",
                    "$TASK_HIGH_LEVEL": "Your task is to suggest a specific prompt delta for the AI system to improve its performance on the failure mode.",
                    "$PROGRAM_STRUCTURE_HINTS": "Every program is organized into nodes, some of which are shown here. Each node has two messages - system and user. System message is comprised of 3 subcomponents: premise, objective, constraints. The user message only has a 'user' subcomponent. Each subcomponent (premise, objective, constraints, user) is a list of topics. Each topic has a name, a template, instructions fields (denoted with $ALL_CAPS) and input fields (denoted with <<ALL_CAPS>>). Both instructions fields and input fields are interpolated into the topic template to generate the final message. Input fields are piped in automatically - don't change them.",
                },
                input_fields=[],
            )
        ],
        objective=[
            Topic(
                topic_name="Objective",
                topic_template="# Objective\nPlease suggest variations on the provided prompt delta to address the failure mode in similar but distinct ways.",
                instructions_fields={},
                input_fields=[],
            )
        ],
        constraints=[
            Topic(
                topic_name="Constraints",
                topic_template="# Constraints\nEnsure that the prompt delta variations are diverse and specific to the failure mode.\n $LITERALS\n $EDITS",
                instructions_fields={
                    "LITERALS": "Values for 'subcomponent' should be one of 'premise', 'objective', 'constraints', or 'user' and values for 'message' should be one of 'system', 'user'.",
                    "EDITS": "All edits should be substitutions for instructions fields that already exist in the program. Keep the key, replace the value.",
                    "TOPIC_NAMES": "The topic names *must* refer to the corresponding topic names in the current program for which the edits will be applied.",
                },
                input_fields=[],
            )
        ],
    ),
    user=UserMessage(
        user=[
            Topic(
                topic_name="User Input",
                topic_template="""# Input Data
## Core Delta
<<<CORE_DELTA>>>

## Current Program
<<<CURRENT_PROGRAM>>>

## Failure Mode
<<<FAILURE_MODE>>>
""",
                instructions_fields={},
                input_fields=[
                    "<<<FAILURE_MODE>>>",
                    "<<<CORE_DELTA>>>",
                    "<<<CURRENT_PROGRAM>>>",
                ],
            )
        ],
    ),
    response_type="pydantic",
    response_model_scheme=PromptDeltasVariationsResponse.schema(),
    demonstrations=[],
)


class DemosForProblemResponse(BaseModel):
    demo_indices: List[int]
    # demo_indices_by_problem_by_node: Dict[str, Dict[str, List[int]]]


get_apt_demos_prompt = PromptTemplate(
    name="Get Apt Demos",
    system=SystemMessage(
        premise=[
            Topic(
                topic_name="Premise",
                topic_template="# Premise\nYou will be given information about an AI system, including a description of the current program and a failure mode it is experiencing.",
                instructions_fields={},
                input_fields=[],
            )
        ],
        objective=[
            Topic(
                topic_name="Objective",
                topic_template="# Objective\nSelect the indices of demos that best represent the given problem and node.",
                instructions_fields={},
                input_fields=[],
            )
        ],
        constraints=[
            Topic(
                topic_name="Constraints",
                topic_template="# Constraints\nChoose diverse demos that specifically address the failure mode.",
                instructions_fields={},
                input_fields=[],
            )
        ],
    ),
    user=UserMessage(
        user=[
            Topic(
                topic_name="User Input",
                topic_template="""# Input Data
## Problem Description
<<<FAILURE_DESCRIPTION>>>

## Failure Mode
<<<FAILURE_MODE>>>

## Example Questions
<<<EXAMPLE_QUESTIONS>>>

## Nodes Responsible
<<<NODES_RESPONSIBLE>>>

## Bootstrapped Demos
<<<BOOTSTRAPPED_DEMOS>>>
""",
                instructions_fields={},
                input_fields=[
                    "<<<FAILURE_DESCRIPTION>>>",
                    "<<<FAILURE_MODE>>>",
                    "<<<EXAMPLE_QUESTIONS>>>",
                    "<<<NODES_RESPONSIBLE>>>",
                    "<<<BOOTSTRAPPED_DEMOS>>>",
                ],
            )
        ],
    ),
    response_type="pydantic",
    response_model_scheme=DemosForProblemResponse.schema(),
    demonstrations=[],
)

# Plan -> Search
# plan_search_dag = None

# Use the solution concepts to create the best interventions
# resolve_deltas_dag = None
