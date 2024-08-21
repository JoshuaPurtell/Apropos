import asyncio

from apropos.src.core.programs.convenience_functions.dag_constructors import (
    build_path_program,
)
from apropos.src.core.programs.prompt import (
    PromptTemplate,
    SystemMessage,
    Topic,
    UserMessage,
)


def gsm8k_plan_execute_example(model_names=["gpt-3.5-turbo", "gpt-3.5-turbo"]):
    plan = PromptTemplate(
        name="Plan Solution",
        system=SystemMessage(
            premise=[
                Topic(
                    topic_name="premise",
                    topic_template="# Premise\n$INTRODUCTION\n$MAIN_INSTRUCTIONS",
                    instructions_fields={
                        "$INTRODUCTION": "You are an AI assisting a colleague in completing a mathematics problem",
                        "$MAIN_INSTRUCTIONS": "You will be given the problem statement.",
                    },
                    input_fields=[],
                )
            ],
            objective=[
                Topic(
                    topic_name="objective",
                    topic_template="# Objective\n$OBJECTIVE",
                    instructions_fields={
                        "$OBJECTIVE": "Please provide a bulleted list of steps to solve the problem. Keep your steps high-level - you are not to attempt to solve the problem or subproblems therein - a later, more capable stage will handle that. However, the later stage will not be given the question, so you must provide all necessary details and information for completing the task, including initial conditions, assumptions, and any other relevant information."
                    },
                    input_fields=[],
                )
            ],
            constraints=[],
        ),
        user=UserMessage(
            user=[
                Topic(
                    topic_name="user",
                    topic_template="# Math Problem\n<<<MATHEMATICS_QUESTION>>>\n\n $ENJOINDER",
                    instructions_fields={
                        "$ENJOINDER": "Your plan is:",
                    },
                    input_fields=["<<<MATHEMATICS_QUESTION>>>"],
                )
            ]
        ),
        response_type="str",
        response_model_scheme=None,
        demonstrations=[],
    )
    execute = PromptTemplate(
        name="Solve Problem",
        system=SystemMessage(
            premise=[
                Topic(
                    topic_name="premise",
                    topic_template="# Premise\n$INTRODUCTION\n$MAIN_INSTRUCTIONS",
                    instructions_fields={
                        "$INTRODUCTION": "You are an AI tasked with solving a mathematics problem",
                        "$MAIN_INSTRUCTIONS": "You will be given the problem statement together with a high-level plan to solve it.",
                    },
                    input_fields=[],
                )
            ],
            objective=[
                Topic(
                    topic_name="objective",
                    topic_template="# Objective\n$OBJECTIVE",
                    instructions_fields={
                        "$OBJECTIVE": "Solve the math problem by carefully following the provided plan. Leave your answer as an integer at the very end of your response."
                    },
                    input_fields=[],
                )
            ],
            constraints=[],
        ),
        user=UserMessage(
            user=[
                Topic(
                    topic_name="user",
                    topic_template="# Plan\n<<<PLAN>>>\n\n $ENJOINDER",  ## Math Question \n<<<MATHEMATICS_QUESTION>>>\n
                    instructions_fields={
                        "$ENJOINDER": "Your solution: ",
                    },
                    input_fields=["<<<PLAN>>>"],  # "<<<MATHEMATICS_QUESTION>>>",
                )
            ]
        ),
        response_type="str",
        response_model_scheme=None,
        demonstrations=[],
    )
    math_plan_execute_path_dag = asyncio.run(
        build_path_program(
            [plan, execute],
            model_names=model_names,
            dag_input_names=["<<<MATHEMATICS_QUESTION>>>"],
            dag_input_aliases={"question": "<<<MATHEMATICS_QUESTION>>>"},
            dag_output_aliases={"<<<FINAL_ANSWER>>>": "answer"},
        )
    )
    return math_plan_execute_path_dag
