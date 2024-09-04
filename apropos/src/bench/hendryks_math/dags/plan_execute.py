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


def hendryks_math_plan_execute_example(model_names=["gpt-3.5-turbo", "gpt-3.5-turbo"]):
    plan = PromptTemplate(
        name="Plan Solution",
        system=SystemMessage(
            premise=[
                Topic(
                    topic_name="premise",
                    topic_template="# Premise\n$INTRODUCTION\n$MAIN_INSTRUCTIONS",
                    instructions_fields={
                        "$INTRODUCTION": "You are an AI assisting a colleague in completing a mathematics problem",
                        "$MAIN_INSTRUCTIONS": "You will be given a mathematics problem statement. Your task is to create a detailed plan to solve the problem, breaking it down into clear, logical steps. Focus on identifying key information, outlining necessary calculations, and providing a structured approach to reach the final answer. Be sure to consider various mathematical domains and techniques that may be applicable to the problem.",
                    },
                    input_fields=[],
                )
            ],
            objective=[
                Topic(
                    topic_name="objective",
                    topic_template="# Objective\n$OBJECTIVE",
                    instructions_fields={
                        "$OBJECTIVE": "Please provide a detailed, step-by-step plan to solve the given mathematics problem. Include all necessary calculations, formulas, and intermediate steps. Ensure that your plan covers all aspects of the problem, including given information, required calculations, and the final answer format. Focus on identifying patterns and making generalizations that can be applied to solve the problem efficiently."
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
                        "$ENJOINDER": "Your plan should include:\n1. A clear statement of the given information and problem to be solved\n2. Identification of relevant mathematical concepts and techniques\n3. Definition of variables and known relationships\n4. A step-by-step approach to solving the problem, including any formulas or calculations needed\n5. Explanation of the reasoning behind each step\n6. Verification of the solution and consideration of alternative approaches\n7. Description of how to present the final answer\n\nPlease provide your detailed plan below:",
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
                        "$INTRODUCTION": "You are an AI mathematical problem-solving assistant with expertise in various mathematical fields.",
                        "$MAIN_INSTRUCTIONS": "You will be given the problem statement together with a high-level plan to solve it. Your task is to implement this plan, verifying its correctness at each step.",
                    },
                    input_fields=[],
                )
            ],
            objective=[
                Topic(
                    topic_name="objective",
                    topic_template="# Objective\n$OBJECTIVE",
                    instructions_fields={
                        "$OBJECTIVE": "Solve the math problem by carefully following the provided plan. Leave your answer at the very end of your response in the format \\boxed\\{YOUR_ANSWER\\}."
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
                    topic_template="# Plan\n<<<PLAN>>>\n\n $ENJOINDER",
                    instructions_fields={
                        "$ENJOINDER": "Your detailed solution (including all calculations, explanations, and any necessary plan adjustments): "
                    },
                    input_fields=["<<<PLAN>>>"],
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
