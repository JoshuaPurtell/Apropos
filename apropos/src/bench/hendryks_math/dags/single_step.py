import asyncio

from apropos.src.core.programs.convenience_functions.dag_constructors import (
    build_single_step_program,
)
from apropos.src.core.programs.prompt import (
    PromptTemplate,
    SystemMessage,
    Topic,
    UserMessage,
)


def hendryks_math_single_step_example(model_name="gpt-3.5-turbo"):
    execute = PromptTemplate(
        name="Math Problem Solution",
        system=SystemMessage(
            premise=[],
            objective=[
                Topic(
                    topic_name="Objective",
                    topic_template="# Objective\nProvide a solution to the provided mathematics problem",
                    instructions_fields={},
                    input_fields=[],
                )
            ],
            constraints=[
                Topic(
                    topic_name="Constraints",
                    topic_template="# Constraints\n Leave your answer at the very end of your response in the format \\boxed\\{YOUR_ANSWER\\}.",
                    instructions_fields={},
                    input_fields=[],
                )
            ],
        ),
        user=UserMessage(
            user=[
                Topic(
                    topic_name="User Input",
                    topic_template="# Mathematics Problem\n<<<MATHEMATICS_QUESTION>>>\n",
                    instructions_fields={},
                    input_fields=["<<<MATHEMATICS_QUESTION>>>"],
                )
            ]
        ),
        response_type="str",
        response_model_scheme=None,
        demonstrations=[],
    )
    math_problem_dag = build_single_step_program(
        execute,
        model_name=model_name,
        dag_input_names=["<<<MATHEMATICS_QUESTION>>>"],
        dag_input_aliases={
            "question": "<<<MATHEMATICS_QUESTION>>>",
        },
        dag_output_aliases={"<<<ANSWER>>>": "answer"},
    )
    return math_problem_dag
