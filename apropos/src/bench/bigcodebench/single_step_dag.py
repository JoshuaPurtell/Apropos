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


def code_problem_single_step(model_name="gpt-3.5-turbo"):
    execute = PromptTemplate(
        name="Code Problem Solution",
        system=SystemMessage(
            premise=[],
            objective=[
                Topic(
                    topic_name="Objective",
                    topic_template="# Objective\nProvide a solution to the provided coding problem",
                    instructions_fields={},
                    input_fields=[],
                )
            ],
            constraints=[
                Topic(
                    topic_name="Constraints",
                    topic_template="# Constraints\n Provide your solution as code that can be appended to the provided code signature and executed to solve the problem. Return your code in the following format: \n```python```. Do not re-write the function signature - return code that can be appended to the signature as-is. Keep in mind that your code, and potentially its file outputs, will be tested - do not delete any files it produces.",
                    instructions_fields={},
                    input_fields=[],
                )
            ],
        ),
        user=UserMessage(
            user=[
                Topic(
                    topic_name="User Input",
                    topic_template="# Coding Task Problem\n<<<CODING_QUESTION>>>\n",
                    instructions_fields={},
                    input_fields=["<<<CODING_QUESTION>>>"],
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
        dag_input_names=["<<<CODING_QUESTION>>>"],
        dag_input_aliases={
            "question": "<<<CODING_QUESTION>>>",
        },
        dag_output_aliases={"<<<ANSWER>>>": "answer"},
    )
    return math_problem_dag
