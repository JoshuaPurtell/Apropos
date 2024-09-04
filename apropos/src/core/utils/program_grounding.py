from dataclasses import dataclass
from pickle import NONE
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

from apropos.src.bench.base import Benchmark
from apropos.src.core.lms.helpers import LLM
from apropos.src.core.programs.convenience_functions.dag_constructors import (
    build_single_step_program,
)
from apropos.src.core.programs.dag import LM_DAG
from apropos.src.core.programs.prompt import (
    PromptTemplate,
    SystemMessage,
    Topic,
    UserMessage,
)
from apropos.src.core.utils.internal_dataset import (
    SyntheticBenchmark,
    SyntheticQuestion,
    SyntheticQuestionGoldOutput,
)


@dataclass
class TopicInfo:
    name: str
    content: str


class TopicResponse(BaseModel):
    names: List[str]
    first_line_indices: List[int]
    last_line_indices: List[int]


async def get_topics_for_prompt(
    prompt: str, other_prompts: List[str], llm: LLM
) -> List[TopicInfo]:
    system_message = """
Please review the provided prompt section and break it down into a sequential series of topics.
Topics have the following properties:
- Focus on a single important theme or aspect of the prompt 
- Can be understood and interpreted independently of other topics/sections

Review the prompt and return the topics, each containing the following information:
- A name for the topic
- The index of the first line of the topic content
- The index of the last line of the topic content

Your topics should be non-overlapping and contiguous.
If your topic corresponds to a present heading or section, don't just include the line including the heading/section, but all content under it as well.

You will be given a list of additional instances of the prompt for reference. 
Do not include content that varies between prompt instances in your topics.
"""
    numbered_prompt_lines = [
        f"{i}: {line}" for i, line in enumerate(prompt.split("\n"))
    ]

    numbered_other_prompts = "\n".join(
        [
            f"<Prompt Instance {i}>\n{prompt_instance}\n</Prompt Instance>"
            for i, prompt_instance in enumerate(other_prompts)
        ]
    )
    prompt_numbered = "\n".join(numbered_prompt_lines)
    user_message = f"""
<Prompt>
{prompt_numbered}
</Prompt>

<Additional Prompt Instances>
{numbered_other_prompts}
</Additional Prompt Instances>

Your topics: """

    topic_response = await llm.async_respond(
        system_prompt=system_message,
        user_prompt=user_message,
        response_model=TopicResponse,
    )
    topics = []
    for i in range(len(topic_response.names)):
        topic_info = TopicInfo(
            name=topic_response.names[i],
            content="\n".join(
                prompt.split("\n")[
                    topic_response.first_line_indices[i] : (
                        topic_response.last_line_indices[i] + 1
                    )
                ]
            ),
        )
        topics.append(topic_info)
    return topics


class TopicBreakdownResponse(BaseModel):
    premise_topics: List[int]
    objective_topics: List[int]
    constraints_topics: List[int]


async def assign_topics_to_prompt_section(
    prompt: str, topics: str, llm: LLM
) -> List[str]:
    system_message = """
You'll be provided with a prompt and a number of topics that have been identified within it.
Please break down the prompt into the following three sections:
- Premise: Any setup or context information about the problem or task at hand that's introduced
- Objective: Any content explicating the main task or goal, providing instructions or details about what the user is trying to achieve
- Constraints: Any requirements, limitations or rules that the user must adhere to while completing the task
"""
    index_topics = [f"{i}: {topic}" for i, topic in enumerate(topics)]
    topics_by_index = "\n".join(index_topics)
    user_message = f"""
<Prompt>
{prompt}
</Prompt>

<Topics By Index>
{topics_by_index}
</Topics By Index>
"""
    assignments = await llm.async_respond(
        system_prompt=system_message,
        user_prompt=user_message,
        response_model=TopicBreakdownResponse,
    )
    final_assignments = []
    for i, topic in enumerate(topics):
        if i in assignments.premise_topics:
            final_assignments.append("premise")
        elif i in assignments.objective_topics:
            final_assignments.append("objective")
        elif i in assignments.constraints_topics:
            final_assignments.append("constraints")
        else:
            raise ValueError(f"Topic {topic} not assigned to a section")
    return final_assignments


class BreakdownFieldsResponse(BaseModel):
    template: str
    instructions: Dict[str, str]
    input_fields: List[str]


async def break_down_prompt_content_for_topic_into_components(
    main_prompt_instance: str,
    prompt_instances: List[str],
    topic_info: TopicInfo,
    llm: LLM,
) -> Topic:
    system_message = """
You'll be provided with a prompt and a section within it that's be highlighted as a topic.
Because the prompt can change with inputs, you'll also be provided with a list of additional prompt instances for reference.

Please breakdown the topic into a template with instruction and input fields, where appropriate.
1. The template represents the original topic, except instruction and input fields are replaced with placeholders.
2. Instructions comprise single, atomic commands.
3. Input fields are segments of the prompt that appear to contain information that can change between instances.

Instruction placeholders are denoted with $ and all-caps. Input field placeholders are denoted with <<<*>>> and all-caps.

# Examples
Topic: # Objective\nProvide a solution to the provided mathematics problem
Output:
- Template: "# Objective\n$SOLVE_MATHEMATICS_PROBLEM"
- Instructions: {"SOLVE_MATHEMATICS_PROBLEM": "Provide a solution to the provided mathematics problem"}
- Input fields: []

Topic: # Constraints\nEnsure you are polite, helpful, and concise. ## User Constraints\n- Use markdown\n- Use code blocks for code snippets
Output:
- Template: "# Constraints\n$CORE_CONSTRAINTS\n## User Constraints\n<<<USER_CONSTRAINTS>>>"
- Instructions: {"CORE_CONSTRAINTS": "Ensure you are polite, helpful, and concise."}
- Input fields: ["USER_CONSTRAINTS"]
"""
    # opic = "\n".join(main_prompt_instance.split('\n')[topic_start_index:topic_end_index])
    additional_prompt_instances = "\n".join(
        [
            f"## Prompt Instance {i}\n{prompt_instance}"
            for i, prompt_instance in enumerate(prompt_instances)
        ]
    )
    user_message = f"""
# First Prompt Instance
{main_prompt_instance}

# Topic Within First Prompt Instance
## Name 
{topic_info.name}

## Content
{topic_info.content}

# Additional Prompt Instances
{additional_prompt_instances}
"""

    breakdown = await llm.async_respond(
        system_prompt=system_message,
        user_prompt=user_message,
        response_model=BreakdownFieldsResponse,
    )
    return Topic(
        topic_name=topic_info.name,
        topic_template=breakdown.template,
        instructions_fields=breakdown.instructions,
        input_fields=[f"<<<{field}>>>" for field in breakdown.input_fields],
    )


async def build_prompt_template(
    name_for_prompt: str,
    system_prompt: str,
    user_prompt: str,
    input_aliases: List[str],
    output_aliases: List[str],
    system_prompt_instances: List[str],
    user_prompt_instances: List[str],
    response_model: Union[str, BaseModel],
    llm: LLM,
):
    # One LM to split the prompt into topics
    system_topics = await get_topics_for_prompt(
        system_prompt, system_prompt_instances[0:10], llm
    )
    user_topics = await get_topics_for_prompt(
        user_prompt, user_prompt_instances[0:10], llm
    )
    # One LM to assign topics to prompt sections (premise, objective, constraints)
    topic_assignments: List[str] = await assign_topics_to_prompt_section(
        system_prompt, system_topics, llm
    )
    # One LM to parse each topic into template, instructions, and input fields
    system_topic_breakdowns = [
        await break_down_prompt_content_for_topic_into_components(
            system_prompt, system_prompt_instances, topic_info, llm
        )
        for topic_info in system_topics
    ]
    user_topics = [
        await break_down_prompt_content_for_topic_into_components(
            user_prompt, user_prompt_instances, topic_info, llm
        )
        for topic_info in user_topics
    ]
    # Then, combine together into a single prompt template
    system_premise_topics = [
        topic
        for i, topic in enumerate(system_topic_breakdowns)
        if topic_assignments[i] == "premise"
    ]
    system_objective_topics = [
        topic
        for i, topic in enumerate(system_topic_breakdowns)
        if topic_assignments[i] == "objective"
    ]
    system_constraints_topics = [
        topic
        for i, topic in enumerate(system_topic_breakdowns)
        if topic_assignments[i] == "constraints"
    ]

    prompt_template = PromptTemplate(
        name=name_for_prompt,
        system=SystemMessage(
            premise=system_premise_topics,
            objective=system_objective_topics,
            constraints=system_constraints_topics,
        ),
        user=UserMessage(user=user_topics),
        response_type="str" if isinstance(response_model, str) else "pydantic",
        response_model_scheme=response_model
        if isinstance(response_model, BaseModel)
        else None,
        demonstrations=[],
    )
    return prompt_template


class AliasesToPromptInputs(BaseModel):
    mapping: Dict[str, str]


async def match_aliases_to_prompt_template_inputs(
    aliases: List[str], prompt_template: PromptTemplate, llm: LLM
) -> Dict[str, str]:
    system_message = """
You'll be provided with a list of aliases, along with inputs to a prompt template.
Please match the aliases to the prompt template inputs by providing a mapping from alias to prompt template input.
"""
    user_message = f"""
# Aliases
{aliases}

# Prompt Template
{prompt_template}
"""
    response = await llm.async_respond(
        system_prompt=system_message,
        user_prompt=user_message,
        response_model=AliasesToPromptInputs,
    )
    return response.mapping  # Alias -> Input


# Does not currently support demonstrations
# Currently only supports a single output (can be str or pydantic)
async def ground_program_to_single_step_dag(
    name_for_prompt: str,
    system_prompt: str,
    user_prompt: str,
    input_aliases: List[str],
    output_alias: str,
    system_prompt_instances: List[str],
    user_prompt_instances: List[str],
    response_model: Union[str, BaseModel],
    ground_llm: LLM,
    program_llm: LLM,
) -> Tuple[PromptTemplate, LM_DAG]:
    prompt_template = await build_prompt_template(
        name_for_prompt,
        system_prompt,
        user_prompt,
        input_aliases,
        output_alias,
        system_prompt_instances,
        user_prompt_instances,
        response_model,
        ground_llm,
    )
    input_aliases_to_inputs = await match_aliases_to_prompt_template_inputs(
        input_aliases, prompt_template, ground_llm
    )
    dag = build_single_step_program(
        prompt_template,
        model_name=program_llm.model_name,
        dag_input_names=[v for k, v in input_aliases_to_inputs.items()],
        dag_input_aliases={k: v for k, v in input_aliases_to_inputs.items()},
        dag_output_aliases={"<<<ANSWER>>>": output_alias},
    )
    return prompt_template, dag


def old_school_search(lhs, rhs, content):
    if isinstance(lhs, str) and isinstance(rhs, str):
        return content.split(lhs)[1].split(rhs)[0]
    elif isinstance(lhs, str):
        return content.split(lhs)[1]
    elif isinstance(rhs, str):
        return content.split(rhs)[0]
    else:
        return None


def get_inputs_from_topic(content: str, topic: Topic) -> Dict[str, str]:
    assert isinstance(topic, Topic), f"Topic is not a Topic: {topic}"
    assert isinstance(content, str)
    if not topic.input_fields:
        return {}
    else:
        inputs = {}
        for field in topic.input_fields:
            lhs = topic.topic_template.split(field)[0]
            rhs = topic.topic_template.split(field)[1]
            rightmost_instruction_field_in_lhs = None
            if topic.instructions_fields:
                instruction_fields_in_lhs = [
                    k for k, v in topic.instructions_fields.items() if v in lhs
                ]
                if instruction_fields_in_lhs:
                    rightmost_instruction_field_in_lhs = instruction_fields_in_lhs[-1]
            if rightmost_instruction_field_in_lhs:
                lhs = lhs.split(rightmost_instruction_field_in_lhs)[1]
            leftmost_instruction_field_in_rhs = None
            if topic.instructions_fields:
                instruction_fields_in_rhs = [
                    k for k, v in topic.instructions_fields.items() if v in rhs
                ]
                if instruction_fields_in_rhs:
                    leftmost_instruction_field_in_rhs = instruction_fields_in_rhs[0]
            if leftmost_instruction_field_in_rhs:
                rhs = rhs.split(leftmost_instruction_field_in_rhs)[0]
            lhs = lhs.rstrip()
            rhs = rhs.lstrip()
            if len(lhs.strip()) > 0 and len(rhs.strip()) > 0:
                content_between = old_school_search(lhs, rhs, content)
            elif rhs.strip():
                content_between = old_school_search(NONE, rhs, content)
            elif lhs.strip():
                content_between = old_school_search(lhs, NONE, content)
            else:
                content_between = old_school_search(NONE, NONE, content)
            if content_between:
                inputs[field] = content_between
        return inputs


def parse_prompt_instance_into_inputs(
    prompt_instance, prompt: PromptTemplate
) -> Dict[str, str]:
    inputs = {}
    for topic in prompt.system.premise:
        inputs.update(get_inputs_from_topic(prompt_instance, topic))
    for topic in prompt.system.objective:
        inputs.update(get_inputs_from_topic(prompt_instance, topic))
    for topic in prompt.system.constraints:
        inputs.update(get_inputs_from_topic(prompt_instance, topic))
    for topic in prompt.user.user:
        inputs.update(get_inputs_from_topic(prompt_instance, topic))
    return inputs


@dataclass
class Metric:
    gold_outputs_for_dataset: Optional[List[Any]] = None
    metric_function: Callable = None


def messages_to_dataset(
    messages: List[Tuple[str, str]],
    prompt_template: PromptTemplate,
    metric: Callable,
    input_keys: List[str],
) -> Benchmark:
    dataset = [
        parse_prompt_instance_into_inputs(messages[i][1], prompt_template)
        for i in range(len(messages))
    ]
    if metric.gold_outputs_for_dataset:
        assert (
            len(metric.gold_outputs_for_dataset) == len(dataset)
        ), f"Gold outputs for dataset must be the same length as the dataset, instead got {len(metric.gold_outputs_for_dataset)} and {len(dataset)}"
        questions = [
            SyntheticQuestionGoldOutput(
                standardized_information={
                    **dataset[i],
                    "answer": metric.gold_outputs_for_dataset[i],
                },
                metric=metric.metric_function,
                input_keys=input_keys,
            )
            for i in range(len(dataset))
        ]
    else:
        questions = [
            SyntheticQuestion(
                standardized_information=dataset[i],
                metric=metric.metric_function,
                input_keys=input_keys,
            )
            for i in range(len(dataset))
        ]
    return SyntheticBenchmark(questions)


async def messages_to_dag_and_benchmark(
    messages: List[Tuple[str, str]],
    metric: Metric,
    input_keys: List[str],
    prompt_name: str,
    ground_llm: LLM,
    program_llm: LLM,
) -> Tuple[LM_DAG, Benchmark]:
    template, dag = await ground_program_to_single_step_dag(
        name_for_prompt=prompt_name,
        system_prompt=messages[0][0],
        user_prompt=messages[0][1],
        input_aliases=input_keys,
        output_alias="<<<ANSWER>>>",
        system_prompt_instances=[messages[i][0] for i in range(1, 100)],
        user_prompt_instances=[messages[i][1] for i in range(1, 100)],
        response_model="str",
        ground_llm=ground_llm,
        program_llm=program_llm,
    )
    benchmark = messages_to_dataset(
        messages, template, metric, list(dag.input_aliases.values())
    )
    return dag, benchmark


# if __name__ == "__main__":
#     import asyncio

#     from apropos.src.bench.hendryks_math.main import (
#         HendryksMath_Benchmark,
#         custom_math_metric,
#     )
#     benchmark = HendryksMath_Benchmark()
#     from apropos.src.bench.hendryks_math.dags.single_step import (
#         hendryks_math_single_step_example,
#     )
#     baseline_single_step_program = hendryks_math_single_step_example(
#         model_name = "claude-3-haiku-20240307"
#     )
#     messages_examples = []
#     for q in benchmark.train:
#         systems_message, user_message = list(baseline_single_step_program.nodes.values())[0].transform.prompt.compile(
#             inputs = {
#                 "<<<MATHEMATICS_QUESTION>>>": q.information["question"],
#             }
#         )
#         messages_examples.append((systems_message, user_message))
#     llm = LLM("claude-3-5-sonnet-20240620")
#     metric = Metric(
#         gold_outputs_for_dataset=[q.information["answer"] for q in benchmark.train],
#         metric_function=custom_math_metric
#     )
#     dag, benchmark = asyncio.run(messages_to_dag_and_benchmark(messages_examples, metric, ["question"], "Solve Problem", llm))
#     print(benchmark.train[0])
#     result  = asyncio.run(
#         benchmark.train[0].compute_and_score_attempt(
#             dag,
#         )
#     )
#     print(result)
