from dataclasses import dataclass
from typing import Any, Dict, List

import networkx as nx

from apropos.src.core.lms.helpers import LLM
from apropos.src.core.programs.dag import (
    LM_DAG,
    PromptRecord,
    Runnable,
    StageRecord,
    TextGraphNode,
    create_pydantic_model_from_schema,
)
from apropos.src.core.programs.prompt import PromptTemplate


async def llm_transform(
    inputs_normalized,
    prompt,
    llm_config: dict = {"model_name": "gpt-3.5-turbo", "temperature": 0},
):
    if prompt.response_type == "pydantic":
        rm = create_pydantic_model_from_schema(prompt.response_model_scheme)
        outputs_unstructured = await prompt.arun(
            inputs=inputs_normalized,
            lm=LLM(llm_config["model_name"], temperature=llm_config["temperature"]),
            custom_instructions_fields={},
            response_model=rm,
        )
    else:
        outputs_unstructured = await prompt.arun(
            inputs=inputs_normalized,
            lm=LLM(llm_config["model_name"], temperature=llm_config["temperature"]),
            custom_instructions_fields={},
            response_model=None,
        )
    return outputs_unstructured


def llm_transform_sync(
    inputs_normalized,
    prompt,
    llm_config: dict = {
        "model_name": "gpt-3.5-turbo",
        "temperature": 0,
        "multi_threaded": False,
    },
):
    if prompt.response_type == "pydantic":
        rm = create_pydantic_model_from_schema(prompt.response_model_scheme)
        outputs_unstructured = prompt.run(
            inputs=inputs_normalized,
            lm=LLM(llm_config["model_name"], temperature=llm_config["temperature"]),
            custom_instructions_fields={},
            response_model=rm,
            multi_threaded=llm_config["multi_threaded"]
            if "multi_threaded" in llm_config
            else False,
        )
    else:
        outputs_unstructured = prompt.run(
            inputs=inputs_normalized,
            lm=LLM(llm_config["model_name"], temperature=llm_config["temperature"]),
            custom_instructions_fields={},
            response_model=None,
            multi_threaded=llm_config["multi_threaded"]
            if "multi_threaded" in llm_config
            else False,
        )
    return outputs_unstructured


async def llm_get_stage_record(
    name, inputs, outputs, prompt, llm_config: dict = {"model_name": "gpt-3.5-turbo"}
):
    system_message_compiled, user_message_compiled = prompt.compile(inputs, {})
    prompt_record = PromptRecord(
        user_message=user_message_compiled,
        system_message=system_message_compiled,
        outputs=outputs,
        lm_name=llm_config["model_name"],
    )
    return StageRecord(
        name=name, prompt_record=prompt_record, inputs=inputs, outputs=outputs
    )


def llm_get_stage_record_sync(
    name, inputs, outputs, prompt, llm_config: dict = {"model_name": "gpt-3.5-turbo"}
):
    system_message_compiled, user_message_compiled = prompt.compile(inputs, {})
    prompt_record = PromptRecord(
        user_message=user_message_compiled,
        system_message=system_message_compiled,
        outputs=outputs,
        lm_name=llm_config["model_name"],
    )
    return StageRecord(
        name=name, prompt_record=prompt_record, inputs=inputs, outputs=outputs
    )


@dataclass
class Transform(Runnable):
    prompt: PromptTemplate
    llm_config: dict

    async def arun(self, inputs: Dict[str, Any]):
        from pydantic import BaseModel

        result = await llm_transform(inputs, self.prompt, self.llm_config)
        assert isinstance(result, str) or isinstance(
            result, BaseModel
        ), f"Result: {result}"
        return result

    def run(self, inputs: Dict[str, Any]):
        return llm_transform_sync(inputs, self.prompt, self.llm_config)


@dataclass
class ProduceStageRecord(Runnable):
    prompt: PromptTemplate
    llm_config: dict

    async def arun(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        return await llm_get_stage_record(
            self.prompt.name, inputs, outputs, self.prompt, self.llm_config
        )

    def run(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        return llm_get_stage_record_sync(
            self.prompt.name, inputs, outputs, self.prompt, self.llm_config
        )


@dataclass
class GetInputFields(Runnable):
    prompt: PromptTemplate

    def run(self):
        return self.prompt.get_input_fields()


def get_runnables_for_llm_based_stage(
    prompt: PromptTemplate,
    llm_config: dict = {"model_name": "gpt-3.5-turbo", "temperature": 0},
):  # model_name: str = "gpt-3.5-turbo"
    transform = Transform(prompt, llm_config)  # "model_name"
    produce_stage_record = ProduceStageRecord(prompt, llm_config)
    get_input_fields = GetInputFields(prompt)
    return transform, produce_stage_record, get_input_fields


def build_single_step_program(
    prompt: PromptTemplate,
    model_name: str,
    dag_input_names: List[str],
    dag_input_aliases,
    dag_output_aliases,
):
    edges = []
    transform_runnable, produce_stage_record_runnable, get_input_fields_runnable = (
        get_runnables_for_llm_based_stage(
            prompt, {"model_name": model_name, "temperature": 0.01}
        )
    )
    input_fields = prompt.get_input_fields()
    output_fields = {"answer": "str"}
    node = TextGraphNode(
        name=prompt.name,
        inputs=input_fields,
        outputs=output_fields,
        transform=transform_runnable,
        produce_stage_record=produce_stage_record_runnable,
        get_input_fields=get_input_fields_runnable,
    )
    nodes = [node]
    for dag_input_name in dag_input_names:
        if dag_input_name in input_fields:
            edges.append((("DAG_INPUT", dag_input_name), (prompt.name, dag_input_name)))
    edges.append(((prompt.name, "answer"), ("DAG_OUTPUT", "answer")))
    return LM_DAG(nodes, edges, dag_input_aliases, dag_output_aliases)


# TODO: change this so the output aliasing is nontrivial
async def build_path_program(
    prompts: List[PromptTemplate],
    model_names: List[str],
    dag_input_names: List[str],
    dag_input_aliases,
    dag_output_aliases,
):
    assert len(prompts) == len(
        model_names
    ), "Each prompt should have a corresponding model name"
    staggered_prompts = prompts[1:] + [None]

    nodes = []
    edges = []
    for model_name, prompt, staggered in zip(model_names, prompts, staggered_prompts):
        transform_runnable, produce_stage_record_runnable, get_input_fields_runnable = (
            get_runnables_for_llm_based_stage(
                prompt, {"model_name": model_name, "temperature": 0}
            )
        )

        input_fields = prompt.get_input_fields()
        output_fields_possibly_with_dag_inputs = (
            {"answer": "str"} if not staggered else staggered.get_input_fields()
        )  # ??
        output_fields = [
            field
            for field in output_fields_possibly_with_dag_inputs
            if field not in dag_input_names
        ]
        node = TextGraphNode(
            name=prompt.name,
            inputs=input_fields,
            outputs=output_fields,
            transform=transform_runnable,
            produce_stage_record=produce_stage_record_runnable,
            get_input_fields=get_input_fields_runnable,
        )
        nodes.append(node)

        # Mostly a path graph, but we allow the input to be added to any node
        for dag_input_name in dag_input_names:
            if dag_input_name in input_fields:
                edges.append(
                    (("DAG_INPUT", dag_input_name), (prompt.name, dag_input_name))
                )

        if not staggered:
            edges.append(((prompt.name, "answer"), ("DAG_OUTPUT", "answer")))
        else:
            staggered_prompt_input_name = output_fields[0]
            edges.append(
                (
                    (prompt.name, staggered_prompt_input_name),
                    (staggered.name, staggered_prompt_input_name),
                )
            )
    return LM_DAG(nodes, edges, dag_input_aliases)


async def build_dag_program(
    prompts: Dict[str, PromptTemplate],
    name_dag: nx.DiGraph,
    model_configs: Dict[str, Dict],
    dag_input_names: List[str] = ["<<<MATHEMATICS_QUESTION>>>"],
    dag_input_aliases={"question": "<<<MATHEMATICS_QUESTION>>>"},
    dag_output_aliases={"<<<FINAL_ANSWER>>>": "answer"},
):
    assert len(prompts) == len(
        model_configs
    ), "Each prompt should have a corresponding model name"
    nodes = []
    edges = []

    outputs_by_node = {"DAG_INPUT": dag_input_names}
    for name in nx.topological_sort(name_dag):
        if name == "DAG_INPUT" or name == "DAG_OUTPUT":
            continue
        prompt = prompts[name]
        model_config = model_configs[name]
        input_fields = prompt.get_input_fields()
        children = list(name_dag.successors(name))
        parents = list(name_dag.predecessors(name))

        output_fields = [
            name_dag.get_edge_data(name, child)["attribute_name"] for child in children
        ]

        transform_runnable, produce_stage_record_runnable, get_input_fields_runnable = (
            get_runnables_for_llm_based_stage(prompt, model_config)
        )
        node = TextGraphNode(
            name=name,
            inputs=input_fields,
            outputs=output_fields,
            transform=transform_runnable,
            produce_stage_record=produce_stage_record_runnable,
            get_input_fields=get_input_fields_runnable,
        )
        nodes.append(node)
        outputs_by_node[name] = output_fields

        parents = list(name_dag.predecessors(name))

        for dag_input_name in dag_input_names:
            if dag_input_name in input_fields:
                edges.append(
                    (("DAG_INPUT", dag_input_name), (prompt.name, dag_input_name))
                )

        for parent in parents:
            parent_outputs = outputs_by_node[parent]
            for parent_output in parent_outputs:
                edges.append(((parent, parent_output), (name, parent_output)))

        if len(children) != 0:
            for child_name in children:
                attribute = name_dag.get_edge_data(name, child_name)["attribute_name"]
                edges.append(((name, attribute), (child_name, attribute)))
    return LM_DAG(nodes, edges, dag_input_aliases, dag_output_aliases)
