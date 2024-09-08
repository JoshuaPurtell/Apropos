import copy
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type

import networkx as nx
from pydantic import BaseModel, create_model

from apropos.src.core.programs.prompt import Demonstration


def get_python_type(json_type: str, item_type: str = None) -> Type:
    """
    Maps JSON schema types to Python types.

    Args:
    json_type (str): The JSON schema type.
    item_type (str): The type of items in an array, if applicable.

    Returns:
    Type: The corresponding Python type.
    """
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": List[get_python_type(item_type)] if item_type else List,
    }
    return type_mapping.get(json_type, str)


def create_pydantic_model_from_schema(
    schema: Dict[str, Any], model_name: str = "DynamicModel"
) -> BaseModel:
    """
    Creates a Pydantic model dynamically from a given schema.

    Args:
    schema (Dict[str, Any]): The schema dictionary of the model.
    model_name (str): The name of the dynamically created model.

    Returns:
    BaseModel: A dynamically created Pydantic model class.
    """
    field_definitions = {}
    for k, v in schema["properties"].items():
        field_type = v.get("type", "string")
        if field_type == "array":
            item_type = v["items"]["type"]
            python_type = get_python_type(field_type, item_type)
        else:
            python_type = get_python_type(field_type)
        field_definitions[k] = (python_type, ...)
    return create_model(model_name, **field_definitions)


@dataclass
class PromptRecord:
    user_message: str
    system_message: str
    outputs: Dict
    lm_name: str


@dataclass
class StageRecord:
    name: str
    inputs: Dict
    outputs: Dict
    prompt_record: PromptRecord

    def to_dict(self):
        return {
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "prompt_record": {
                "user_message": self.prompt_record.user_message,
                "system_message": self.prompt_record.system_message,
                "outputs": self.prompt_record.outputs,
                "lm_name": self.prompt_record.lm_name,
            },
        }


@dataclass
class AnnotatedStageRecord(StageRecord):
    name: str
    inputs: Dict
    outputs: Dict
    prompt_record: PromptRecord
    efficacious: bool
    annotation: str

    def to_demonstration(self) -> Demonstration:
        return Demonstration(
            inputs=self.inputs,
            outputs=self.outputs,
            prompt=self.prompt_record.system_message,
            gold_outputs=None,
            annotation=self.annotation,
        )


@dataclass
class DagRecord:
    stage_records: List[StageRecord]
    name_graph: nx.DiGraph

    def trivially_annotate(self):
        annotated_stages = []
        for stage in self.stage_records:
            annotated_stages.append(
                AnnotatedStageRecord(
                    name=stage.name,
                    inputs=stage.inputs,
                    outputs=stage.outputs,
                    prompt_record=stage.prompt_record,
                    efficacious=None,
                    annotation=None,
                )
            )
        return AnnotatedDagRecord(
            stage_records=annotated_stages,
            name_graph=self.name_graph,
            efficacious=None,
            annotation=None,
        )

    def to_dict(self):
        return {
            "stage_records": [stage.to_dict() for stage in self.stage_records],
            "name_graph": nx.to_dict_of_dicts(self.name_graph),
        }


@dataclass
class AnnotatedDagRecord(DagRecord):
    stage_records: List[AnnotatedStageRecord]
    name_graph: nx.DiGraph
    efficacious: bool
    annotation: str

    def to_stage_demonstrations(
        self,
        include_annotation=False,
        include_efficacy=False,
        include_inputs=True,
        include_outputs=True,
        include_prompt=False,
    ) -> List[Dict]:
        demonstrations = []
        for stage_record in self.stage_records:
            demonstrations.append(stage_record.to_demonstration())
        return demonstrations

    def to_dag_demonstration(
        self,
        include_annotation=False,
        include_efficacy=False,
        include_inputs=True,
        include_outputs=True,
        include_prompt=False,
    ) -> str:
        demonstration = ""
        for i, stage_record in enumerate(self.stage_records):
            demonstration += f"# Stage {i+1}\n"
            demonstration += stage_record.to_demonstration().to_string()
        if include_annotation:
            demonstration += f"Annotation for program result: {self.annotation}\n"
        if include_efficacy:
            demonstration += f"Efficacy: {self.efficacious}\n"
        return demonstration


class Runnable:
    @abstractmethod
    async def arun(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def to_dict(self):
        d = {}
        if hasattr(self, "llm_config"):
            d["llm_config"] = self.llm_config
        if hasattr(self, "prompt"):
            d["prompt"] = self.prompt.to_dict()
        return d


@dataclass
class TextGraphNode:
    name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    get_input_fields: Runnable
    transform: Runnable
    produce_stage_record: Runnable

    def to_dict(self):
        return {
            "name": self.name,
            "runnables": {
                "get_input_fields": self.get_input_fields.to_dict(),
                "transform": self.transform.to_dict(),
                "produce_stage_record": self.produce_stage_record.to_dict(),
            },
        }

    async def arun(self, inputs: Dict[str, Any], verbose: bool = False):
        self.inputs = inputs
        self.outputs = await self.transform.arun(inputs)
        return self.outputs

    def run(self, inputs: Dict[str, Any], verbose: bool = False):
        self.inputs = inputs
        self.outputs = self.transform.run(inputs)
        return self.outputs

    async def get_record(self):
        return await self.produce_stage_record.arun(
            inputs=self.inputs, outputs=self.outputs
        )

    def get_record_sync(self):
        return self.produce_stage_record.run(inputs=self.inputs, outputs=self.outputs)

    def get_inputs(self):
        return self.get_input_fields.run()


@dataclass
class LM_DAG:
    nodes: Dict[str, TextGraphNode]
    edges: List[Tuple[Tuple[str, str], Tuple[str, str]]]
    input_aliases: Dict[str, str]
    output_aliases: Dict[str, str]

    def __init__(self, nodes, edges, input_aliases={}, output_aliases={}):
        self.nodes = {node.name: node for node in nodes}
        self.edges = edges
        self.input_aliases = input_aliases
        self.output_aliases = output_aliases

    def to_dict(self):
        return {
            "nodes": {
                node_name: node.to_dict() for node_name, node in self.nodes.items()
            },
            "edges": self.edges,
            "input_aliases": self.input_aliases,
            "output_aliases": self.output_aliases,
        }

    def get_node_ancestors(self, node_name):
        graph = nx.DiGraph()
        for edge in self.edges:
            graph.add_edge(edge[0][0], edge[1][0])
        # Retrieve all ancestors of the node
        ancestors = nx.ancestors(graph, node_name)
        return list(nx.topological_sort(graph.subgraph(ancestors)))

    # update names

    def get_node_children(self, node_name):
        graph = nx.DiGraph()
        for edge in self.edges:
            graph.add_edge(edge[0][0], edge[1][0])
        children = [n for n in graph.successors(node_name)]
        return children

    def get_node_descendants(self, node_name):
        graph = nx.DiGraph()
        for edge in self.edges:
            graph.add_edge(edge[0][0], edge[1][0])
        descendants = nx.descendants(graph, node_name)
        return list(nx.topological_sort(graph.subgraph(descendants)))

    def get_node_relationship(self, node_a, node_b):
        graph = nx.DiGraph()
        for edge in self.edges:
            graph.add_edge(edge[0][0], edge[1][0])

        if nx.has_path(graph, node_a, node_b):
            return "parent"
        elif nx.has_path(graph, node_b, node_a):
            return "child"
        else:
            return "neither"

    def compile(self):
        g = nx.DiGraph()
        for edge in self.edges:
            g.add_edge(edge[0][0], edge[1][0])
        return g

    async def arun(
        self,
        base_inputs: Dict[str, Any],
        frozen_params: Dict = {"dag_record": None, "unfrozen_node_names": []},
        verbose: bool = False,
    ):
        if frozen_params["dag_record"] is not None:
            return await self.arun_frozen(
                base_inputs,
                frozen_params["dag_record"],
                frozen_params["unfrozen_node_names"],
                verbose=verbose,
            )
        else:
            return await self.arun_standard(base_inputs, verbose)

    async def arun_frozen(
        self,
        base_inputs: Dict[str, Any],
        dag_record: DagRecord,
        unfrozen_node_names: List[str],
        verbose: bool = False,
    ):
        all_unfrozen_node_names = set(unfrozen_node_names)
        for node_name in unfrozen_node_names:
            all_unfrozen_node_names.update(self.get_node_descendants(node_name))
        frozen_node_names = set(self.nodes.keys()) - all_unfrozen_node_names
        unaliased_base_inputs = copy.deepcopy(base_inputs)
        for k, v in unaliased_base_inputs.items():
            if k in self.input_aliases:
                del base_inputs[k]
                base_inputs[self.input_aliases[k]] = v

        assert set(
            [edge[0][1] for edge in self.edges if edge[0][0] == "DAG_INPUT"]
        ).issubset(
            set(base_inputs.keys())
        ), f"Inputs needed: {[edge[0][1] for edge in self.edges if edge[0][0] == 'DAG_INPUT']}, Inputs given: {base_inputs.keys()}"
        stage_records = []
        g = self.compile()
        outputs = {}
        for node_name in nx.topological_sort(g):
            if node_name in ["DAG_INPUT", "DAG_OUTPUT"]:
                continue
            node = self.nodes[node_name]
            inputs_needed = node.get_inputs()
            inputs = {}
            for edge in self.edges:
                if edge[1][0] != node_name:
                    continue
                if edge[1][0] == "DAG_OUTPUT":
                    continue
                if edge[0][0] == "DAG_INPUT":
                    inputs[edge[1][1]] = base_inputs[edge[0][1]]
                else:
                    if list(outputs[edge[0][0]].keys()) == [""]:
                        outputs[edge[0][0]][edge[0][1]] = outputs[edge[0][0]][""]
                    inputs[edge[1][1]] = outputs[edge[0][0]][edge[0][1]]

            assert set(inputs_needed).issubset(
                set(inputs.keys())
            ), f"Inputs needed: {inputs_needed}, Inputs given: {inputs.keys()}"
            inputs_normalized = {k: str(v) for k, v in inputs.items()}
            if node_name not in frozen_node_names:
                outputs_unstructured = await self.nodes[node_name].arun(
                    inputs=inputs_normalized, verbose=verbose
                )
                outputs[node_name] = {}
                if isinstance(outputs_unstructured, str):
                    outputs[node_name][""] = outputs_unstructured
                else:
                    dictified = (
                        outputs_unstructured.dict()
                        if isinstance(outputs_unstructured, BaseModel)
                        else outputs_unstructured
                    )
                    for key in dictified.keys():
                        outputs[node_name][key] = dictified[key]
                stage_records.append(await self.nodes[node_name].get_record())
            else:
                for record in dag_record.stage_records:
                    if (
                        record.name == node_name
                    ):  # TODO: this should be a runnable record
                        if isinstance(record.outputs, str):
                            outputs[node_name] = {}
                            outputs[node_name][""] = record.outputs
                        else:
                            outputs[node_name] = record.outputs
                        stage_records.append(record)
                        break

        name_graph = self.compile()
        dag_record = DagRecord(stage_records=stage_records, name_graph=name_graph)

        dag_outputs = {}
        for edge in self.edges:
            if edge[1][0] == "DAG_OUTPUT":
                if list(outputs[edge[0][0]].keys()) == [""]:
                    outputs[edge[0][0]][edge[0][1]] = outputs[edge[0][0]][""]
                dag_outputs[edge[1][1]] = outputs[edge[0][0]][edge[0][1]]

        for k in list(
            dag_outputs.keys()
        ):  # Use list to create a copy of keys for safe iteration
            if k in self.output_aliases:
                dag_outputs[self.output_aliases[k]] = dag_outputs[k]
                del dag_outputs[k]
        if verbose:
            return dag_outputs, dag_record

        return dag_outputs

    async def arun_standard(self, base_inputs: Dict[str, Any], verbose: bool = False):
        unaliased_base_inputs = copy.deepcopy(base_inputs)
        for k, v in unaliased_base_inputs.items():
            if k in self.input_aliases:
                del base_inputs[k]
                base_inputs[self.input_aliases[k]] = v

        assert set(
            [edge[0][1] for edge in self.edges if edge[0][0] == "DAG_INPUT"]
        ).issubset(
            set(base_inputs.keys())
        ), f"Inputs needed: {[edge[0][1] for edge in self.edges if edge[0][0] == 'DAG_INPUT']}, Inputs given: {base_inputs.keys()}"
        stage_records = []
        g = self.compile()
        outputs = {}
        for node_name in nx.topological_sort(g):
            if node_name in ["DAG_INPUT", "DAG_OUTPUT"]:
                continue
            node = self.nodes[node_name]
            inputs_needed = node.get_inputs()
            inputs = {}
            for edge in self.edges:
                if edge[1][0] != node_name:
                    continue
                if edge[1][0] == "DAG_OUTPUT":
                    continue
                if edge[0][0] == "DAG_INPUT":
                    inputs[edge[1][1]] = base_inputs[edge[0][1]]
                else:
                    if list(outputs[edge[0][0]].keys()) == [""]:
                        # print("Warning - fix this")
                        outputs[edge[0][0]][edge[0][1]] = outputs[edge[0][0]][""]
                    inputs[edge[1][1]] = outputs[edge[0][0]][edge[0][1]]
            assert set(inputs_needed).issubset(
                set(inputs.keys())
            ), f"Inputs needed: {inputs_needed}, Inputs given: {inputs.keys()}"
            inputs_normalized = {k: str(v) for k, v in inputs.items()}
            outputs_unstructured = await self.nodes[node_name].arun(
                inputs=inputs_normalized, verbose=verbose
            )
            assert isinstance(outputs_unstructured, str) or isinstance(
                outputs_unstructured, BaseModel
            ), f"Outputs unstructured: {outputs_unstructured}"
            outputs[node_name] = {}
            if isinstance(outputs_unstructured, str):
                outputs[node_name][""] = outputs_unstructured
            else:
                dictified = (
                    outputs_unstructured.dict()
                    if isinstance(outputs_unstructured, BaseModel)
                    else outputs_unstructured
                )
                for key in dictified.keys():
                    outputs[node_name][key] = dictified[key]

            stage_records.append(await self.nodes[node_name].get_record())
        name_graph = self.compile()
        dag_record = DagRecord(stage_records=stage_records, name_graph=name_graph)

        dag_outputs = {}
        for edge in self.edges:
            if edge[1][0] == "DAG_OUTPUT":
                if list(outputs[edge[0][0]].keys()) in [[""], [""]]:
                    outputs[edge[0][0]][edge[0][1]] = outputs[edge[0][0]][""]
                elif (
                    not edge[0][1] in list(outputs[edge[0][0]].keys())
                    and edge[0][1] == "answer"
                ):
                    outputs[edge[0][0]]["answer"] = outputs[edge[0][0]]
                dag_outputs[edge[1][1]] = outputs[edge[0][0]][edge[0][1]]

        for k in list(dag_outputs.keys()):
            if k in self.output_aliases:
                dag_outputs[self.output_aliases[k]] = dag_outputs[k]
                del dag_outputs[k]

        if verbose:
            return dag_outputs, dag_record
        return dag_outputs

    def run_standard(self, base_inputs: Dict[str, Any], verbose: bool = False):
        unaliased_base_inputs = copy.deepcopy(base_inputs)
        for k, v in unaliased_base_inputs.items():
            if k in self.input_aliases:
                del base_inputs[k]
                base_inputs[self.input_aliases[k]] = v

        assert set(
            [edge[0][1] for edge in self.edges if edge[0][0] == "DAG_INPUT"]
        ).issubset(
            set(base_inputs.keys())
        ), f"Inputs needed: {[edge[0][1] for edge in self.edges if edge[0][0] == 'DAG_INPUT']}, Inputs given: {base_inputs.keys()}"
        stage_records = []
        g = self.compile()
        outputs = {}
        for node_name in nx.topological_sort(g):
            if node_name in ["DAG_INPUT", "DAG_OUTPUT"]:
                continue
            node = self.nodes[node_name]
            inputs_needed = node.get_inputs()  # Assuming get_inputs can be synchronous
            inputs = {}
            for edge in self.edges:
                if edge[1][0] != node_name:
                    continue
                if edge[1][0] == "DAG_OUTPUT":
                    continue
                if edge[0][0] == "DAG_INPUT":
                    inputs[edge[1][1]] = base_inputs[edge[0][1]]
                else:
                    if list(outputs[edge[0][0]].keys()) == [""]:
                        outputs[edge[0][0]][edge[0][1]] = outputs[edge[0][0]][""]
                    inputs[edge[1][1]] = outputs[edge[0][0]][edge[0][1]]
            assert set(inputs_needed).issubset(
                set(inputs.keys())
            ), f"Inputs needed: {inputs_needed}, Inputs given: {inputs.keys()}"
            inputs_normalized = {k: str(v) for k, v in inputs.items()}
            outputs_unstructured = node.run(
                inputs=inputs_normalized, verbose=verbose
            )  # Assuming run can be synchronous
            outputs[node_name] = {}

            if isinstance(outputs_unstructured, str):
                outputs[node_name][""] = outputs_unstructured
            else:
                dictified = (
                    outputs_unstructured.dict()
                    if isinstance(outputs_unstructured, BaseModel)
                    else outputs_unstructured
                )
                for key in dictified.keys():
                    outputs[node_name][key] = dictified[key]

            stage_records.append(node.get_record_sync())
        name_graph = self.compile()
        dag_record = DagRecord(stage_records=stage_records, name_graph=name_graph)

        dag_outputs = {}
        for edge in self.edges:
            if edge[1][0] == "DAG_OUTPUT":
                if list(outputs[edge[0][0]].keys()) in [[""], [""]]:
                    outputs[edge[0][0]][edge[0][1]] = outputs[edge[0][0]][""]
                elif (
                    not edge[0][1] in list(outputs[edge[0][0]].keys())
                    and edge[0][1] == "answer"
                ):
                    outputs[edge[0][0]]["answer"] = outputs[edge[0][0]]
                dag_outputs[edge[1][1]] = outputs[edge[0][0]][edge[0][1]]

        for k in list(
            dag_outputs.keys()
        ):  # Use list to create a copy of keys for safe iteration
            if k in self.output_aliases:
                dag_outputs[self.output_aliases[k]] = dag_outputs[k]
                del dag_outputs[k]

        if verbose:
            return dag_outputs, dag_record
        return dag_outputs
