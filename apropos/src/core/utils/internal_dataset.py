import random
from typing import Callable, Dict, List, Tuple, Union

from apropos.src.bench.base import QABenchmark, Question
from apropos.src.core.programs.dag import LM_DAG, DagRecord


# class SyntheticQuestion(Question):
#     information: Dict
#     correctness: bool
#     input_keys: List[str]
#     metric: Callable
#     def __init__(self, standardized_information: Dict, metric: Callable, input_keys: List[str]):
#         self.information = standardized_information
#         self.correctness = None
#         self.input_keys = input_keys
#         self.metric = metric

#     def compute_and_score_attempt_sync(self, lm_dag: LM_DAG) -> Tuple[bool, DagRecord]:
#         unique_inputs = list(set([edge[0][1] for edge in lm_dag.edges if edge[0][0] == "DAG_INPUT"]))
#         assert len(unique_inputs) == len(self.input_keys), f"There should be exactly one input edge, instead got {unique_inputs}"
#         assert all([input in unique_inputs for input in self.input_keys]), f"All input keys should be in the unique inputs, instead got {unique_inputs}"
#         output_edges = [edge for edge in lm_dag.edges if edge[1][0] == "DAG_OUTPUT"]
#         assert len(output_edges) == 1, f"There should be exactly one output edge, instead got {len(output_edges)}"

#         output, dag_record = lm_dag.run_standard({"question": self.information['question']}, verbose=True)
#         answer = output["answer"]
#         score = self.metric(answer)
#         return score, dag_record

#     async def compute_and_score_attempt(self, lm_dag: LM_DAG, frozen_params: Dict = {"dag_record": None, "unfrozen_node_names": []}) -> Tuple[bool, DagRecord]:
#         unique_inputs = list(set([edge[0][1] for edge in lm_dag.edges if edge[0][0] == "DAG_INPUT"]))
#         assert len(unique_inputs) == len(self.input_keys), f"There should be exactly one input edge, instead got {unique_inputs}"
#         assert all([input in unique_inputs for input in self.input_keys]), f"All input keys should be in the unique inputs, wanted {self.input_keys} instead got {unique_inputs}"
#         output_edges = [edge for edge in lm_dag.edges if edge[1][0] == "DAG_OUTPUT"]
#         assert len(output_edges) == 1, f"There should be exactly one output edge, instead got {len(output_edges)}"
#         output, dag_record = await lm_dag.arun({
#             k: v for k,v in self.information.items() if k in unique_inputs
#         },frozen_params=frozen_params, verbose=True)
#         answer = output["answer"]
#         score = self.metric(answer)
#         return score, dag_record

# class SyntheticQuestionGoldOutput(Question):
#     information: Dict
#     correctness: bool
#     input_keys: List[str]
#     metric: Callable
#     def __init__(self, standardized_information: Dict, metric: Callable, input_keys: List[str]):
#         self.information = standardized_information
#         self.correctness = None
#         self.input_keys = input_keys
#         self.metric = metric

#     def compute_and_score_attempt_sync(self, lm_dag: LM_DAG) -> Tuple[bool, DagRecord]:
#         assert "answer" in self.information, f"Answer should be in the information for questions with gold output, instead got {self.information}"
#         unique_inputs = list(set([edge[0][1] for edge in lm_dag.edges if edge[0][0] == "DAG_INPUT"]))
#         assert len(unique_inputs) == len(self.input_keys), f"There should be exactly one input edge, instead got {unique_inputs}"
#         assert all([input in unique_inputs for input in self.input_keys]), f"All input keys should be in the unique inputs, instead got {unique_inputs}"
#         output_edges = [edge for edge in lm_dag.edges if edge[1][0] == "DAG_OUTPUT"]
#         assert len(output_edges) == 1, f"There should be exactly one output edge, instead got {len(output_edges)}"

#         output, dag_record = lm_dag.run_standard({"question": self.information['question']}, verbose=True)
#         answer = output["answer"]
#         score = self.metric(self.information['answer'], answer)
#         return score, dag_record


#     async def compute_and_score_attempt(self, lm_dag: LM_DAG, frozen_params: Dict = {"dag_record": None, "unfrozen_node_names": []}) -> Tuple[bool, DagRecord]:
#         assert "answer" in self.information, f"Answer should be in the information for questions with gold output, instead got {self.information}"
#         unique_inputs = list(set([edge[0][1] for edge in lm_dag.edges if edge[0][0] == "DAG_INPUT"]))
#         assert len(unique_inputs) == len(self.input_keys), f"There should be exactly one input edge, instead got {unique_inputs}"
#         assert all([input in unique_inputs for input in self.input_keys]), f"All input keys should be in the unique inputs, wanted {self.input_keys} instead got {unique_inputs}"
#         output_edges = [edge for edge in lm_dag.edges if edge[1][0] == "DAG_OUTPUT"]
#         assert len(output_edges) == 1, f"There should be exactly one output edge, instead got {len(output_edges)}"
#         output, dag_record = await lm_dag.arun({
#             k: v for k,v in self.information.items() if k in unique_inputs
#         },frozen_params=frozen_params, verbose=True)
#         answer = output["answer"]
#         score = self.metric(self.information['answer'], answer)
#         return score, dag_record
class BaseSyntheticQuestion(Question):
    information: Dict
    correctness: bool
    input_keys: List[str]
    metric: Callable

    def __init__(
        self, standardized_information: Dict, metric: Callable, input_keys: List[str]
    ):
        self.information = standardized_information
        self.correctness = None
        self.input_keys = input_keys
        self.metric = metric

    def _validate_dag(self, lm_dag: LM_DAG):
        unique_inputs = list(
            set([edge[0][1] for edge in lm_dag.edges if edge[0][0] == "DAG_INPUT"])
        )
        assert len(unique_inputs) == len(
            self.input_keys
        ), f"There should be exactly one input edge, instead got {unique_inputs}"
        assert all(
            [input in unique_inputs for input in self.input_keys]
        ), f"All input keys should be in the unique inputs, instead got {unique_inputs}"
        output_edges = [edge for edge in lm_dag.edges if edge[1][0] == "DAG_OUTPUT"]
        assert (
            len(output_edges) == 1
        ), f"There should be exactly one output edge, instead got {len(output_edges)}"

    def compute_and_score_attempt_sync(self, lm_dag: LM_DAG) -> Tuple[bool, DagRecord]:
        self._validate_dag(lm_dag)
        output, dag_record = lm_dag.run_standard(
            {"question": self.information["question"]}, verbose=True
        )
        answer = output["answer"]
        score = self._calculate_score(answer)
        return score, dag_record

    async def compute_and_score_attempt(
        self,
        lm_dag: LM_DAG,
        frozen_params: Dict = {"dag_record": None, "unfrozen_node_names": []},
    ) -> Tuple[bool, DagRecord]:
        self._validate_dag(lm_dag)
        output, dag_record = await lm_dag.arun(
            {k: v for k, v in self.information.items() if k in self.input_keys},
            frozen_params=frozen_params,
            verbose=True,
        )
        answer = output["answer"]
        score = self._calculate_score(answer)
        return score, dag_record

    def _calculate_score(self, answer):
        raise NotImplementedError("Subclasses must implement this method")


class SyntheticQuestion(BaseSyntheticQuestion):
    def _calculate_score(self, answer):
        return self.metric(answer)


class SyntheticQuestionGoldOutput(BaseSyntheticQuestion):
    def __init__(
        self, standardized_information: Dict, metric: Callable, input_keys: List[str]
    ):
        super().__init__(standardized_information, metric, input_keys)
        assert (
            "answer" in self.information
        ), f"Answer should be in the information for questions with gold output, instead got {self.information}"

    def _calculate_score(self, answer):
        return self.metric(self.information["answer"], answer)


class SyntheticBenchmark(QABenchmark):
    def __init__(
        self, dataset: List[Union[SyntheticQuestion, SyntheticQuestionGoldOutput]]
    ):
        train = dataset[: len(dataset) // 2]
        test = dataset[len(dataset) // 2 :]
        random.seed(42)
        random.shuffle(train)
        random.shuffle(test)
        self.train = [question for question in train[: len(train) // 3]]
        self.dev = [
            question for question in train[len(train) // 3 : 2 * len(train) // 3]
        ]
        self.test = [question for question in test]
        # print(
        #     "Size of train, dev, test:", len(self.train), len(self.dev), len(self.test)
        # )
