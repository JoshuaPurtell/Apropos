from typing import Dict, Tuple

from apropos.src.bench.base import QABenchmark, Question
from apropos.src.core.programs.dag import LM_DAG, DagRecord
from datasets import load_dataset


def extract_numeric(text: str):
    def find_numbers(text):
        return [
            int(num.replace(",", "")) for num in re.findall(r"\b\d+(?:,\d{3})*\b", text)
        ]

    import re

    for paragraph in text.split("\n")[::-1]:
        for line in paragraph.split(".")[::-1]:
            numbers = find_numbers(line)
            if len(numbers) > 0:
                return str(numbers[-1])
    return None


def josh_gsm8k_metric(gold, pred, trace=None):
    if pred is None:
        return False
    numeric = extract_numeric(pred)
    solution = extract_numeric(gold)
    if numeric is None:
        return False
    else:
        clean_numeric = numeric.replace(",", "")
        return int(clean_numeric) == int(solution)


def standardize_gsm8k_question(hf_dict):
    question = hf_dict["question"]

    answer = hf_dict["answer"].strip().split()
    assert answer[-2] == "####"

    gold_reasoning = " ".join(answer[:-2])
    answer = str(int(answer[-1].replace(",", "")))

    return {"question": question, "gold_reasoning": gold_reasoning, "answer": answer}


class GSM8k_Question(Question):
    information: Dict

    def __init__(self, standardized_information: Dict):
        self.information = standardized_information

    def compute_and_score_attempt_sync(self, lm_dag) -> Tuple[bool, DagRecord]:
        unique_inputs = list(
            set([edge[0][1] for edge in lm_dag.edges if edge[0][0] == "DAG_INPUT"])
        )
        assert (
            len(unique_inputs) == 1
        ), f"There should be exactly one input edge, instead got {unique_inputs}"
        assert (
            unique_inputs[0] == "<<<MATHEMATICS_QUESTION>>>"
        ), f"The input edge should be for the question, instead got {unique_inputs[0]}"
        output_edges = [edge for edge in lm_dag.edges if edge[1][0] == "DAG_OUTPUT"]
        assert (
            len(output_edges) == 1
        ), f"There should be exactly one output edge, instead got {len(output_edges)}"
        assert (
            output_edges[0][1][1] == f"answer"
        ), "The output edge should be for the answer, instead got {output_edges[0][1][1]}"

        output, dag_record = lm_dag.run_standard(
            {"question": self.information["question"]}, verbose=True
        )
        answer = output["answer"]
        score = josh_gsm8k_metric(self.information["answer"], answer)
        return score, dag_record

    async def compute_and_score_attempt(
        self,
        lm_dag: LM_DAG,
        frozen_params: Dict = {"dag_record": None, "unfrozen_node_names": []},
    ) -> Tuple[bool, DagRecord]:
        unique_inputs = list(
            set([edge[0][1] for edge in lm_dag.edges if edge[0][0] == "DAG_INPUT"])
        )
        assert (
            len(unique_inputs) == 1
        ), f"There should be exactly one input edge, instead got {unique_inputs}"
        assert (
            unique_inputs[0] == "<<<MATHEMATICS_QUESTION>>>"
        ), f"The input edge should be for the question, instead got {unique_inputs[0]}"
        output_edges = [edge for edge in lm_dag.edges if edge[1][0] == "DAG_OUTPUT"]
        assert (
            len(output_edges) == 1
        ), f"There should be exactly one output edge, instead got {len(output_edges)}"
        assert (
            output_edges[0][1][1] == f"answer"
        ), "The output edge should be for the answer, instead got {output_edges[0][1][1]}"

        output, dag_record = await lm_dag.arun(
            {"question": self.information["question"]},
            frozen_params=frozen_params,
            verbose=True,
        )
        answer = output["answer"]
        score = josh_gsm8k_metric(self.information["answer"], answer)
        return score, dag_record


class GSM8k_Benchmark(QABenchmark):
    def __init__(self):
        super().__init__()
        dataset = load_dataset("gsm8k", "main")
        standardized_train = [
            standardize_gsm8k_question(hf_dict) for hf_dict in dataset["train"]
        ]
        standardized_test = [
            standardize_gsm8k_question(hf_dict) for hf_dict in dataset["test"]
        ]

        self.train = [
            GSM8k_Question(info)
            for info in standardized_train[: len(standardized_train) // 2]
        ]
        self.dev = [
            GSM8k_Question(info)
            for info in standardized_train[len(standardized_train) // 2 :]
        ]
        self.test = [GSM8k_Question(info) for info in standardized_test]
