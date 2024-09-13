import json
import os
import random
import re
from typing import Dict, Tuple

from apropos.src.bench.base import QABenchmark, Question
from apropos.src.core.programs.dag import LM_DAG, DagRecord
from datasets import load_dataset

random.seed(42)


def extract_boxed(text: str):
    match = re.search(r"\\boxed{((?:[^{}]|{[^{}]*})*)}", text)
    if match:
        return match.group(1)
    return None


def standardize_hendryks_question(hf_dict):
    question = hf_dict["problem"]
    reasoning_with_answer = hf_dict["solution"]
    answer = extract_boxed(reasoning_with_answer)
    if not answer:
        return None
    reasoning = reasoning_with_answer
    return {
        "question": question,
        "answer": answer,
        "gold_reasoning": reasoning,
        "topic": hf_dict["type"],
    }


def custom_math_metric(gold, pred, trace=None):
    extracted_pred = extract_boxed(pred)
    if extracted_pred is None:
        return False
    else:
        return gold == extracted_pred


class MATH_Question(Question):
    information: Dict
    correctness: bool

    def __init__(self, standardized_information: Dict):
        self.information = standardized_information
        self.correctness = None

    def compute_and_score_attempt_sync(self, lm_dag: LM_DAG) -> Tuple[bool, DagRecord]:
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

        output, dag_record = lm_dag.run_standard(
            {"question": self.information["question"]}, verbose=True
        )
        answer = output["answer"]
        score = custom_math_metric(self.information["answer"], answer)
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

        output, dag_record = await lm_dag.arun(
            {"question": self.information["question"]},
            frozen_params=frozen_params,
            verbose=True,
        )
        answer = output["answer"]
        score = custom_math_metric(self.information["answer"], answer)
        return score, dag_record


class HendryksMath_Benchmark(QABenchmark):
    def __init__(self):
        if not os.path.exists("datasets/competition_math/dataset.json"):
            dataset = load_dataset("competition_math", "main")
            os.makedirs("datasets/competition_math", exist_ok=True)
            with open("datasets/competition_math/dataset.json", "w") as f:
                json.dump(
                    {"train": list(dataset["train"]), "test": list(dataset["test"])}, f
                )
        else:
            with open("datasets/competition_math/dataset.json", "r") as f:
                dataset = json.load(f)

        train = [
            standardize_hendryks_question(hf_dict) for hf_dict in list(dataset["train"])
        ]
        test = [
            standardize_hendryks_question(hf_dict) for hf_dict in list(dataset["test"])
        ]
        train = [
            {
                "question": info["question"],
                "answer": info["answer"],
                "gold_reasoning": info["gold_reasoning"],
                "topic": info["topic"],
            }
            for info in train
            if info and info["topic"]
        ]
        test = [
            {
                "question": info["question"],
                "answer": info["answer"],
                "gold_reasoning": info["gold_reasoning"],
                "topic": info["topic"],
            }
            for info in test
            if info
        ]

        random.seed(42)
        random.shuffle(train)
        random.shuffle(test)
        self.train = [MATH_Question(info) for info in train[: len(train) // 3]]
        self.dev = [
            MATH_Question(info) for info in train[len(train) // 3 : 2 * len(train) // 3]
        ]
        self.test = [MATH_Question(info) for info in test]

        # print(
        #     "Size of train, dev, test:", len(self.train), len(self.dev), len(self.test)
        # )


if __name__ == "__main__":
    math = MATH_Benchmark()
