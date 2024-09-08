import ast
import random
import os
import pickle
from typing import Dict, List, Literal, Tuple

from apropos.src.bench.base import QABenchmark, Question
from apropos.src.bench.bigcodebench.backends.danger import execute_code_locally
from apropos.src.bench.bigcodebench.backends.docker import execute_code_remotely_docker
from apropos.src.bench.bigcodebench.backends.modal import execute_code_remotely_modal
from apropos.src.core.programs.dag import LM_DAG, DagRecord
from datasets import load_dataset

# Connect to modal


def map_to_topic(libs: List[str]) -> str:
    libs_by_topic = {
        "File and Data Management": ["os", "shutil", "glob", "csv", "pathlib"],
        "Web and Network Operations": [
            "requests",
            "urllib",
            "cgi",
            "http.server",
            "ipaddress",
        ],
        "Data Science and Analysis": [
            "pandas",
            "numpy",
            "scipy",
            "matplotlib",
            "seaborn",
            "sklearn",
            "statsmodels",
        ],
        "Database Operations": ["sqlite3"],
        "Datetime and Timezone Handling": ["datetime", "pytz"],
        "Text and String Processing": ["re"],
        "Encryption and Security": ["hashlib", "cryptography", "rsa"],
        "Image and Video Processing": ["cv2", "sklearn"],
        "JSON and Serialization": ["json", "yaml"],
        "Mathematical and Numerical Operations": ["math"],
        "Randomness and Probability": ["random"],
        "File Formats and Data Exporting": ["csv", "json", "base64"],
    }
    # Count occurrences of each topic
    topic_counts = {}
    for topic, topic_libs in libs_by_topic.items():
        count = sum(1 for lib in libs if lib in topic_libs)
        if count > 0:
            topic_counts[topic] = count

    # Find the topic(s) with the highest count
    if not topic_counts:
        return "Uncategorized"

    max_count = max(topic_counts.values())
    top_topics = [topic for topic, count in topic_counts.items() if count == max_count]

    # If there's only one top topic, return it
    if len(top_topics) == 1:
        return top_topics[0]

    # If there's a tie, randomly choose one
    return random.choice(top_topics)


def standardize_bcbc_question(hf_dict: Dict) -> Dict:
    question = hf_dict["complete_prompt"]
    answer = hf_dict["canonical_solution"]
    reasoning = answer
    return {
        "question": question,
        "answer": answer,
        "gold_reasoning": reasoning,
        "eval_info": {"code_prompt": hf_dict["code_prompt"], "test": hf_dict["test"]},
        "topic": map_to_topic(ast.literal_eval(hf_dict["libs"])),
    }


def composite_code_metric(correctness, result_dict):
    return correctness and result_dict["testsRun"] > 0


def strip_out_code(answer):
    if "```python" in answer:
        return answer.split("```python")[1].split("```")[0]
    elif "```" in answer:
        return answer.split("```")[1].split("```")[0]
    else:
        return answer


class BigCodeBench_Question(Question):
    information: Dict
    correctness: bool
    mode: Literal["local", "modal", "docker"] = "local"

    def __init__(
        self,
        standardized_information: Dict,
        mode: Literal["local", "modal", "docker"] = "docker",
    ):
        self.information = standardized_information
        self.correctness = None
        self.mode = mode

    def compute_and_score_attempt_sync(self, lm_dag: LM_DAG) -> Tuple[bool, DagRecord]:
        unique_inputs = list(
            set([edge[0][1] for edge in lm_dag.edges if edge[0][0] == "DAG_INPUT"])
        )
        assert (
            len(unique_inputs) == 1
        ), f"There should be exactly one input edge, instead got {unique_inputs}"
        assert (
            unique_inputs[0] == "<<<CODING_QUESTION>>>"
        ), f"The input edge should be for the question, instead got {unique_inputs[0]}"
        output_edges = [edge for edge in lm_dag.edges if edge[1][0] == "DAG_OUTPUT"]
        assert (
            len(output_edges) == 1
        ), f"There should be exactly one output edge, instead got {len(output_edges)}"
        output, dag_record = lm_dag.run_standard(
            {"question": self.information["question"]}, verbose=True
        )
        answer = output["answer"]
        answer = strip_out_code(answer)
        if self.mode == "local":
            correctness, result_dict = execute_code_locally(self.information, answer)
            score = composite_code_metric(correctness, result_dict)
        elif self.mode == "docker":
            correctness, result_dict = asyncio.run(
                execute_code_remotely_docker(self.information, answer)
            )
            score = composite_code_metric(correctness, result_dict)
        elif self.mode == "modal":
            correctness, result_dict = asyncio.run(
                execute_code_remotely(self.information, answer)
            )
            score = composite_code_metric(correctness, result_dict)
        else:
            raise ValueError("Invalid mode")
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
            unique_inputs[0] == "<<<CODING_QUESTION>>>"
        ), f"The input edge should be for the question, instead got {unique_inputs[0]}"
        output_edges = [edge for edge in lm_dag.edges if edge[1][0] == "DAG_OUTPUT"]
        assert (
            len(output_edges) == 1
        ), f"There should be exactly one output edge, instead got {len(output_edges)}"
        output, dag_record = await lm_dag.arun(
            {"question": self.information["question"]}, verbose=True
        )
        answer = output["answer"]
        answer = strip_out_code(answer)
        if self.mode == "local":
            raise NotImplementedError("Local mode not safe - use docker or modal")
            correctness, result_dict = execute_code_locally(self.information, answer)
            score = composite_code_metric(correctness, result_dict)
        elif self.mode == "docker":
            correctness, result_dict = await execute_code_remotely_docker(
                self.information, answer
            )
            score = composite_code_metric(correctness, result_dict)
        elif self.mode == "modal":
            correctness, result_dict = await execute_code_remotely_modal(
                self.information, answer
            )
            score = composite_code_metric(correctness, result_dict)
        else:
            raise ValueError("Invalid mode")
        return score, dag_record


class BigCodeBenchComplete_Benchmark(QABenchmark):
    def __init__(self, mode: Literal["local", "remote"] = "remote"):
        cache_dir = "datasets/bigcodebench"
        cache_file = os.path.join(cache_dir, "cached_questions.pkl")

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                train, test = pickle.load(f)
        else:
            ds = load_dataset("bigcode/bigcodebench", "default")
            train_test_split = ds["v0.1.0_hf"].train_test_split(test_size=0.5, seed=42)
            train = [standardize_bcbc_question(q) for q in train_test_split["train"]]
            test = [standardize_bcbc_question(q) for q in train_test_split["test"]]

            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump((train, test), f)

        random.seed(42)
        random.shuffle(train)
        random.shuffle(test)

        self.train = [
            BigCodeBench_Question(info, mode=mode) for info in train[: len(train) // 3]
        ]
        self.dev = [
            BigCodeBench_Question(info, mode=mode) for info in train[len(train) // 3 :]
        ]
        self.test = [BigCodeBench_Question(info, mode=mode) for info in test]


async def test_gold_on_split(split: str = "train"):
    bcb = BigCodeBenchComplete_Benchmark(mode="docker")
    if split == "train":
        questions = bcb.train
    elif split == "dev":
        questions = bcb.dev
    elif split == "test":
        questions = bcb.test

    async def score_gold(question: BigCodeBench_Question):
        answer = question.information["answer"]
        correctness, result_dict = await execute_code_remotely_docker(
            question.information, answer
        )
        return correctness, result_dict

    gold_scores = await asyncio.gather(
        *[score_gold(question) for question in questions]
    )
    gold_scores = [score for score, _ in gold_scores]
    import numpy as np

    print(f"Gold scores for {split}: {gold_scores}")
    print(f"Mean gold score for {split}: {np.mean(gold_scores)}")
    print(f"Num correct for {split}: {np.sum(gold_scores)}")
    print(f"Num total for {split}: {len(gold_scores)}")


if __name__ == "__main__":
    import asyncio

    bcb = BigCodeBenchComplete_Benchmark(mode="docker")
    print("Size of train, dev, test:", len(bcb.train), len(bcb.dev), len(bcb.test))
    from apropos.src.bench.bigcodebench.single_step_dag import code_problem_single_step
    # asyncio.run(test_gold_on_split(split="train"))

    dag = code_problem_single_step(model_name="gemini-1.5-flash-latest")
    import time
    import numpy as np

    t0 = time.time()
    scores = asyncio.run(bcb.score_dag(dag, n=100, patches=["A", "B"]))
    t1 = time.time()
    print("Time taken:", t1 - t0)
    print("Scores:", np.mean(scores))
