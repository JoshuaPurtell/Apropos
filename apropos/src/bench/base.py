import asyncio
import random
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Union

from apropos.src.core.programs.dag import LM_DAG, DagRecord


@dataclass
class Question:
    information: Dict
    correctness: bool

    @abstractmethod
    async def compute_and_score_attempt(self, lm_dag: LM_DAG) -> Tuple[bool, DagRecord]:
        pass

    @abstractmethod
    def compute_and_score_attempt_sync(self, lm_dag: LM_DAG) -> Tuple[bool, DagRecord]:
        pass


class Benchmark:
    train: List[Question]
    test: List[Question]
    dev: List[Question]


class QABenchmark(Benchmark):
    train: List[Question]
    test: List[Question]
    dev: List[Question]

    def get_questions_by_patches(
        self,
        questions: List[Question],
        n: int,
        patches: List[str],
        sort_type: Literal["random", "first", "last"],
    ) -> List[Question]:
        questions_by_patch = {}
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i in range(0, len(questions), 100):
            if i // 100 >= len(alphabet):
                continue
            questions_by_patch[alphabet[i // 100]] = questions[i : i + 100]

        data = []
        for patch in patches:
            if patch not in questions_by_patch:
                raise ValueError(f"Patch {patch} not found in questions")
            data.extend(questions_by_patch[patch])

        if sort_type == "first":
            return data[:n]
        elif sort_type == "last":
            return data[-n:]
        elif sort_type == "random":
            random.seed(0)
            return random.sample(data, n)

    def score_dag_parsync(
        self,
        lm_dag: LM_DAG,
        split: Literal["train", "dev", "test"] = "dev",
        n: int = 50,
        verbose: bool = False,
        sort_type: str = "first",
        patches: List[str] = ["A"],
    ) -> Union[List[bool], Tuple[List[bool], List[DagRecord]]]:
        if split == "train":
            questions = self.train
        elif split == "dev":
            questions = self.dev
        elif split == "test":
            questions = self.test
        else:
            raise ValueError("Split must be one of 'train', 'dev', or 'test'")

        questions = self.get_questions_by_patches(questions, n, patches, sort_type)

        scores, records = [], []
        # Use ThreadPoolExecutor to parallelize the scoring
        for node in lm_dag.nodes.values():
            node.transform.llm_config["multi_threaded"] = True

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(
                    lambda q: q.compute_and_score_attempt_sync(lm_dag), questions
                )
            )

        for score, record in results:
            scores.append(score)
            records.append(record)
        if verbose:
            return scores, records
        return scores

    def get_questions(
        self,
        split: Literal["train", "dev", "test"] = "dev",
        n: int = 50,
        sort_type: str = "first",
        patches: List[str] = ["A"],
    ) -> List[Question]:
        if split == "train":
            questions = self.train
        elif split == "dev":
            questions = self.dev
        elif split == "test":
            questions = self.test
        else:
            raise ValueError("Split must be one of 'train', 'dev', or 'test'")

        return self.get_questions_by_patches(questions, n, patches, sort_type)

    def score_dag_sync(
        self,
        lm_dag: LM_DAG,
        split: Literal["train", "dev", "test"] = "dev",
        n: int = 50,
        verbose: bool = False,
        sort_type: str = "first",
        patches: List[str] = ["A"],
    ) -> Union[List[bool], Tuple[List[bool], List[DagRecord]]]:
        if split == "train":
            questions = self.train
        elif split == "dev":
            questions = self.dev
        elif split == "test":
            questions = self.test
        else:
            raise ValueError("Split must be one of 'train', 'dev', or 'test'")

        questions = self.get_questions_by_patches(questions, n, patches, sort_type)
        scores, records = [], []
        for question in questions:
            score, record = question.compute_and_score_attempt_sync(lm_dag)
            scores.append(score)
            records.append(record)

        if verbose:
            return scores, records
        return scores

    async def score_dag(
        self,
        lm_dag: LM_DAG,
        split: Literal["train", "dev", "test"] = "dev",
        n: int = 50,
        frozen_params: Dict = {"dag_record": None, "unfrozen_node_names": []},
        verbose=False,
        sort_type: Literal["random", "first", "last"] = "first",
        patches: List[str] = ["A"],
    ) -> Union[List[bool], Tuple[List[bool], List[DagRecord]]]:
        if split == "train":
            questions = self.train
        elif split == "dev":
            questions = self.dev
        elif split == "test":
            questions = self.test
        else:
            raise ValueError("Split must be one of 'train', 'dev', or 'test'")

        if n // 100 + 1 > len(patches):
            raise ValueError("Number of patches must be at least the number of batches")

        questions = self.get_questions_by_patches(questions, n, patches, sort_type)

        scores = []
        records = []
        tasks = [
            question.compute_and_score_attempt(lm_dag, frozen_params=frozen_params)
            for question in questions
        ]
        score_and_records = await asyncio.gather(*tasks)
        for score, record in score_and_records:
            scores.append(score)
            records.append(record)
        if verbose:
            return scores, records
        return scores


class GameBenchmark:
    pass
