import asyncio
import random
from typing import Dict, List, Type

import numpy as np
import tqdm

from apropos.src.bench.base import Benchmark
from apropos.src.core.optimizers.base import DAGOptimizer
from apropos.src.core.programs.dag import LM_DAG
from apropos.src.core.programs.prompt import Demonstration

random.seed(42)


class BreadthFirstRandomSearch_DAG(DAGOptimizer):
    student_program: LM_DAG
    teacher_program: LM_DAG
    dataset_handler: Type[Benchmark]

    def __init__(
        self,
        student_program: LM_DAG,
        dataset_handler: Type[Benchmark],
        teacher_program: LM_DAG = None,
        cfg: Dict = None,
    ):
        self.student_program = student_program
        self.dataset_handler = dataset_handler
        self.teacher_program = teacher_program or student_program
        self.cfg = cfg or {
            "optimization": {
                "combination_size": 5,
                "n_iterations": 100,
                "program_search_parallelization_factor": 10,
                "validation_size": 30,
                "test_size": 100,
            },
            "bootstrapping": {
                "patches": ["A", "B"],
                "n_questions": 50,
            },
            "verbose": True,
        }

    async def evaluate_combination_of_demonstrations(
        self,
        demonstrations_by_node_name: Dict[str, List[Demonstration]],
    ):
        candidate = self.get_fewshot_program(demonstrations_by_node_name)
        validation_size = self.cfg["optimization"]["validation_size"]
        patches = ["A"] if validation_size < 100 else ["A", "B"]
        scores, dag_records = await self.dataset_handler.score_dag(
            candidate, n=validation_size, verbose=True, split="dev", patches=patches
        )
        return scores, dag_records

    async def propose_combination_of_demonstrations(
        self, demonstrations_by_stage: List[List[str]], n_combinations_to_sample=10
    ):
        random.seed(42)
        combinations: List[List[Demonstration]] = [[]]
        combination_size = self.cfg["optimization"]["combination_size"]
        for _ in range(n_combinations_to_sample):
            combination = []
            for stage in demonstrations_by_stage:
                combination += random.sample(stage, combination_size)
            combinations.append(combination)
        return combinations

    async def evaluate_combinations(self, combinations_by_name):
        scores_by_combo = []
        n_iterations = self.cfg["optimization"]["n_iterations"]
        program_search_parallelization_factor = self.cfg["optimization"][
            "program_search_parallelization_factor"
        ]
        assert (
            program_search_parallelization_factor > 0
        ), "Program search parallelization factor must be greater than 0."
        assert (
            program_search_parallelization_factor < n_iterations
        ), "Program search parallelization factor must be less than the number of iterations."
        for index in tqdm.tqdm(
            range(0, n_iterations, program_search_parallelization_factor)
        ):
            chunk_combinations = {
                name: combinations[
                    index : (index + program_search_parallelization_factor)
                ]
                for name, combinations in combinations_by_name.items()
            }
            tasks = []
            for j in range(
                np.min([len(chunk_combinations[k]) for k in chunk_combinations])
            ):
                tasks.append(
                    self.evaluate_combination_of_demonstrations(
                        {
                            node_name: chunk_combinations[node_name][j]
                            for node_name in chunk_combinations
                        },
                    )
                )
            results = await asyncio.gather(*tasks)
            scores = [score for score, _ in results]
            scores_by_combo.extend(scores)
        return scores_by_combo

    async def optimize_demonstrations(self) -> LM_DAG:
        if self.cfg["verbose"]:
            print("BFRS - Bootstrapping demonstrations...")

        bootstrapped_demonstrations: Dict[
            str, List[Demonstration]
        ] = await self.bootstrap_demonstrations(
            n=self.cfg["bootstrapping"]["n_questions"],
            patches=self.cfg["bootstrapping"]["patches"],
        )
        combinations_by_name = {name: [] for name in self.student_program.nodes}
        combination_size = self.cfg["optimization"]["combination_size"]
        n_iterations = self.cfg["optimization"]["n_iterations"]
        for name, demos in bootstrapped_demonstrations.items():
            assert (
                len(demos) >= combination_size
            ), f"Too few demonstrations for bootstrapping - used {len(demos)} inputs for combo size {combination_size}."
            n_combinations = [
                random.sample(demos, combination_size) for _ in range(n_iterations)
            ]
            combinations_by_name[name] = n_combinations
        print("...done - Evaluating combinations...")
        scores_by_combo = await self.evaluate_combinations(combinations_by_name)
        best_combination_index = scores_by_combo.index(max(scores_by_combo))

        best_combination = {
            name: combinations_by_name[name][best_combination_index]
            for name in self.student_program.nodes
        }
        print("Best combination: ", best_combination_index)
        best_program = self.get_fewshot_program(best_combination)

        return best_program
