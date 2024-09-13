from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Literal
from copy import deepcopy
import random
import optuna
from apropos.src.bench.base import Benchmark, Question
from apropos.src.core.optimizers.base import DAGOptimizer
from apropos.src.core.programs.dag import LM_DAG, DagRecord
from apropos.src.core.programs.prompt import Demonstration, PromptTemplate, Topic
from pydantic import BaseModel
import concurrent.futures
import time
import os


class PromptDelta(BaseModel):
    description: str
    message: Literal["system", "user"]
    subcomponent: Literal["premise", "objective", "constraints", "user"]
    topic_name: str
    instructions_fields_edits: Dict[str, str]  # field_name, field_value
    # TODO: LATER
    # topic_template_edits: Dict[str,str]#before, after
    # input fields later ....

    def _apply_to_topic(self, topic: Topic) -> Topic:
        # if len(self.topic_template_edits) > 0:
        #     for before, after in self.topic_template_edits.items():
        #         topic.topic_template = topic.topic_template.replace(
        #             before, after
        #         )
        if len(self.instructions_fields_edits) > 0:
            for field_name, field_value in self.instructions_fields_edits.items():
                topic.instructions_fields[field_name] = field_value
        return topic

    def to_dict(self):
        return {
            "description": self.description,
            "message": self.message,
            "subcomponent": self.subcomponent,
            "topic_name": self.topic_name,
            "instructions_fields_edits": self.instructions_fields_edits,
        }

    def check_if_overlapping(self, prompt_delta: "PromptDelta") -> bool:
        return bool(
            set(self.instructions_fields_edits.keys())
            & set(prompt_delta.instructions_fields_edits.keys())
        )

    def validate(self, prompt: PromptTemplate):
        if self.message == "system":
            topics = getattr(prompt.system, self.subcomponent)
        elif self.message == "user":
            topics = getattr(prompt.user, self.subcomponent)
        else:
            raise ValueError(f"Invalid message type: {self.message}")
        for topic in topics:
            if topic.topic_name not in self.topic_name:
                raise ValueError(
                    f"Topic {self.topic_name} is not in {self.subcomponent} of prompt {self.message}"
                )

    def apply(self, prompt: PromptTemplate) -> PromptTemplate:
        if self.description == "null":
            return prompt
        prompt = deepcopy(prompt)
        if self.message == "system":
            topic_to_change, index_to_change = next(
                (
                    (topic, i)
                    for i, topic in enumerate(getattr(prompt.system, self.subcomponent))
                    if topic.topic_name == self.topic_name
                ),
                (None, -1),
            )
            if topic_to_change is None:
                raise ValueError(
                    f"Topic {self.topic_name} not found in {self.subcomponent} of prompt {self.message}"
                )
            changed_topic = self._apply_to_topic(topic_to_change)
            getattr(prompt.system, self.subcomponent)[index_to_change] = changed_topic
        elif self.message == "user":
            topic_to_change, index_to_change = next(
                (
                    (topic, i)
                    for i, topic in enumerate(prompt.user.user)
                    if topic.topic_name == self.topic_name
                ),
                (None, -1),
            )
            if topic_to_change is None:
                raise ValueError(
                    f"Topic {self.topic_name} not found in user of prompt {self.message}"
                )
            changed_topic = self._apply_to_topic(topic_to_change)
            getattr(prompt.user, self.subcomponent)[index_to_change] = changed_topic
        else:
            raise ValueError(f"Invalid message type: {self.message}")
        return prompt


@dataclass
class SearchSpace:
    prompt_delta_variations_by_problem_by_node: Dict[str, Dict[str, List[PromptDelta]]]
    demos_by_problem_by_node: Dict[str, Dict[str, List[Demonstration]]]


class RandomSearch_Optimizer:
    def __init__(
        self,
        seed: int,
        n_optuna_trials: int,
        max_demos_per_node: int = 5,
        questions_for_val: List[Question] = None,
    ):
        self.seed = seed
        self.n_optuna_trials = n_optuna_trials
        self.scored_attempts = []
        random.seed(self.seed)
        self.max_demos_per_node = max_demos_per_node
        self.questions_for_val = questions_for_val

    def search(
        self, baseline_program: LM_DAG, search_space: SearchSpace
    ) -> Tuple[LM_DAG, List[Tuple[List[Tuple[float, int]], LM_DAG]]]:
        def get_random_intervention():
            chosen_prompt_deltas_by_node = {
                node_name: [] for node_name in baseline_program.nodes.keys()
            }
            chosen_demos_by_node = {
                node_name: [] for node_name in baseline_program.nodes.keys()
            }
            for (
                problem_name,
                involved_node_names_with_deltas,
            ) in search_space.prompt_delta_variations_by_problem_by_node.items():
                for (
                    node_name,
                    prompt_delta_variations,
                ) in involved_node_names_with_deltas.items():
                    chosen_prompt_deltas_by_node[node_name].append(
                        random.choice(prompt_delta_variations)
                    )
            for problem_name, demos in search_space.demos_by_problem_by_node.items():
                for node_name, demos_for_node in demos.items():
                    chosen_demos_by_node[node_name].extend(
                        random.sample(demos_for_node, k=3)
                    )

            # Pare down any oversamples
            for node_name in chosen_demos_by_node.keys():
                chosen_demos_by_node[node_name] = random.sample(
                    chosen_demos_by_node[node_name],
                    min(len(chosen_demos_by_node[node_name]), self.max_demos_per_node),
                )
            return chosen_prompt_deltas_by_node, chosen_demos_by_node

        def to_candidate(chosen_prompt_deltas_by_node, chosen_demos_by_node):
            candidate_dag = deepcopy(baseline_program)
            for node_name, prompt_deltas in chosen_prompt_deltas_by_node.items():
                for prompt_delta in prompt_deltas:
                    candidate_dag.nodes[
                        node_name
                    ].transform.prompt = prompt_delta.apply(
                        candidate_dag.nodes[node_name].transform.prompt
                    )
            for node_name, demos in chosen_demos_by_node.items():
                candidate_dag.nodes[node_name].demonstrations = demos
            return candidate_dag

        best_score = float("-inf")
        best_params = None

        for _ in range(self.n_optuna_trials):
            chosen_prompt_deltas_by_node, chosen_demos_by_node = (
                get_random_intervention()
            )
            candidate_dag = to_candidate(
                chosen_prompt_deltas_by_node, chosen_demos_by_node
            )
            scores_with_records = [
                question.compute_and_score_attempt_sync(candidate_dag)
                for question in self.questions_for_val
            ]
            score = sum([score for score, _ in scores_with_records]) / len(
                scores_with_records
            )
            self.scored_attempts.append((score, scores_with_records, candidate_dag))

            if score > best_score:
                best_score = score
                best_params = {
                    "prompt_deltas": chosen_prompt_deltas_by_node,
                    "demos": chosen_demos_by_node,
                }
        best_candidate_dag = to_candidate(
            best_params["prompt_deltas"], best_params["demos"]
        )
        return best_candidate_dag, self.scored_attempts


class TPE_Optimizer:
    def __init__(
        self, seed: int, n_optuna_trials: int, questions_for_val: List[Question] = []
    ):
        self.seed = seed
        self.n_optuna_trials = n_optuna_trials
        self.questions_for_val = questions_for_val
        pass

    def search(
        self, baseline_program: LM_DAG, search_space: SearchSpace
    ) -> Tuple[LM_DAG, List[Tuple[List[Tuple[float, int]], LM_DAG]]]:
        def get_intervention_for_trial(trial):
            chosen_prompt_deltas_by_node = {
                node_name: [] for node_name in baseline_program.nodes.keys()
            }
            chosen_demos_by_node = {
                node_name: [] for node_name in baseline_program.nodes.keys()
            }
            for (
                problem_name,
                involved_node_names_with_deltas,
            ) in search_space.prompt_delta_variations_by_problem_by_node.items():
                for (
                    node_name,
                    prompt_delta_variations,
                ) in involved_node_names_with_deltas.items():
                    chosen_prompt_deltas_by_node[node_name].append(
                        prompt_delta_variations[
                            trial.suggest_categorical(
                                f"prompt_delta_{node_name}_{problem_name}",
                                range(len(prompt_delta_variations)),
                            )
                        ]
                    )

            for (
                problem_name,
                demos_by_node,
            ) in search_space.demos_by_problem_by_node.items():
                for node_name, demos in demos_by_node.items():
                    num_demos = min(3, len(demos))
                    for i in range(num_demos):
                        chosen_demo_index = trial.suggest_categorical(
                            f"demo_{node_name}_{problem_name}_{i}", range(len(demos))
                        )
                        chosen_demos_by_node[node_name].append(demos[chosen_demo_index])

            return chosen_prompt_deltas_by_node, chosen_demos_by_node

        def to_candidate(
            chosen_prompt_deltas_by_node: Dict[str, List[PromptDelta]],
            chosen_demos_by_node: Dict[str, List[Demonstration]],
        ):
            # import pdb; pdb.set_trace()
            assert isinstance(chosen_prompt_deltas_by_node, dict)
            assert all(
                isinstance(prompt_deltas, list)
                for prompt_deltas in chosen_prompt_deltas_by_node.values()
            )
            assert all(
                all(
                    isinstance(prompt_delta, PromptDelta)
                    for prompt_delta in prompt_deltas
                )
                for prompt_deltas in chosen_prompt_deltas_by_node.values()
            )
            assert isinstance(chosen_demos_by_node, dict)
            assert all(
                isinstance(demos, list) for demos in chosen_demos_by_node.values()
            )
            assert all(
                isinstance(demo, Demonstration)
                for demos in chosen_demos_by_node.values()
                for demo in demos
            )
            candidate_dag = deepcopy(baseline_program)
            for node_name, prompt_deltas in chosen_prompt_deltas_by_node.items():
                for prompt_delta in prompt_deltas:
                    candidate_dag.nodes[
                        node_name
                    ].transform.prompt = prompt_delta.apply(
                        candidate_dag.nodes[node_name].transform.prompt
                    )
            for node_name, demos in chosen_demos_by_node.items():
                candidate_dag.nodes[node_name].demonstrations = demos
            return candidate_dag

        scored_attempts = []

        # each trial specifies a delta for each node, including a null delta
        def create_objective() -> Callable:
            nonlocal scored_attempts

            def objective(trial):
                chosen_prompt_deltas_by_node, chosen_demos_by_node = (
                    get_intervention_for_trial(trial)
                )
                candidate_dag = to_candidate(
                    chosen_prompt_deltas_by_node, chosen_demos_by_node
                )
                print(f"Running trial {trial}")
                t0 = time.time()
                # Set the dag to parallelizable
                for node in candidate_dag.nodes.values():
                    node.transform.llm_config["multi_threaded"] = True
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max(1, os.cpu_count() - 2)
                ) as executor:
                    futures = [
                        executor.submit(
                            question.compute_and_score_attempt_sync, candidate_dag
                        )
                        for question in self.questions_for_val
                    ]
                    scores_with_records = [
                        future.result()
                        for future in concurrent.futures.as_completed(futures)
                    ]
                score = sum([score for score, _ in scores_with_records]) / len(
                    scores_with_records
                )
                print(
                    f"... got score {score} in {time.time() - t0:.2f} seconds"
                )  # Would love to know why it stalls / is so slow sometimes
                scored_attempts.append((score, scores_with_records, candidate_dag))
                return score

            return objective

        print("Beginning TPE search")
        objective = create_objective()
        sampler = optuna.samplers.TPESampler(seed=self.seed, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_optuna_trials)
        best_chosen_prompt_deltas_by_node, best_chosen_demos_by_node = (
            get_intervention_for_trial(study.best_trial)
        )
        best_candidate_dag = to_candidate(
            best_chosen_prompt_deltas_by_node, best_chosen_demos_by_node
        )
        scores = [score for score, _, _ in scored_attempts]
        print("Scores: ", scores)
        return best_candidate_dag, scored_attempts
