import asyncio
import random
from copy import deepcopy
from typing import Dict, List, Literal, Tuple, Type

from tqdm import tqdm

from apropos.src.bench.base import Benchmark, Question
from apropos.src.core.lms.helpers import LLM
from apropos.src.core.optimizers.base import DAGOptimizer
from apropos.src.core.optimizers.miprov2p1.instruction_gen_promtps import (
    InterventionConceptsResponse,
    ProblemsResponse,
    PromptDeltasResponse,
    PromptDeltasVariationsResponse,
    QuestionCurationResponse,
    get_intervention_concepts_prompt,
    get_prompt_delta_prompt,
    get_prompt_delta_variations_prompt,
    get_simple_prompt_delta_prompt,
    identify_problems_prompt,
    question_curation_prompt,
)
from apropos.src.core.optimizers.miprov2p1.search_backends import (
    PromptDelta,
    RandomSearch_Optimizer,
    SearchSpace,
    TPE_Optimizer,
)
from apropos.src.core.programs.dag import LM_DAG, DagRecord
from apropos.src.core.programs.prompt import Demonstration

random.seed(42)

# Core idea behind MIPRO-v2: test variations for solving the same problem


class MIPrO_V2p1_DAG(DAGOptimizer):
    student_program: LM_DAG
    teacher_program: LM_DAG
    dataset_handler: Type[Benchmark]
    search_space: SearchSpace

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
        self.search_space = SearchSpace(
            prompt_delta_variations_by_problem_by_node={},
            demos_by_problem_by_node={},
        )
        self.cfg = cfg

    def curate_questions_for_bootstrapping(
        self,
        learnable_questions: List[Question],
        successes: List[List[DagRecord]],
        failures: List[List[DagRecord]],
        pass_at_ks: List[float],
    ) -> Tuple[List[Question], List[Question], List[Question]]:
        question_curation_response: QuestionCurationResponse = (
            question_curation_prompt.run(
                inputs={
                    "<<<HIGHLY_LEARNABLE_QUESTIONS>>>": "\n".join(
                        [
                            f"### Question {i}\n{question}"
                            for i, question in enumerate(learnable_questions, 1)
                        ]
                    ),
                    "<<<SUCCESS_RATES>>>": str(pass_at_ks),
                    "<<<SUCCESS_TRACES>>>": str(
                        [random.choice(successes)] if successes else []
                    ),
                    "<<<FAILURE_TRACES>>>": str(
                        [random.choice(failures)] if failures else []
                    ),
                },
                lm=LLM(model_name="gpt-4o-mini"),
                response_model=QuestionCurationResponse,
            )
        )
        # shoudl be pydantic?
        # See what happens?
        demo_questions = [
            learnable_questions[i]
            for i in question_curation_response.bootstrapping_question_indices
            if i in range(len(learnable_questions))
        ]
        train_questions = [
            learnable_questions[i]
            for i in question_curation_response.train_question_indices
            if i in range(len(learnable_questions))
        ]
        val_questions = [
            learnable_questions[i]
            for i in question_curation_response.val_question_indices
            if i in range(len(learnable_questions))
        ]

        return demo_questions, train_questions, val_questions

    async def bootstrap_demonstrations_on_subset(self, questions: List[Question]):
        true_train = self.dataset_handler.train
        self.dataset_handler.train = questions
        if len(questions) > 100:
            patches = [chr(65 + i) for i in range(len(questions) // 100 + 1)]
        else:
            patches = ["A"]
        demos = await self.bootstrap_demonstrations(n=len(questions), patches=patches)
        self.dataset_handler.train = true_train
        return demos

    async def get_learnable_questions(
        self,
    ) -> Tuple[
        List[Question], List[List[DagRecord]], List[List[DagRecord]], List[float]
    ]:
        import asyncio

        from tqdm import tqdm

        async def process_question(question_index):
            question = self.dataset_handler.train[question_index]
            temp_schedule = [
                self.cfg["learnable_questions"]["base_temp"] + 0.01 * i
                for i in range(self.cfg["learnable_questions"]["k_for_pass_at_k"])
            ]
            temp_scheduled_programs = [
                deepcopy(self.student_program)
                for _ in range(self.cfg["learnable_questions"]["k_for_pass_at_k"])
            ]

            for program, temperature in zip(temp_scheduled_programs, temp_schedule):
                for node in program.nodes.values():
                    node.transform.llm_config["temperature"] = temperature

            correctnesses_with_records = await asyncio.gather(
                *[
                    question.compute_and_score_attempt(program)
                    for program in temp_scheduled_programs
                ]
            )
            correctnesses = [
                correctness for correctness, _ in correctnesses_with_records
            ]

            if sum(correctnesses) not in [0, len(correctnesses)]:
                return (
                    question,
                    [
                        record
                        for correctness, record in correctnesses_with_records
                        if correctness
                    ],
                    [
                        record
                        for correctness, record in correctnesses_with_records
                        if not correctness
                    ],
                    sum(correctnesses) / len(correctnesses),
                )
            return None

        max_questions = self.cfg["learnable_questions"]["max_n_to_sample"]
        max_to_obtain = self.cfg["learnable_questions"]["max_n_to_obtain"]

        results = []
        batch_size = min(max_to_obtain, 20)
        for batch_start in tqdm(
            range(0, max_questions, batch_size), desc="Sampling questions for batch"
        ):
            batch_end = min(batch_start + batch_size, max_questions)
            batch_results = await asyncio.gather(
                *[process_question(i) for i in range(batch_start, batch_end)]
            )
            results.extend([r for r in batch_results if r is not None])
            if len(results) >= max_to_obtain:
                results = results[:max_to_obtain]
                break

        learnable_questions, successes, failures, pass_at_ks = zip(*results)
        return (
            list(learnable_questions),
            list(successes),
            list(failures),
            list(pass_at_ks),
        )

    async def propose_prompt_instruction_components_simple_async(
        self,
        learnable_questions: List[Question],
        successes: List[List[DagRecord]],
        failures: List[List[DagRecord]],
        pass_at_ks: List[float],
        bootstrapped_demos_by_node: Dict[str, Dict[str, List[Demonstration]]],
    ):
        # Get simple prompt deltas
        program_string = str(
            {
                node_name: self.student_program.to_dict()["nodes"][node_name][
                    "runnables"
                ]["transform"]["prompt"]
                for node_name in self.student_program.nodes.keys()
            }
        )
        valid_topics_by_subcomponent_by_node = {}
        unique_subcomponents_by_message = {
            "system": ["premise", "objective", "constraints"],
            "user": ["user"],
        }
        nodes_dictified = self.student_program.to_dict()["nodes"]
        for node_name in nodes_dictified:
            prompt = nodes_dictified[node_name]["runnables"]["transform"]["prompt"]
            for message in unique_subcomponents_by_message:
                for subcomponent in unique_subcomponents_by_message[message]:
                    valid_topics_by_subcomponent_by_node[(message, subcomponent)] = [
                        t["topic_name"] for t in prompt[message][subcomponent]
                    ]

        def process_delta_response(
            prompt_delta_response: PromptDeltasResponse,
        ) -> Dict[str, List[PromptDelta]]:
            assert all(
                prompt_delta_response.subcomponents[i]
                in ["premise", "objective", "constraints", "user"]
                for i in range(len(prompt_delta_response.subcomponents))
            )
            core_prompt_deltas_by_node = {}
            for i in range(len(prompt_delta_response.descriptions)):
                core_prompt_delta = PromptDelta(
                    description=prompt_delta_response.descriptions[i],
                    message=prompt_delta_response.messages[i],
                    subcomponent=prompt_delta_response.subcomponents[i],
                    topic_name=prompt_delta_response.topic_names[i],
                    instructions_fields_edits=prompt_delta_response.instructions_fields_edits[
                        i
                    ],
                )
                if (
                    not prompt_delta_response.node_names[i]
                    in core_prompt_deltas_by_node
                ):
                    core_prompt_deltas_by_node[prompt_delta_response.node_names[i]] = []
                core_prompt_deltas_by_node[prompt_delta_response.node_names[i]].append(
                    core_prompt_delta
                )

            for (
                node_name,
                core_prompt_deltas,
            ) in core_prompt_deltas_by_node.items():  # ["prompt_delta"]
                for core_prompt_delta in core_prompt_deltas:
                    core_prompt_delta.validate(
                        self.student_program.nodes[node_name].transform.prompt
                    )
                    core_prompt_delta.apply(
                        self.student_program.nodes[node_name].transform.prompt
                    )
            return core_prompt_deltas_by_node

        deltas_response: PromptDeltasResponse = (
            await get_simple_prompt_delta_prompt.arun(
                inputs={
                    "<<<HIGHLY_LEARNABLE_QUESTIONS>>>": "\n".join(
                        [
                            f"### Question {i}\n{question}"
                            for i, question in enumerate(learnable_questions, 1)
                        ]
                    ),
                    "<<<SUCCESS_RATES>>>": str(pass_at_ks),
                    "<<<SUCCESS_TRACES>>>": str(
                        [random.choice(successes)] if successes else []
                    ),
                    "<<<FAILURE_TRACES>>>": str(
                        [random.choice(failures)] if failures else []
                    ),
                    "<<<NODE_NAMES>>>": str(list(self.student_program.nodes.keys())),
                    "<<<CURRENT_PROGRAM>>>": program_string,
                    "<<<VALID_TOPICS_BY_SUBCOMPONENT_BY_NODE>>>": str(
                        valid_topics_by_subcomponent_by_node
                    ),
                },
                lm=LLM(model_name="gpt-4o-mini"),
                response_model=PromptDeltasResponse,
            )
        )
        core_prompt_deltas = process_delta_response(deltas_response)
        return {"General": core_prompt_deltas}, {"General": bootstrapped_demos_by_node}

    # Propose prompt instruction components (Simple => Plan Search)
    async def propose_prompt_instruction_components_complex_async(
        self,
        learnable_questions: List[Question],
        successes: List[List[DagRecord]],
        failures: List[List[DagRecord]],
        pass_at_ks: List[float],
        bootstrapped_demos_by_node: Dict[str, Dict[str, List[Demonstration]]],
    ):
        random.seed(42)
        # # changes can either be to the template + adding/removing instructions (later)
        # Stringify these
        problems_response: ProblemsResponse = await identify_problems_prompt.arun(
            inputs={
                "<<<HIGHLY_LEARNABLE_QUESTIONS>>>": "\n".join(
                    [
                        f"### Question {i}\n{question}"
                        for i, question in enumerate(learnable_questions, 1)
                    ]
                ),
                "<<<SUCCESS_RATES>>>": str(pass_at_ks),
                "<<<SUCCESS_TRACES>>>": str(
                    [random.choice(successes)] if successes else []
                ),
                "<<<FAILURE_TRACES>>>": str(
                    [random.choice(failures)] if failures else []
                ),
                "<<<NODE_NAMES>>>": str(list(self.student_program.nodes.keys())),
            },
            lm=LLM(model_name="gpt-4o-mini"),
            response_model=ProblemsResponse,
        )
        nodes_responsible_by_name = [
            node
            for sublist in problems_response.nodes_responsible_by_name
            for node in sublist
        ]
        assert set(
            nodes_responsible_by_name
        ).issubset(
            set(self.student_program.nodes.keys())
        ), f"Nodes responsible by name: {nodes_responsible_by_name} not found in student program: {self.student_program.nodes.keys()}"
        assert (
            min(
                [
                    len(examples)
                    for examples in problems_response.example_question_indices
                ]
            )
            >= 1
        ), f"Example question indices: {problems_response.example_question_indices} have less than 1 example question"

        # Convert into a list of problems, write an intervention for each via concept -> delta
        problems = [
            {
                "problem_name": problems_response.problem_names[i],
                "problem_description": problems_response.problem_descriptions[i],
                "example_questions": [
                    learnable_questions[j]
                    for j in problems_response.example_question_indices[i]
                    if j in range(len(learnable_questions))
                ],
                "nodes_responsible_by_name": problems_response.nodes_responsible_by_name[
                    i
                ],
            }
            for i in range(len(problems_response.problem_descriptions))
        ]
        deltas_by_problem_by_node = {}
        demos_by_problem_by_node = {}
        print("Getting intervention concepts for {} problems".format(len(problems)))
        for problem in problems:
            deltas_by_problem_by_node[problem["problem_name"]] = {}
            demos_by_problem_by_node[problem["problem_name"]] = {}
            print("... Getting intervention concepts")
            intervention_concepts_response: InterventionConceptsResponse = (
                await get_intervention_concepts_prompt.arun(
                    inputs={
                        "<<<PROGRAM>>>": str(self.student_program.to_dict()),
                        "<<<VALID_TOPICS_BY_SUBCOMPONENT_BY_NODE>>>": str(
                            valid_topics_by_subcomponent_by_node
                        ),
                    },
                    lm=LLM(model_name="gpt-4o-mini"),
                    response_model=InterventionConceptsResponse,
                )
            )
            print("... Getting prompt deltas")
            program_string = str(
                {
                    node_name: self.student_program.to_dict()["nodes"][node_name][
                        "runnables"
                    ]["transform"]["prompt"]
                    for node_name in problem["nodes_responsible_by_name"]
                }
            )

            # import pdb; pdb.set_trace()
            valid_topics_by_subcomponent_by_node = {}
            unique_subcomponents_by_message = {
                "system": ["premise", "objective", "constraints"],
                "user": ["user"],
            }
            nodes_dictified = self.student_program.to_dict()["nodes"]
            for node_name in nodes_dictified:
                prompt = nodes_dictified[node_name]["runnables"]["transform"]["prompt"]
                for message in unique_subcomponents_by_message:
                    for subcomponent in unique_subcomponents_by_message[message]:
                        valid_topics_by_subcomponent_by_node[
                            (message, subcomponent)
                        ] = [t["topic_name"] for t in prompt[message][subcomponent]]

            async def get_prompt_delta_response(temperature: float):
                prompt_delta_response: PromptDeltasResponse = await get_prompt_delta_prompt.arun(
                    inputs={
                        "<<<INTERVENTION_CONCEPTS>>>": str(
                            intervention_concepts_response.intervention_concepts_by_node
                        ),
                        "<<<CURRENT_PROGRAM>>>": program_string,
                        "<<<VALID_TOPICS_BY_SUBCOMPONENT_BY_NODE>>>": str(
                            valid_topics_by_subcomponent_by_node
                        ),
                        "<<<FAILURE_DESCRIPTION>>>": problem["problem_description"],
                    },
                    lm=LLM(model_name="gpt-4o-mini", temperature=temperature),
                    response_model=PromptDeltasResponse,
                )
                return prompt_delta_response

            def process_delta_response(
                prompt_delta_response: PromptDeltasResponse,
            ) -> List[Dict[str, Any]]:
                assert all(
                    prompt_delta_response.subcomponents[i]
                    in ["premise", "objective", "constraints", "user"]
                    for i in range(len(prompt_delta_response.subcomponents))
                )
                core_prompt_deltas = [
                    {
                        "node_name": prompt_delta_response.node_names[i],
                        "prompt_delta": PromptDelta(
                            description=prompt_delta_response.descriptions[i],
                            message=prompt_delta_response.messages[i],
                            subcomponent=prompt_delta_response.subcomponents[i],
                            topic_name=prompt_delta_response.topic_names[i],
                            instructions_fields_edits=prompt_delta_response.instructions_fields_edits[
                                i
                            ],
                        ),
                    }
                    for i in range(len(prompt_delta_response.descriptions))
                ]
                for core_prompt_delta in core_prompt_deltas:
                    core_prompt_delta["prompt_delta"].validate(
                        self.student_program.nodes[
                            core_prompt_delta["node_name"]
                        ].transform.prompt
                    )
                    core_prompt_delta["prompt_delta"].apply(
                        self.student_program.nodes[
                            core_prompt_delta["node_name"]
                        ].transform.prompt
                    )
                return core_prompt_deltas

            got_valid_delta = False
            temp_schedule_for_tries = [0, 0.1]
            temp_schedule_index = 0
            while not got_valid_delta and temp_schedule_index < len(
                temp_schedule_for_tries
            ):
                try:
                    prompt_delta_response = await get_prompt_delta_response(
                        temperature=temp_schedule_for_tries[temp_schedule_index]
                    )
                    core_prompt_deltas = process_delta_response(prompt_delta_response)
                    got_valid_delta = True
                except Exception as e:
                    print(
                        f"Error: {e}, retrying with temperature {temp_schedule_for_tries[temp_schedule_index + 1]}"
                    )
                    temp_schedule_index += 1
            if not got_valid_delta:
                raise ValueError("Could not get valid delta")
            # Make sure these are valid

            prompt_delta_variations = [[] for _ in range(len(core_prompt_deltas))]
            print("... Getting prompt delta variations")
            for i, core_prompt_delta in tqdm(enumerate(core_prompt_deltas)):
                variations_response: PromptDeltasVariationsResponse = (
                    await get_prompt_delta_variations_prompt.arun(
                        inputs={
                            "<<<CORE_DELTA>>>": str(
                                core_prompt_delta["prompt_delta"].to_dict()
                            ),
                            "<<<CURRENT_PROGRAM>>>": program_string,
                            "<<<FAILURE_MODE>>>": problem["problem_description"],
                        },
                        lm=LLM(model_name="gpt-4o-mini"),
                        response_model=PromptDeltasVariationsResponse,
                    )
                )
                for j, variation in enumerate(variations_response.messages):
                    prompt_delta = PromptDelta(
                        description=variations_response.descriptions[j],
                        message=variations_response.messages[j],
                        subcomponent=variations_response.subcomponents[j],
                        topic_name=variations_response.topic_names[j],
                        instructions_fields_edits=variations_response.instructions_fields_edits[
                            j
                        ],
                    )
                    prompt_delta.validate(
                        self.student_program.nodes[
                            core_prompt_delta["node_name"]
                        ].transform.prompt
                    )
                    prompt_delta.apply(
                        self.student_program.nodes[
                            core_prompt_delta["node_name"]
                        ].transform.prompt
                    )
                    prompt_delta_variations[i].append(prompt_delta)
                prompt_delta_variations[i].append(core_prompt_delta["prompt_delta"])
                if not problem["problem_name"] in deltas_by_problem_by_node:
                    deltas_by_problem_by_node[problem["problem_name"]] = {}
                if (
                    not core_prompt_delta["node_name"]
                    in deltas_by_problem_by_node[problem["problem_name"]]
                ):
                    deltas_by_problem_by_node[problem["problem_name"]][
                        core_prompt_delta["node_name"]
                    ] = []
                deltas_by_problem_by_node[problem["problem_name"]][
                    core_prompt_delta["node_name"]
                ].extend(prompt_delta_variations[i])

            bootstrapped_demos_for_node = bootstrapped_demos_by_node[
                core_prompt_delta["node_name"]
            ]
            print("... Getting apt demos")

            ## TODO too slow!
            # demo_indices_response: DemosForProblemResponse = await get_apt_demos_prompt.arun(
            #     inputs = {
            #         "<<<BOOTSTRAPPED_DEMOS>>>": "\n".join([f"{i}: {demo}" for i, demo in enumerate(bootstrapped_demos_for_node)]),
            #         "<<<FAILURE_DESCRIPTION>>>": problem["problem_description"],
            #         "<<<FAILURE_MODE>>>": problem["failure_mode"],
            #         "<<<EXAMPLE_QUESTIONS>>>": str(problem["example_questions"]),
            #         "<<<NODES_RESPONSIBLE>>>": str(problem["nodes_responsible_by_name"]),
            #     },
            #     lm=LLM(model_name="gpt-4o-mini"),
            #     response_model=DemosForProblemResponse
            # )
            # demos_by_problem_by_node[problem["problem_name"]][core_prompt_delta["node_name"]] = [bootstrapped_demos_for_node[i] for i in demo_indices_response.demo_indices]
            demos_by_problem_by_node[problem["problem_name"]][
                core_prompt_delta["node_name"]
            ] = bootstrapped_demos_for_node
        print("Counts for deltas_by_problem_by_node:")

        for problem, nodes in deltas_by_problem_by_node.items():
            print(f"  Problem: {problem}")
            for node, deltas in nodes.items():
                assert isinstance(
                    deltas, list
                ), f"Deltas for problem {problem} and node {node} are not a list: {deltas}"
                assert all(
                    isinstance(delta, PromptDelta) for delta in deltas
                ), f"Deltas for problem {problem} and node {node} are not all PromptDelta: {deltas}"
                print(f"    Node: {node}, Delta count: {len(deltas)}")

        print("\nCounts for demos_by_problem_by_node:")
        for problem, nodes in demos_by_problem_by_node.items():
            print(f"  Problem: {problem}")
            for node, demos in nodes.items():
                print(f"    Node: {node}, Demo count: {len(demos)}")
        return deltas_by_problem_by_node, demos_by_problem_by_node
        # Add a plan-search DAG once pydantic output model is supported for DAGs in apropos
        # intervention_concepts_response: InterventionConceptsResponse = plan_search_dag.run_standard(
        # # step to scope which instructions etc to change
        # prompt_deltas_by_node = {}
        # #for problem_description

        # 1. Generate problems
        # 2. Write interventions
        # 2. Filter bootstrapped demos by problem and node
        pass

    def prepare_candidate_dag(
        self,
        prompt_deltas_by_node: Dict[str, List[PromptDelta]],
        demos_by_node: Dict[str, List[Demonstration]],
    ):
        candidate_dag = deepcopy(self.student_program)
        for node_name, prompt_deltas in prompt_deltas_by_node.items():
            for prompt_delta in prompt_deltas:
                candidate_dag.nodes[node_name].transform.prompt = prompt_delta.apply(
                    candidate_dag.nodes[node_name].transform.prompt
                )
        for node_name, demos in demos_by_node.items():
            candidate_dag.nodes[node_name].demonstrations = demos
        return candidate_dag

    # Evaluate a program
    def evaluate_program(
        self, dag: LM_DAG, questions: List[Question]
    ) -> Tuple[float, int]:
        results = [question.evaluate(dag) for question in questions]
        return sum([result for result in results]) / len(results), len(results)

    # TPE sampler over demos + instructions
    def search(self, algorithm: Literal["TPE", "RandomSearch"] = "TPE"):
        if algorithm == "TPE":
            optimizer = TPE_Optimizer(
                seed=self.cfg["seed"],
                n_optuna_trials=self.cfg["n_optuna_trials"],
                questions_for_val=self.dataset_handler.dev[0 : self.cfg["dev_size"]],
            )
        elif algorithm == "RandomSearch":
            optimizer = RandomSearch_Optimizer(
                seed=self.cfg["seed"],
                n_optuna_trials=self.cfg["n_optuna_trials"],
                questions_for_val=self.dataset_handler.dev[0 : self.cfg["dev_size"]],
            )
        return optimizer.search(
            baseline_program=self.student_program,
            search_space=self.search_space,
        )

    def choose_best_program(self, programs: List[LM_DAG], scores: List[float]):
        return programs[scores.index(max(scores))]

    async def optimize_program(self) -> LM_DAG:
        print("Starting optimization")
        print("Getting learnable questions ...")
        (
            learnable_questions,
            successes,
            failures,
            pass_at_ks,
        ) = await self.get_learnable_questions()
        print("... got N learnable questions: ", len(learnable_questions))
        print("Curating questions for bootstrapping ...")
        questions_for_bootstrapping, questions_for_train, questions_for_val = (
            self.curate_questions_for_bootstrapping(
                learnable_questions=learnable_questions,
                successes=successes,
                failures=failures,
                pass_at_ks=pass_at_ks,
            )
        )
        print(
            "... chose lengths: ",
            len(questions_for_bootstrapping),
            len(questions_for_train),
            len(questions_for_val),
        )
        bootstrapped_demos_by_node = await self.bootstrap_demonstrations_on_subset(
            questions_for_bootstrapping
        )
        # instruction_deltas_by_problem_by_node, demos_by_problem_by_node = await self.propose_prompt_instruction_components_complex_async(questions_for_train, successes, failures, pass_at_ks, bootstrapped_demos_by_node)
        (
            instruction_deltas_by_problem_by_node,
            demos_by_problem_by_node,
        ) = await self.propose_prompt_instruction_components_simple_async(
            questions_for_train,
            successes,
            failures,
            pass_at_ks,
            bootstrapped_demos_by_node,
        )
        self.search_space = SearchSpace(
            prompt_delta_variations_by_problem_by_node=instruction_deltas_by_problem_by_node,
            demos_by_problem_by_node=demos_by_problem_by_node,
        )
        print("Searching over candidate programs...")
        best_program, scored_attempts = self.search(algorithm="TPE")
        print("... found best program")
        print("All scores: ", [score for score, _, _ in scored_attempts])
        return best_program


if __name__ == "__main__":
    from apropos.src.bench.hendryks_math.dags.plan_execute import (
        hendryks_math_plan_execute_example,
    )
    from apropos.src.bench.hendryks_math.main import HendryksMath_Benchmark

    benchmark = HendryksMath_Benchmark()
    plan_execute_dag = hendryks_math_plan_execute_example(
        model_names=["gpt-4o-mini"] * 2
    )

    DEV_SIZE = 30
    TEST_SIZE = 100
    mipro_v2p1 = MIPrO_V2p1_DAG(
        student_program=plan_execute_dag,
        dataset_handler=benchmark,
        teacher_program=plan_execute_dag,
        cfg={
            "seed": 42,
            "n_optuna_trials": 5,
            "dev_size": DEV_SIZE,
            "learnable_questions": {
                "max_n_to_obtain": 20,
                "max_n_to_sample": 100,
                "base_temp": 0.0,
                "k_for_pass_at_k": 5,
            },
        },
    )
    best_program = asyncio.run(mipro_v2p1.optimize_program())
    baseline_dag_scores = benchmark.score_dag_parsync(
        plan_execute_dag, split="test", n=TEST_SIZE
    )
    optimized_dag_scores = benchmark.score_dag_parsync(
        best_program, split="test", n=TEST_SIZE
    )
    print(
        "Baseline Program Performance: ",
        sum(baseline_dag_scores) / len(baseline_dag_scores),
    )
    print(
        "Optimized Program Performance: ",
        sum(optimized_dag_scores) / len(optimized_dag_scores),
    )

    # Not too bad!
    # Eval the programs on test
