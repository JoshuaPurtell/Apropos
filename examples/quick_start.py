from apropos.src.core.lms.helpers import LLM
import numpy as np
from apropos.src.core.utils.program_grounding import (
    Metric,
    messages_to_dag_and_benchmark,
)
from apropos.src.core.optimizers.baselines.bffsrs import BreadthFirstRandomSearch_DAG
import asyncio


def get_hendryks_messages_and_gold_outputs():
    from apropos.src.bench.hendryks_math.main import (
        HendryksMath_Benchmark,
        custom_math_metric,
    )

    benchmark = HendryksMath_Benchmark()
    from apropos.src.bench.hendryks_math.dags.single_step import (
        hendryks_math_single_step_example,
    )

    baseline_single_step_program = hendryks_math_single_step_example(
        model_name="claude-3-haiku-20240307"
    )
    messages_examples = []
    for q in benchmark.train:
        systems_message, user_message = list(
            baseline_single_step_program.nodes.values()
        )[0].transform.prompt.compile(
            inputs={
                "<<<MATHEMATICS_QUESTION>>>": q.information["question"],
            }
        )
        messages_examples.append((systems_message, user_message))
    gold_outputs = [q.information["answer"] for q in benchmark.train]
    return messages_examples, gold_outputs, custom_math_metric


async def optimize_and_score_program(cfg, dag, benchmark, n_to_score: int = 100):
    print("Evaluating baseline program...")
    baseline_program_scores, _ = await benchmark.score_dag(
        dag, n=n_to_score, verbose=True, split="test", patches=["A", "B"]
    )
    print("Baseline program scores:", np.mean(baseline_program_scores))
    bffsrs = BreadthFirstRandomSearch_DAG(
        student_program=dag,
        teacher_program=dag,
        dataset_handler=benchmark,
        cfg=cfg,
    )
    print("Optimizing demonstrations...")
    optimized_dag = await bffsrs.optimize_demonstrations()

    print("Evaluating optimized program...")
    optimized_program_scores, _ = await benchmark.score_dag(
        optimized_dag, n=n_to_score, verbose=True, split="test", patches=["A", "B"]
    )
    print("Optimized program scores:", np.mean(optimized_program_scores))


async def main(
    messages_examples,
    metric,
    input_names,
    name_for_prompt,
    grounding_llm,
    program_llm,
    bffsrs_config,
):
    dag, benchmark = await messages_to_dag_and_benchmark(
        messages_examples,
        metric,
        input_names,
        name_for_prompt,
        grounding_llm,
        program_llm,
    )

    await optimize_and_score_program(bffsrs_config, dag, benchmark)


if __name__ == "__main__":
    # REPLACE WITH YOUR OWN DATA
    # messages_examples: List[List[Dict[str, str]]], len N
    # gold_outputs: List[str], len N
    # metric: Metric
    messages_examples, gold_outputs, custom_math_metric = (
        get_hendryks_messages_and_gold_outputs()
    )

    grounding_llm = LLM(
        "claude-3-5-sonnet-20240620"
    )  # For structuring prompts, better to have high-power
    program_llm = LLM("gpt-4o-mini")  # For generating programs
    metric = Metric(
        gold_outputs_for_dataset=gold_outputs, metric_function=custom_math_metric
    )
    input_names = ["question"]
    name_for_prompt = "Solve Math Problem"
    bffsrs_config = {
        "optimization": {
            "combination_size": 3,
            "n_iterations": 10,
            "validation_size": 30,
            "test_size": 30,
            "program_search_parallelization_factor": 5,
        },
        "bootstrapping": {
            "patches": ["A", "B"],
            "n_questions": 30,  # 100
        },
        "verbose": True,
    }
    asyncio.run(
        main(
            messages_examples,
            metric,
            input_names,
            name_for_prompt,
            grounding_llm,
            program_llm,
            bffsrs_config,
        )
    )
