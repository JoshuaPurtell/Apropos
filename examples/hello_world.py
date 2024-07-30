from bench.gradeschool_math.main import GSM8k_Benchmark
from bench.gradeschool_math.dags.plan_execute import gsm8k_plan_execute_example
from bench.gradeschool_math.dags.single_step import gsm8k_single_step_example
import asyncio
from src.optimizers.baselines.bffsrs import BreadthFirstRandomSearch_DAG

if __name__ == "__main__":
    benchmark = GSM8k_Benchmark()
    plan_execute_dag = gsm8k_plan_execute_example(
        model_names=["llama-3.1-8b-instant"] * 2
    )
    bffsrs_config = {
        "optimization": {
            "combination_size": 3,
            "n_iterations": 10,
            "validation_size": 30,
            "test_size": 100,
        },
        "bootstrapping": {
            "patches": ["A", "B"],
            "n_questions": 20,
        },
        "verbose": True,
    }
    bffsrs = BreadthFirstRandomSearch_DAG(
        student_program=plan_execute_dag,
        teacher_program=plan_execute_dag,
        dataset_handler=benchmark,
    )
    print("Optimizing demonstrations...")
    optimized_program_with_demos = asyncio.run(bffsrs.optimize_demonstrations())
    print("Evaluating programs...")
    baseline_program_scores, _ = asyncio.run(
        benchmark.score_dag(
            plan_execute_dag, n=100, verbose=True, split="test", patches=["A", "B"]
        )
    )
    optimized_program_scores, _ = asyncio.run(
        benchmark.score_dag(
            optimized_program_with_demos,
            n=100,
            verbose=True,
            split="test",
            patches=["A", "B"],
        )
    )
    print(
        f"Baseline program score: {sum(baseline_program_scores)/len(baseline_program_scores)}"
    )
    print(
        f"Optimized program score: {sum(optimized_program_scores)/len(optimized_program_scores)}"
    )
