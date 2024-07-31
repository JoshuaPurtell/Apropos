from bench.hendryks_math.main import HendryksMath_Benchmark
from bench.hendryks_math.dags.plan_execute import hendryks_math_plan_execute_example
from bench.base import Benchmark
import asyncio
from src.programs.dag import LM_DAG

from src.optimizers.baselines.bffsrs import BreadthFirstRandomSearch_DAG
from bench.hendryks_math.dags.single_step import hendryks_math_single_step_example

async def main(
        benchmark: Benchmark,
        plan_execute_dag: LM_DAG,
        single_step_dag: LM_DAG,
        bffsrs_config: dict,
):
    
    bffsrs = BreadthFirstRandomSearch_DAG(
        student_program=plan_execute_dag,
        teacher_program=plan_execute_dag,
        dataset_handler=benchmark,
        cfg=bffsrs_config,
    )
    print("Optimizing demonstrations...")
    optimized_program_with_demos = await bffsrs.optimize_demonstrations()
    
    print("Evaluating programs...")
    baseline_program_scores, _ = await benchmark.score_dag(
        plan_execute_dag, n=100, verbose=True, split="test", patches=["A", "B"]
    )
    optimized_program_scores, _ = await benchmark.score_dag(
        optimized_program_with_demos,
        n=100,
        verbose=True,
        split="test",
        patches=["A", "B"],
    )

    single_step_program_scores, _ = await benchmark.score_dag(
        single_step_dag, n=100, verbose=True, split="test", patches=["A", "B"]
    )
    
    print(
        f"Unoptimized Plan-Execute program score: {sum(baseline_program_scores)/len(baseline_program_scores)}"
    )
    print(
        f"Optimized Plan-Execute program score: {sum(optimized_program_scores)/len(optimized_program_scores)}"
    )
    print(
        f"Baseline Single-Step program score: {sum(single_step_program_scores)/len(single_step_program_scores)}"
    )

    pass

if __name__ == "__main__":
    benchmark = HendryksMath_Benchmark()
    plan_execute_dag = hendryks_math_plan_execute_example(
        model_names=["claude-3-haiku-20240307"] * 2
    )
    baseline_single_step_program = hendryks_math_single_step_example(
        model_name = "claude-3-haiku-20240307"
    )
    bffsrs_config = {
        "optimization": {
            "combination_size": 3,
            "n_iterations": 10,
            "validation_size": 30,
            "test_size": 100,
            "program_search_parallelization_factor": 5
        },
        "bootstrapping": {
            "patches": ["A", "B"],
            "n_questions": 100,
        },
        "verbose": True,
    }
    asyncio.run(main(
        benchmark=benchmark,
        plan_execute_dag=plan_execute_dag,
        single_step_dag=baseline_single_step_program,
        bffsrs_config=bffsrs_config,
    ))
    # After 10 iterations
    # Unoptimized Plan-Execute program score: 0.13
    # Optimized Plan-Execute program score: 0.28
    # Baseline Single-Step program score: 0.24
