from apropos.src.bench.hendryks_math.dags.plan_execute import (
    hendryks_math_plan_execute_example,
)
from apropos.src.bench.hendryks_math.main import HendryksMath_Benchmark
from apropos.src.core.optimizers.miprov2p1.algorithm import MIPrO_V2p1_DAG
import asyncio

if __name__ == "__main__":
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
