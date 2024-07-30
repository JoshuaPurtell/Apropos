from bench.gradeschool_math.main import GSM8k_Benchmark
from bench.gradeschool_math.dags.plan_execute import gsm8k_plan_execute_example
import asyncio

if __name__ == "__main__":
    benchmark = GSM8k_Benchmark()
    plan_execute_dag = gsm8k_plan_execute_example()
    score, dag_record = asyncio.run(
        benchmark.train[0].compute_and_score_attempt(plan_execute_dag)
    )
    print(score)
    print(dag_record)
