import copy
from typing import Callable, Dict, List, Optional, Type

from apropos.src.bench.base import Benchmark
from apropos.src.core.programs.dag import LM_DAG
from apropos.src.core.programs.prompt import Demonstration


class DAGOptimizer:
    student_program: LM_DAG
    teacher_program: Optional[LM_DAG]
    dataset_handler: Type[Benchmark]

    def get_fewshot_program(
        self, demonstrations_by_node_name: Dict[str, List[Demonstration]]
    ):
        fewshot_program = copy.deepcopy(self.student_program)
        for node_name, demonstrations_for_stage in demonstrations_by_node_name.items():
            optimal_demos = demonstrations_for_stage
            node = fewshot_program.nodes[node_name]
            node.get_input_fields.prompt.demonstrations = optimal_demos
            node.transform.prompt.demonstrations = optimal_demos
            node.produce_stage_record.prompt.demonstrations = optimal_demos
        return fewshot_program

    # Annotator must send DAG to annotated DAG
    async def bootstrap_demonstrations(
        self, annotator: Optional[Callable] = None, n: int = 20, patches=["A", "B"]
    ) -> Dict[str, List[Demonstration]]:
        if self.teacher_program:
            candidate = copy.deepcopy(self.teacher_program)
        else:
            candidate = copy.deepcopy(self.student_program)
        scores, dag_records = await self.dataset_handler.score_dag(
            candidate, n=n, verbose=True, patches=patches, split="train"
        )
        fewshot_demonstrations_by_node = {
            node_key: [] for node_key in candidate.nodes.keys()
        }
        for dag_record, score in zip(dag_records, scores):
            if score == True:
                if not annotator:
                    annotated_stage_demonstrations = (
                        dag_record.trivially_annotate().to_stage_demonstrations()
                    )
                else:
                    annotated_stage_demonstrations = annotator(
                        dag_record
                    ).to_stage_demonstrations()
                for i, stage_demonstrations in enumerate(
                    annotated_stage_demonstrations
                ):
                    fewshot_demonstrations_by_node[
                        list(candidate.nodes.keys())[i]
                    ].append(stage_demonstrations)
        return fewshot_demonstrations_by_node
