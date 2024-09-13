from typing import Dict, Any, Tuple
import tiktoken
from apropos.src.core.lms.cost_table import cost_table_by_model


class CostMonitor:
    token_counts: Dict
    tokenizer: Any
    pricing: Dict

    def __init__(self, model_name):
        self.token_counts = {"system": 0, "user": 0, "response": 0}
        self.tokenizer = (
            tiktoken.encoding_for_model(model_name)
            if "gpt" in model_name
            else tiktoken.encoding_for_model("gpt-4")
        )
        model_name_matches = [k for k in cost_table_by_model.keys() if k in model_name]
        if len(model_name_matches) == 0:
            print(
                f"Warning: Model {model_name} not found in cost table. Using default pricing of 0."
            )
            self.pricing = {"system": 0, "user": 0, "response": 0}

        else:
            model_name = model_name_matches[0]
        self.pricing = {
            "system": cost_table_by_model[model_name]["cost_per_input_token"],
            "user": cost_table_by_model[model_name]["cost_per_input_token"],
            "response": cost_table_by_model[model_name]["cost_per_output_token"],
        }

    def update_token_counts(self, system_message, user_message, response_message):
        self.token_counts["system"] += len(self.tokenizer.encode(system_message))
        self.token_counts["user"] += len(self.tokenizer.encode(user_message))
        self.token_counts["response"] += len(self.tokenizer.encode(response_message))

    def final_cost(self) -> Tuple[float, int]:
        return sum(
            [self.token_counts[k] * self.pricing[k] for k in self.token_counts]
        ), sum([self.token_counts[k] for k in self.token_counts])
