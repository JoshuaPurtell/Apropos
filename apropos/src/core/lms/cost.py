from typing import Dict, Any, Tuple
import tiktoken

cost_table_by_model = {
    "gpt-4o-mini-2024-07-18": {
        "cost_per_input_token": 1.5e-07,
        "cost_per_output_token": 6e-07,
    },
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite": {
        "cost_per_input_token": 1.8e-07,
        "cost_per_output_token": 1.8e-07,
    },
    "gemini-1.5-flash-8b-exp-0827": {
        "cost_per_input_token": 7.5e-08,
        "cost_per_output_token": 3e-07,
    },
    "gemini-1.5-flash-exp-0827": {
        "cost_per_input_token": 7.5e-08,
        "cost_per_output_token": 3e-07,
    },
    "gemini-1.5-flash": {
        "cost_per_input_token": 7.5e-08,
        "cost_per_output_token": 3e-07,
    },
    "claude-3-haiku-20240307": {},
    "deepseek-chat": {
        "cost_per_input_token": 1.4e-07,
        "cost_per_output_token": 2.8e-07,
    },
    "llama3-8b-8192": {
        "cost_per_input_token": 5e-08,
        "cost_per_output_token": 8e-08,
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {
        "cost_per_input_token": 1.8e-07,
        "cost_per_output_token": 1.8e-07,
    },
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {
        "cost_per_input_token": 6e-07,
        "cost_per_output_token": 6e-07,
    },
    "hermes-3-llama-3.1-405b-fp8-128k": {
        "cost_per_input_token": 5e-06,
        "cost_per_output_token": 5e-06,
    },
    "gpt-4o-2024-08-06": {
        "cost_per_input_token": 2.5e-06,
        "cost_per_output_token": 10e-06,
    },
    "claude-3-5-sonnet-20240620": {
        "cost_per_input_token": 3e-06,
        "cost_per_output_token": 15e-06,
    },
    "gemini-1.5-pro": {
        "cost_per_input_token": 2.5e-06,
        "cost_per_output_token": 2.5e-06,
    },
    "claude-3-haiku-20240307": {
        "cost_per_input_token": 2.5e-07,
        "cost_per_output_token": 1.25e-06,
    },
}


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
            raise ValueError(f"Model {model_name} not found in cost table")
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
