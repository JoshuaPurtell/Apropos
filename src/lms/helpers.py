import os
from datetime import datetime
from typing import Dict, List, Optional, Type

import loguru
from dotenv import load_dotenv

from pydantic import BaseModel

from src.lms.vendors.anthropic_api import AnthropicAPIProvider
from src.lms.vendors.deepmind_api import DeepmindAPIProvider
from src.lms.vendors.openai_api import OpenAIAPIProvider
from src.lms.vendors.groq_api import GroqAPIProvider
from src.lms.vendors.deepseek_api import DeepSeekAPIProvider
from src.lms.vendors.together_api import TogetherAPIProvider
from src.lms.vendors.ollama_api import OllamaAPIProvider

load_dotenv()
from typing import Dict

logger = loguru.logger

def log_backoff(details):
    pass


def build_messages(sys_msg: str, user_msg: str) -> List[Dict]:
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]

GROQ_MODELS = ["llama3-8b-8192","llama3-70b-8192","llama-3.1-8b-instant","llama-3.1-70b-versatile"]
TOGETHER_MODELS = ["Qwen/Qwen1.5-4B-Chat","meta-llama/Meta-Llama-3-8B-Instruct-Lite","meta-llama/Llama-2-13b-chat-hf"]
OLLAMA_MODELS = ["gemma2:2b"]
MODEL_MAP = {
    "openai": lambda x: "gpt" in x,
    "groq": lambda x: x in GROQ_MODELS,
    "deepmind": lambda x: "gemini" in x,
    "claude": lambda x: "claude" in x,
    "deepseek": lambda x: "deepseek" in x,
    "together": lambda x: x in TOGETHER_MODELS,
    "ollama": lambda x: x in OLLAMA_MODELS
}
providers = {
    "openai": OpenAIAPIProvider(),
    "groq": GroqAPIProvider(),
    "deepmind": DeepmindAPIProvider(),
    "claude": AnthropicAPIProvider(),
    "deepseek": DeepSeekAPIProvider(),
    "together": TogetherAPIProvider(),
    "ollama": OllamaAPIProvider()
}

class LLM:
    model_name: str

    def __init__(
        self,
        model_name: str,
        temperature: Optional[float] = 0,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens else 1000
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

    def save(self, system_prompt, user_prompt, response_model, response):
        if response_model:
            response = response.dict()

    def log_response(self, system_prompt: str, user_prompt: str, response: str):
        log_dir = "logs/llm/bulk"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        now = datetime.now()
        formatted_now = now.strftime("%H_%d_%m_%Y")
        i = 0
        while os.path.exists(f"{log_dir}/log_{formatted_now}_{i}.txt"):
            i += 1
        log_content = f"System Prompt:\n```{system_prompt}```\n\nUser Prompt:```\n{user_prompt}\n\nLLM Response:\n```{response}```"
        with open(f"{log_dir}/log_{formatted_now}_{i}.txt", "w") as file:
            file.write(log_content)


    def sync_respond(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Optional[Type[BaseModel]] = None,
    ):
        messages = build_messages(system_prompt, user_prompt)
        provider_name = next((k for k, v in MODEL_MAP.items() if v(self.model_name)), None)
        provider = providers[provider_name]
        if response_model:
            return provider.sync_chat_completion_with_response_model(
                messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:
            return provider.sync_chat_completion(
                messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

    async def async_respond(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Optional[Type[BaseModel]] = None,
    ):
        messages = build_messages(system_prompt, user_prompt)
        provider_name = next((k for k, v in MODEL_MAP.items() if v(self.model_name)), None)
        provider = providers[provider_name]
        if response_model:
            return await provider.async_chat_completion_with_response_model(
                messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_model=response_model
            )
        else:
            return await provider.async_chat_completion(
                messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

        
