import os
from datetime import datetime
from typing import Dict, List, Optional, Type

import loguru
from pydantic import BaseModel

from src.lms.vendors.openai_ai_api import (
    async_openai_chat_completion,
    async_openai_chat_completion_with_response_model,
    sync_openai_chat_completion_with_response_model,
    sync_openai_chat_completion,
)
from src.lms.vendors.groq_api import (
    async_groq_chat_completion,
)

GROQ_MODELS = ["llama3-8b-8192", "llama-3.1-8b-instant"]


logger = loguru.logger


def log_backoff(details):
    pass


def build_messages(sys_msg: str, user_msg: str) -> List[Dict]:
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]


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

    def route_model_to_provider(self, model_name):
        if "gpt" in model_name:
            return "openai"
        elif model_name in GROQ_MODELS:
            return "groq"
        else:
            raise ValueError(f"Model {model_name} not supported")

    def sync_respond(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Optional[Type[BaseModel]] = None,
    ):
        messages = build_messages(system_prompt, user_prompt)

        provider = self.route_model_to_provider(self.model_name)
        if provider == "openai":
            if response_model:
                return sync_openai_chat_completion_with_response_model(
                    messages, self.model_name, self.temperature, response_model
                )
            else:
                return sync_openai_chat_completion(
                    messages, self.model_name, self.temperature, self.max_tokens
                )
        else:
            raise ValueError(f"Provider {provider} not supported")

    async def async_respond(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Optional[Type[BaseModel]] = None,
    ):
        messages = build_messages(system_prompt, user_prompt)
        provider = self.route_model_to_provider(self.model_name)
        if provider == "openai":
            if response_model:
                return await async_openai_chat_completion_with_response_model(
                    messages,
                    self.model_name,
                    temperature=self.temperature,
                    response_model=response_model,
                    max_tokens=self.max_tokens,
                )
            else:
                return await async_openai_chat_completion(
                    messages,
                    self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
        elif provider == "groq":
            if response_model:
                raise NotImplementedError(
                    "We haven't added Groq response model support yet"
                )
            return await async_groq_chat_completion(
                messages,
                self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:
            raise ValueError(f"Provider {provider} not supported")
