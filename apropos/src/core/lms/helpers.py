import os
from datetime import datetime
from typing import Dict, List, Optional, Type

import loguru
from pydantic import BaseModel

from apropos.src.core.lms.vendors.anthropic_api import AnthropicAPIProvider
from apropos.src.core.lms.vendors.deepmind_api import DeepmindAPIProvider
from apropos.src.core.lms.vendors.deepseek_api import DeepSeekAPIProvider
from apropos.src.core.lms.vendors.groq_api import GroqAPIProvider
from apropos.src.core.lms.vendors.lambda_api import LambdaAPIProvider
from apropos.src.core.lms.vendors.ollama_api import OllamaAPIProvider
from apropos.src.core.lms.vendors.openai_api import OpenAIAPIProvider
from apropos.src.core.lms.vendors.together_api import TogetherAPIProvider

logger = loguru.logger


def log_backoff(details):
    pass


def build_messages(
    sys_msg: str,
    user_msg: str,
    images_bytes: List = [],
    model_name: Optional[str] = None,
) -> List[Dict]:
    if len(images_bytes) > 0 and "gpt" in model_name:
        return [
            {"role": "system", "content": sys_msg},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_msg}]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_bytes}"},
                    }
                    for image_bytes in images_bytes
                ],
            },
        ]
    elif len(images_bytes) > 0 and "claude" in model_name:
        system_info = {"role": "system", "content": sys_msg}
        user_info = {
            "role": "user",
            "content": [{"type": "text", "text": user_msg}]
            + [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_bytes,
                    },
                }
                for image_bytes in images_bytes
            ],
        }
        return [system_info, user_info]
    elif len(images_bytes) > 0:
        raise ValueError("Images are not yet supported for this model")
    else:
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]


GROQ_MODELS = [
    "llama3-8b-8192",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
]
OLLAMA_MODELS = [
    "gemma2:2b",
    "gemma2",
    "gemma2:27b",
    "llama3.1",
    "llama3.1:70b",
    "mistral",
    "llama3.1:405b",
]
TOGETHER_MODELS = [
    "Qwen/Qwen1.5-4B-Chat",
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
]
MODEL_MAP = {
    "openai": lambda x: "gpt" in x or "o1" in x,
    "groq": lambda x: x in GROQ_MODELS,
    "claude": lambda x: "claude" in x,
    "together": lambda x: x in TOGETHER_MODELS,
    "deepmind": lambda x: "gemini" in x,
    "deepseek": lambda x: "deepseek" in x,
    "lambda": lambda x: "hermes" in x,
    "ollama": lambda x: x in OLLAMA_MODELS,
}

providers = {
    "openai": OpenAIAPIProvider,
    "groq": GroqAPIProvider,
    "claude": AnthropicAPIProvider,
    "together": TogetherAPIProvider,
    "deepmind": DeepmindAPIProvider,
    "deepseek": DeepSeekAPIProvider,
    "lambda": LambdaAPIProvider,
    "ollama": OllamaAPIProvider,
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
        images_bytes: List[str] = [],
        response_model: Optional[Type[BaseModel]] = None,
        multi_threaded: bool = False,
    ):
        messages = build_messages(
            sys_msg=system_prompt,
            user_msg=user_prompt,
            images_bytes=images_bytes,
            model_name=self.model_name,
        )
        provider_name = next(
            (k for k, v in MODEL_MAP.items() if v(self.model_name)), None
        )
        provider = (
            providers[provider_name](multi_threaded=multi_threaded)
            if provider_name != "groq"
            else providers["groq"](force_structured_output=True)
        )
        if response_model:
            return provider.sync_chat_completion_with_response_model(
                messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_model=response_model,
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
        images_bytes: List[str] = [],
        response_model: Optional[Type[BaseModel]] = None,
    ):
        messages = build_messages(
            sys_msg=system_prompt,
            user_msg=user_prompt,
            images_bytes=images_bytes,
            model_name=self.model_name,
        )
        provider_name = next(
            (k for k, v in MODEL_MAP.items() if v(self.model_name)), None
        )
        provider = (
            providers[provider_name]()
            if provider_name != "groq"
            else providers["groq"](force_structured_output=True)
        )
        if response_model:
            result = await provider.async_chat_completion_with_response_model(
                messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_model=response_model,
            )
            assert isinstance(result, str) or isinstance(
                result, BaseModel
            ), f"Result: {result}"
            return result
        else:
            result = await provider.async_chat_completion(
                messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return result


if __name__ == "__main__":
    import base64

    lm = LLM("claude-3-haiku-20240307")
    image_path = "apropos/bench/crafter/crafter.png"

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    image_bytes = encode_image(image_path)
    response = lm.sync_respond(
        system_prompt="Hello",
        user_prompt="Hi",
        images_bytes=[image_bytes],
    )
    print(response)
