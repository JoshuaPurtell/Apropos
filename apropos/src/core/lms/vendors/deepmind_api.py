import asyncio
import logging
import os
from typing import Dict, List

import backoff
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from apropos.src.core.lms.cache_init import cache
from apropos.src.core.lms.vendors.base import BaseProvider
from apropos.src.core.lms.vendors.json_structured_outputs.core import (
    add_json_instructions_to_messages,
    extract_pydantic_model_from_response_sync,
    extract_pydantic_model_from_response_async,
)
import google.api_core.exceptions

# Suppress all logging from google.generativeai
logging.getLogger("google.generativeai").setLevel(logging.ERROR)
os.environ["GRPC_VERBOSITY"] = "ERROR"
# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress other warnings
import warnings

warnings.filterwarnings("ignore")

BACKOFF_TOLERANCE = 10


class DeepmindAPIProvider(BaseProvider):
    def __init__(self, multi_threaded: bool = False):
        self.supports_response_model = True
        if multi_threaded:
            raise ValueError("Deepmind API does not currently support multi-threading")

    @backoff.on_exception(
        backoff.expo,
        (
            Exception,
            google.api_core.exceptions.ResourceExhausted,
        ),  # Replace with specific exceptions if known
        max_tries=BACKOFF_TOLERANCE,
    )
    async def hit_gemini_async(
        self,
        messages: List[Dict],
        temperature: float = 0,
        model_name: str = "gemini-1.5-flash",
    ) -> str:
        code_generation_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"temperature": temperature},
            system_instruction=messages[0]["content"],
        )
        result = await code_generation_model.generate_content_async(
            messages[1]["content"],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
        try:
            return result.text
        except Exception as e:
            print("Gemini failed", e)
            print(result)
            return "Gemini failed"

    @backoff.on_exception(
        backoff.expo,
        (
            Exception,
            google.api_core.exceptions.ResourceExhausted,
        ),  # Replace with specific exceptions if known
        max_tries=BACKOFF_TOLERANCE,
    )
    def hit_gemini_sync(
        self,
        messages: List[Dict],
        temperature: float = 0,
        model_name: str = "gemini-1.5-flash",
    ) -> str:
        code_generation_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"temperature": temperature},
            system_instruction=messages[0]["content"],
        )
        result = code_generation_model.generate_content(
            messages[1]["content"],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
        try:
            return result.text
        except Exception as e:
            print("Gemini failed", e)
            print(result)
            return "Gemini failed"

    @backoff.on_exception(
        backoff.expo,
        (Exception),  # Replace with specific exceptions if known
        max_tries=BACKOFF_TOLERANCE,
    )
    async def async_chat_completion_with_response_model(
        self, messages, model, temperature, max_tokens, response_model
    ):
        messages_with_json_formatting_instructions = add_json_instructions_to_messages(
            messages, response_model
        )
        hit = cache.hit_cache(
            messages_with_json_formatting_instructions,
            model,
            temperature,
            response_model,
        )
        if hit:
            return hit
        raw_text_api_response = await self.async_chat_completion(
            messages_with_json_formatting_instructions, model, temperature, max_tokens
        )
        structured_api_response = await extract_pydantic_model_from_response_async(
            raw_text_api_response, response_model
        )
        cache.add_to_cache(
            messages_with_json_formatting_instructions,
            model,
            temperature,
            response_model,
            structured_api_response,
        )
        return structured_api_response

    @backoff.on_exception(
        backoff.expo,
        (
            Exception,
            google.api_core.exceptions.ResourceExhausted,
        ),  # Replace with specific exceptions if known
        max_tries=BACKOFF_TOLERANCE,
    )
    def sync_chat_completion_with_response_model(
        self, messages, model, temperature, max_tokens, response_model
    ):
        messages_with_json_formatting_instructions = add_json_instructions_to_messages(
            messages, response_model
        )
        hit = cache.hit_cache(
            messages_with_json_formatting_instructions,
            model,
            temperature,
            response_model,
        )
        if hit:
            return hit
        raw_text_api_response = self.sync_chat_completion(
            messages_with_json_formatting_instructions, model, temperature, max_tokens
        )
        structured_api_response = extract_pydantic_model_from_response_sync(
            raw_text_api_response, response_model
        )
        cache.add_to_cache(
            messages_with_json_formatting_instructions,
            model,
            temperature,
            response_model,
            structured_api_response,
        )
        return structured_api_response

    @backoff.on_exception(
        backoff.expo,
        (Exception),  # Replace with specific exceptions if known
        max_tries=BACKOFF_TOLERANCE,
    )
    async def async_chat_completion(self, messages, model, temperature, max_tokens):
        hit = cache.hit_cache(messages, model, temperature, None)
        if hit:
            return hit
        result = await self.hit_gemini_async(
            messages=messages,
            model_name=model,
            temperature=temperature,
        )
        cache.add_to_cache(messages, model, temperature, None, result)
        return result

    @backoff.on_exception(
        backoff.expo,
        (Exception),  # Replace with specific exceptions if known
        max_tries=BACKOFF_TOLERANCE,
    )
    def sync_chat_completion(self, messages, model, temperature, max_tokens):
        hit = cache.hit_cache(messages, model, temperature, None)
        if hit:
            return hit
        result = self.hit_gemini_sync(
            messages=messages,
            model_name=model,
            temperature=temperature,
        )
        cache.add_to_cache(messages, model, temperature, None, result)
        return result


if __name__ == "__main__":
    import asyncio

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can answer questions about the capital of France.",
        },
        {"role": "user", "content": " What is t he capital of France? "},
    ]
    response = asyncio.run(
        DeepmindAPIProvider().async_chat_completion(
            messages=messages,
            model="gemini-1.5-flash",
            temperature=0.0,
            max_tokens=150,
        )
    )
    print(response)
    sync_response = DeepmindAPIProvider().sync_chat_completion(
        messages=messages,
        model="gemini-1.5-flash",
        temperature=0.0,
        max_tokens=150,
    )
    print(sync_response)
