# client = AsyncOpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

import os

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from apropos.src.core.lms.vendors.json_structured_outputs.core import (
    add_json_instructions_to_messages,
    extract_pydantic_model_from_response_sync,
    extract_pydantic_model_from_response_async,
)
from apropos.src.core.lms.vendors.openai_like import OpenAIStandardProvider


class DeepSeekAPIProvider(OpenAIStandardProvider):
    def __init__(self, multi_threaded=False):
        self.sync_client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.async_client = AsyncOpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.supports_response_model = False
        super().__init__(
            self.sync_client, self.async_client, multi_threaded=multi_threaded
        )

    async def async_chat_completion_with_response_model(
        self, messages, model, temperature, max_tokens, response_model
    ):
        messages_with_json_formatting_instructions = add_json_instructions_to_messages(
            messages, response_model
        )
        raw_text_api_response = await self.async_chat_completion(
            messages_with_json_formatting_instructions, model, temperature, max_tokens
        )
        structured_api_response = await extract_pydantic_model_from_response_async(
            raw_text_api_response, response_model
        )
        return structured_api_response

    def sync_chat_completion_with_response_model(
        self, messages, model, temperature, max_tokens, response_model
    ):
        messages_with_json_formatting_instructions = add_json_instructions_to_messages(
            messages, response_model
        )
        raw_text_api_response = self.sync_chat_completion(
            messages_with_json_formatting_instructions, model, temperature, max_tokens
        )
        structured_api_response = extract_pydantic_model_from_response_sync(
            raw_text_api_response, response_model
        )
        return structured_api_response


if __name__ == "__main__":
    import asyncio

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can answer questions about the capital of France.",
        },
        {"role": "user", "content": "What is the capital of France?"},
    ]

    class Response(BaseModel):
        city: str

    response = asyncio.run(
        DeepSeekAPIProvider().async_chat_completion_with_response_model(
            messages=messages,
            model="deepseek-coder",
            temperature=0.0,
            max_tokens=150,
            response_model=Response,
        )
    )
    print(response)
