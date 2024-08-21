import os
import backoff
import together
from together import AsyncTogether, Together
from apropos.src.core.lms.cache_init import cache
from apropos.src.core.lms.vendors.json_structured_outputs.core import (
    add_json_instructions_to_messages,
    extract_pydantic_model_from_response_sync,
    extract_pydantic_model_from_response_async,
)
from apropos.src.core.lms.vendors.openai_like import OpenAIStandardProvider

BACKOFF_TOLERANCE = 1  # 20

class TogetherAPIProvider(OpenAIStandardProvider):
    def __init__(self):
        self.sync_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        self.async_client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"))
        self.supports_response_model = False

    @backoff.on_exception(
        backoff.expo,
        (together.error.RateLimitError,together.error.APIConnectionError,together.error.APIError),
        max_tries=BACKOFF_TOLERANCE,
        giveup=lambda e: getattr(e, "status_code", None) != 429,
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
            if isinstance(hit, dict) and response_model:
                if "response" in hit:
                    return response_model(**hit["response"])
                return response_model(**hit)
            return hit if not "response" in hit else hit["response"]
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
        (together.error.RateLimitError,together.error.APIConnectionError,together.error.APIError),
        max_tries=BACKOFF_TOLERANCE,
        giveup=lambda e: getattr(e, "status_code", None) != 429,
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
            if isinstance(hit, dict) and response_model:
                if "response" in hit:
                    return response_model(**hit["response"])
                return response_model(**hit)
            return hit if not "response" in hit else hit["response"]
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


if __name__ == "__main__":
    import asyncio

    messages = [
        {
            "role": "system",
            "content": "You are a helpful a ssistant that can answer questions abou t the capital of France.",
        },
        {"role": "user", "content": "What  is the capital of France?   "},
    ]
    response = asyncio.run(
        TogetherAPIProvider().async_chat_completion(
            messages=messages,
            model="teknium/OpenHermes-2p5-Mistral-7B",
            temperature=0.0,
            max_tokens=150,
        )
    )
    print(response)
