import backoff
from anthropic import AsyncAnthropic, Anthropic, APIError
from src.lms.cache_init import cache
import time
from src.lms.vendors.base import BaseProvider
from src.lms.vendors.json_structured_outputs.core import add_json_instructions_to_messages, extract_pydantic_model_from_response

BACKOFF_TOLERANCE = 10

class AnthropicAPIProvider(BaseProvider):
    def __init__(self):
        self.sync_client = Anthropic()
        self.async_client = AsyncAnthropic()
        self.supports_response_model = True

    @backoff.on_exception(
        backoff.expo,
        APIError,
        max_tries=BACKOFF_TOLERANCE,
        giveup=lambda e: getattr(e, 'status_code', None) != 429,
    )
    def sync_chat_completion_with_response_model(self, messages, model, temperature, max_tokens, response_model):
        messages_with_json_formatting_instructions = add_json_instructions_to_messages(messages, response_model)
        hit = cache.hit_cache(messages_with_json_formatting_instructions, model, temperature, response_model)
        if hit:
            return hit
        raw_text_api_response = self.sync_chat_completion(messages_with_json_formatting_instructions, model, temperature, max_tokens)
        structured_api_response = extract_pydantic_model_from_response(raw_text_api_response, response_model)
        cache.add_to_cache(messages_with_json_formatting_instructions, model, temperature, response_model, structured_api_response)
        return structured_api_response

    @backoff.on_exception(
        backoff.expo,
        APIError,
        max_tries=BACKOFF_TOLERANCE,
        giveup=lambda e: getattr(e, 'status_code', None) != 429,
    )
    async def async_chat_completion_with_response_model(self, messages, model, temperature, max_tokens, response_model):
        messages_with_json_formatting_instructions = add_json_instructions_to_messages(messages, response_model)
        hit = cache.hit_cache(messages_with_json_formatting_instructions, model, temperature, response_model)
        if hit:
            return hit
        raw_text_api_response = await self.async_chat_completion(messages_with_json_formatting_instructions, model, temperature, max_tokens)
        structured_api_response = await extract_pydantic_model_from_response(raw_text_api_response, response_model)
        cache.add_to_cache(messages_with_json_formatting_instructions, model, temperature, response_model, structured_api_response)
        return structured_api_response

    @backoff.on_exception(
        backoff.expo,
        APIError,
        max_tries=BACKOFF_TOLERANCE,
        giveup=lambda e: getattr(e, 'status_code', None) != 429,
    )
    def sync_chat_completion(self, messages, model, temperature, max_tokens):
        t0 = time.time()
        hit = cache.hit_cache(messages, model, temperature, None)
        t1 = time.time()
        print(f"Time taken to hit cache: {t1-t0:.2e} seconds")
        if hit:
            return hit
        system = messages[0]["content"]
        if temperature > 1:
            temperature = 0.999
        response = self.sync_client.messages.create(
            system=system,
            messages=messages[1:],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        output = response.content[0].text
        cache.add_to_cache(messages, model, temperature, None, output)
        return output

    @backoff.on_exception(
        backoff.expo,
        APIError,
        max_tries=BACKOFF_TOLERANCE,
        giveup=lambda e: getattr(e, 'status_code', None) != 429,
    )
    async def async_chat_completion(self, messages, model, temperature, max_tokens):
        hit = cache.hit_cache(messages, model, temperature, None)
        if hit:
            return hit
        system = messages[0]["content"]
        if temperature > 1:
            temperature = 0.999
        response = await self.async_client.messages.create(
            system=system,
            messages=messages[1:],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        output = response.content[0].text
        cache.add_to_cache(messages, model, temperature, None, output)
        return output

if __name__ == "__main__":
    import asyncio
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can answer questions about the capital of France."
        },
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]
    response = asyncio.run(AnthropicAPIProvider().async_chat_completion(
        messages=messages,
        model="claude-3-haiku-20240307",
        temperature=0.0,
        max_tokens=150,
    ))
    print(response)