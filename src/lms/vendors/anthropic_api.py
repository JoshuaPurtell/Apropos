import backoff
from anthropic import AsyncAnthropic, Anthropic, APIError
import anthropic
from src.lms.api_caching_fxns import safecache_wrapper
from src.lms.cache_init import cache, old_cache
from src.lms.vendors.base import BaseProvider

from src.lms.vendors.json_structured_outputs.core import add_json_instructions_to_messages, extract_pydantic_model_from_response

class AnthropicAPIProvider(BaseProvider):
    def __init__(self):
        self.sync_client = Anthropic()
        self.async_client = AsyncAnthropic()

    
    async def async_chat_completion_with_response_model(self, messages, model, temperature, max_tokens, response_model):
        messages_with_json_formatting_instructions = add_json_instructions_to_messages(messages, response_model)
        raw_text_api_response = await self.async_chat_completion(messages_with_json_formatting_instructions, model, temperature, max_tokens)
        structured_api_response = await extract_pydantic_model_from_response(raw_text_api_response, response_model)
        return structured_api_response

    def sync_chat_completion_with_response_model(self, messages, model, temperature, max_tokens, response_model):
        messages_with_json_formatting_instructions = add_json_instructions_to_messages(messages, response_model)
        raw_text_api_response = self.sync_chat_completion(messages_with_json_formatting_instructions, model, temperature, max_tokens)
        structured_api_response = extract_pydantic_model_from_response(raw_text_api_response, response_model)
        return structured_api_response
    

    @backoff.on_exception(
        backoff.expo,
        (APIError,anthropic.InternalServerError),
        max_tries=20,
        giveup=lambda e: getattr(e, 'status_code', None) != 429,
        logger=None
    )
    async def async_chat_completion(self, messages, model, temperature, max_tokens):
        hit = safecache_wrapper.hit_cache(messages, model, temperature, None, cache, old_cache)
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
        safecache_wrapper.add_to_cache(messages, model, temperature, None, output, cache)
        return output

    @backoff.on_exception(
        backoff.expo,
        (APIError, anthropic.InternalServerError),
        max_tries=20,
        giveup=lambda e: getattr(e, 'status_code', None) != 429,
        logger=None
    )
    def sync_chat_completion(self, messages, model, temperature, max_tokens):
        hit = safecache_wrapper.hit_cache(messages, model, temperature, None, cache, old_cache)
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
        safecache_wrapper.add_to_cache(messages, model, temperature, None, output, cache)
        return output