from src.lms.vendors.openai_like import OpenAIStandardProvider
from groq import AsyncGroq, Groq
import instructor
import os
import backoff
import pydantic_core
from src.lms.cache_init import cache
import groq
from src.lms.vendors.json_structured_outputs.core import add_json_instructions_to_messages, extract_pydantic_model_from_response

BACKOFF_TOLERANCE = 200

class GroqAPIProvider(OpenAIStandardProvider):
    def __init__(self, use_instructor=False):
        if use_instructor:
            self.sync_client = instructor.patch(Groq(api_key=os.environ.get("GROQ_API_KEY")))
            self.async_client = instructor.patch(AsyncGroq(api_key=os.environ.get("GROQ_API_KEY")))
        else:
            self.sync_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            self.async_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        self.supports_response_model = True
    
    @backoff.on_exception(
        backoff.expo,
        (groq.RateLimitError, groq.APITimeoutError, pydantic_core._pydantic_core.ValidationError),
        max_tries=BACKOFF_TOLERANCE,
        logger=None,
        on_giveup=lambda e: print(e) if isinstance(e, pydantic_core._pydantic_core.ValidationError) else None
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
        (groq.RateLimitError, groq.APITimeoutError, pydantic_core._pydantic_core.ValidationError),
        max_tries=BACKOFF_TOLERANCE,
        logger=None,
        on_giveup=lambda e: print(e) if isinstance(e, pydantic_core._pydantic_core.ValidationError) else None
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
    response = asyncio.run(GroqAPIProvider().async_chat_completion(
        messages=messages,
        model="llama3-8b-8192",
        temperature=0.0,
        max_tokens=150,
    ))
    print(response)