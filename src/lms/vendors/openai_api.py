from src.lms.vendors.openai_like import OpenAIStandardProvider
from openai import OpenAI, AsyncOpenAI
import pydantic_core
import backoff
import openai
from src.lms.vendors.json_structured_outputs.core import add_json_instructions_to_messages, extract_pydantic_model_from_response

BACKOFF_TOLERANCE = 15

class OpenAIAPIProvider(OpenAIStandardProvider):
    def __init__(self):
        self.sync_client = OpenAI()
        self.async_client = AsyncOpenAI()
        self.supports_response_model = True

    @backoff.on_exception(
            backoff.expo,
            (openai.RateLimitError, pydantic_core._pydantic_core.ValidationError),
            max_tries=BACKOFF_TOLERANCE,
            logger=None,
            on_giveup=lambda e: print(e) if isinstance(e, pydantic_core._pydantic_core.ValidationError) else None
        )
    async def async_chat_completion_with_response_model(self, messages, model, temperature, max_tokens, response_model):
        messages_with_json_formatting_instructions = add_json_instructions_to_messages(messages, response_model)
        raw_text_api_response = await self.async_chat_completion(messages_with_json_formatting_instructions, model, temperature, max_tokens)
        structured_api_response = await extract_pydantic_model_from_response(raw_text_api_response, response_model)
        return structured_api_response
    @backoff.on_exception(
            backoff.expo,
            (openai.RateLimitError, pydantic_core._pydantic_core.ValidationError),
            max_tries=BACKOFF_TOLERANCE,
            logger=None,
            on_giveup=lambda e: print(e) if isinstance(e, pydantic_core._pydantic_core.ValidationError) else None
        )
    def sync_chat_completion_with_response_model(self, messages, model, temperature, max_tokens, response_model):
        messages_with_json_formatting_instructions = add_json_instructions_to_messages(messages, response_model)
        raw_text_api_response = self.sync_chat_completion(messages_with_json_formatting_instructions, model, temperature, max_tokens)
        structured_api_response = extract_pydantic_model_from_response(raw_text_api_response, response_model)
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
    response = asyncio.run(OpenAIAPIProvider().async_chat_completion(
        messages=messages,
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=150,
    ))
    print(response)