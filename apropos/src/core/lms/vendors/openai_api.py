import backoff
from networkx import within_inter_cluster
import openai
import pydantic_core
from openai import AsyncOpenAI, OpenAI
from apropos.src.core.lms.cache_init import cache
from apropos.src.core.lms.vendors.openai_like import OpenAIStandardProvider
from apropos.src.core.lms.vendors.json_structured_outputs.core import (
    add_json_instructions_to_messages,
    extract_pydantic_model_from_response_async,
    extract_pydantic_model_from_response_sync,
)
import instructor

BACKOFF_TOLERANCE = 0  # 20


class OpenAIAPIProvider(OpenAIStandardProvider):
    def __init__(self, multi_threaded=True):
        self.sync_client = OpenAI()
        self.async_client = AsyncOpenAI()
        self.multi_threaded = multi_threaded
        # super().__init__(self.sync_client, self.async_client, multi_threaded=multi_threaded)
        self.supports_response_model = True

    @backoff.on_exception(
        backoff.expo,
        (pydantic_core._pydantic_core.ValidationError,),
        max_tries=BACKOFF_TOLERANCE,
        logger=None,
        on_giveup=lambda e: print(e)
        if isinstance(e, pydantic_core._pydantic_core.ValidationError)
        else None,
    )
    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, pydantic_core._pydantic_core.ValidationError),
        max_tries=BACKOFF_TOLERANCE,
        logger=None,
        on_giveup=lambda e: print(e)
        if isinstance(e, pydantic_core._pydantic_core.ValidationError)
        else None,
    )
    async def async_chat_completion_with_response_model(
        self, messages, model, temperature, max_tokens, response_model
    ):
        if "o1" in model:
            message = f"<instructions>{messages[0]['content']}</instructions>\n\n<information>{messages[1]['content']}</information>"
            messages = [{"role": "user", "content": message}]
            temperature = 1
        reminder = False
        retries = 2
        succeeded = False
        while not succeeded and retries > 0:
            try:
                messages_with_json_formatting_instructions = (
                    add_json_instructions_to_messages(
                        messages, response_model, reminder
                    )
                )
                raw_text_api_response = await self.async_chat_completion(
                    messages_with_json_formatting_instructions,
                    model,
                    temperature,
                    max_tokens,
                )
                structured_api_response = (
                    await extract_pydantic_model_from_response_async(
                        raw_text_api_response, response_model
                    )
                )
                succeeded = True
            except Exception as e:
                print("Error in async_chat_completion_with_response_model: ", e)
                reminder = True
                temperature += 0.01
                retries -= 1
        return structured_api_response

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, pydantic_core._pydantic_core.ValidationError),
        max_tries=BACKOFF_TOLERANCE,
        logger=None,
        on_giveup=lambda e: print(e)
        if isinstance(e, pydantic_core._pydantic_core.ValidationError)
        else None,
    )
    def sync_chat_completion_with_response_model(
        self, messages, model, temperature, max_tokens, response_model
    ):
        if "o1" in model:
            message = f"<instructions>{messages[0]['content']}</instructions>\n\n<information>{messages[1]['content']}</information>"
            messages = [{"role": "user", "content": message}]
            temperature = 1
        reminder = False
        retries = 2
        succeeded = False
        while not succeeded and retries > 0:
            try:
                messages_with_json_formatting_instructions = (
                    add_json_instructions_to_messages(
                        messages, response_model, reminder
                    )
                )
                raw_text_api_response = self.sync_chat_completion(
                    messages_with_json_formatting_instructions,
                    model,
                    temperature,
                    max_tokens,
                )
                structured_api_response = extract_pydantic_model_from_response_sync(
                    raw_text_api_response, response_model
                )
                succeeded = True
            except Exception as e:
                print("Error in sync_chat_completion_with_response_model: ", e)
                reminder = True
                temperature += 0.01
                retries -= 1
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
    response = asyncio.run(
        OpenAIAPIProvider().async_chat_completion(
            messages=messages,
            model="o1-preview",
            temperature=0.0,
            max_tokens=500,
        )
    )
    print(response)
