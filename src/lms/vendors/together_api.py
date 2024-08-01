from src.lms.vendors.openai_like import OpenAIStandardProvider
from together import AsyncTogether, Together
import instructor
import os

class TogetherAPIProvider(OpenAIStandardProvider):
    def __init__(self):
        self.sync_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        self.async_client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"))
        self.supports_response_model = False

if __name__ == "__main__":
    import asyncio
    messages = [
        {
            "role": "system",
            "content": "You are a helpful a ssistant that can answer questions abou t the capital of France."
        },
        {
            "role": "user",
            "content": "What  is the capital of France?   "
        }
    ]
    response = asyncio.run(TogetherAPIProvider().async_chat_completion(
        messages=messages,
        model="teknium/OpenHermes-2p5-Mistral-7B",
        temperature=0.0,
        max_tokens=150,
    ))
    print(response)