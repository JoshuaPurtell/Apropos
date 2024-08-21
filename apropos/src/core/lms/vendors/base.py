from abc import ABC, abstractmethod


class BaseProvider(ABC):
    @abstractmethod
    async def async_chat_completion(self, messages, model, temperature, max_tokens):
        pass

    @abstractmethod
    def sync_chat_completion(self, messages, model, temperature, max_tokens):
        pass
