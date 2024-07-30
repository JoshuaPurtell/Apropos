import hashlib
from typing import Dict, List, Type

import loguru
from diskcache import Cache
from pydantic import BaseModel

logger = loguru.logger

# Create a cache object
cache = Cache(directory=".cache")


def generate_cache_key(messages: List[Dict], model: str, temperature: float) -> str:
    model = model
    key = (
        "".join([msg["content"] for msg in messages])
        + model
        + "{:.1000g}".format(temperature)
    )
    return hashlib.sha256(key.encode()).hexdigest()


def generate_cache_key_with_response_model(
    messages: List[Dict],
    model: str,
    temperature: float,
    response_model: Type[BaseModel],
) -> str:
    key = (
        "".join([msg["content"] for msg in messages])
        + model
        + "{:.10g}".format(temperature)
        + str(response_model.schema())
    )
    return hashlib.sha256(key.encode()).hexdigest()
