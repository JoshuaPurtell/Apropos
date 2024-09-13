from typing import Dict, List, Type
from dataclasses import dataclass
import json

import loguru
import sqlite3
import json
from pydantic import BaseModel
from typing import Dict, List, Type
from dataclasses import dataclass
import json
import loguru
from diskcache import Cache
from pydantic import BaseModel
import hashlib
import time

logger = loguru.logger

DISKCACHE_SIZE_LIMIT = 10 * 1024 * 1024 * 1024  # 10 GB


@dataclass
class SafeCache:
    def __init__(self, fast_cache_dir, slow_cache_db):
        self.fast_cache = Cache(fast_cache_dir, size_limit=DISKCACHE_SIZE_LIMIT)
        self.conn = sqlite3.connect(slow_cache_db)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS cache
                              (key TEXT PRIMARY KEY, response TEXT, information TEXT)""")
        self.conn.commit()

    def get_cache_key(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        response_model: Type[BaseModel],
    ):
        if not all([isinstance(msg["content"], str) for msg in messages]):
            normalized_messages = "".join([str(msg["content"]) for msg in messages])
        else:
            normalized_messages = "".join([msg["content"] for msg in messages])
        normalized_model = model
        normalized_temperature = f"{temperature:.2f}"[:4]
        normalized_response_model = (
            str(response_model.schema()) if response_model else ""
        )
        return hashlib.sha256(
            (
                normalized_messages
                + normalized_model
                + normalized_temperature
                + normalized_response_model
            ).encode()
        ).hexdigest()

    def hit_cache(self, messages, model, temperature, response_model):
        fast_result = self.hit_cache_fast(messages, model, temperature, response_model)
        if fast_result and isinstance(fast_result, dict) and "response" in fast_result:
            fast_result = fast_result["response"]
        if fast_result and isinstance(fast_result, dict):
            return response_model(**fast_result)
        elif fast_result:
            return fast_result
        slow_result = self.hit_cache_slow(messages, model, temperature, response_model)
        if slow_result:
            return (
                slow_result
                if not "response" in slow_result
                else slow_result["response"]
            )
        return None

    def hit_cache_fast(self, messages, model, temperature, response_model):
        key = self.get_cache_key(messages, model, temperature, response_model)
        if key in self.fast_cache:
            try:
                cache_data = self.fast_cache[key]
            except AttributeError:
                return None
            if response_model is not None:
                if isinstance(cache_data["response"], dict):
                    response = cache_data["response"]
                    return response_model(**(response))
            return cache_data["response"]
        return None

    def hit_cache_slow(self, messages, model, temperature, response_model):
        key = self.get_cache_key(messages, model, temperature, response_model)
        self.cursor.execute("SELECT response FROM cache WHERE key = ?", (key,))
        result = self.cursor.fetchone()
        if result:
            return json.loads(result[0])
        return None

    def hit_cache_brute_force(self, messages, model, temperature, response_model):
        target_features = self.get_cached_features(
            messages, model, temperature, response_model
        )
        target_system = target_features["system"]
        target_user = target_features["user"]
        target_model = target_features["model"]

        self.cursor.execute("SELECT response, information FROM cache")
        results = self.cursor.fetchall()

        for response, information in results:
            info = json.loads(information)
            if (
                info["system"] == target_system
                and info["user"] == target_user
                and info["model"] == target_model
            ):
                return json.loads(response)
        return None

    def get_cached_features(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        response_model: Type[BaseModel],
    ):
        if len(messages) == 2:
            return {
                "system": hashlib.sha256((messages[0]["content"]).encode()).hexdigest(),
                "user": hashlib.sha256(
                    (str(messages[1]["content"])).encode()
                ).hexdigest(),
                "model": model,
                "temperature": f"{temperature:.2f}"[:4],
            }
        elif len(messages) == 1:
            return {
                "system": hashlib.sha256("".encode()).hexdigest(),
                "user": hashlib.sha256((messages[0]["content"]).encode()).hexdigest(),
                "model": model,
                "temperature": f"{temperature:.2f}"[:4],
            }

    def add_to_cache(self, messages, model, temperature, response_model, response):
        key = self.get_cache_key(messages, model, temperature, response_model)
        information = json.dumps(
            self.get_cached_features(messages, model, temperature, response_model)
        )

        if isinstance(response, BaseModel):
            response_dict = response.model_dump()
            response_class = response.__class__.__name__
        else:
            response_dict = response
            response_class = None

        cache_data = {
            "response": response_dict,
            "response_class": response_class,
            "information": information,
        }

        self.fast_cache[key] = cache_data
        self.cursor.execute(
            "INSERT OR REPLACE INTO cache (key, response, information) VALUES (?, ?, ?)",
            (key, json.dumps(cache_data), information),
        )
        self.conn.commit()
        return key

    def dump_fast_to_slow(self):
        for key in self.fast_cache:
            cache_data = self.fast_cache[key]
            self.cursor.execute("SELECT key FROM cache WHERE key = ?", (key,))
            if not self.cursor.fetchone():
                self.cursor.execute(
                    "INSERT INTO cache (key, response, information) VALUES (?, ?, ?)",
                    (
                        key,
                        json.dumps(cache_data["response"]),
                        cache_data["information"],
                    ),
                )
        self.conn.commit()

    def close(self):
        self.dump_fast_to_slow()
        self.fast_cache.close()
        self.conn.close()


@dataclass
class ThreadedCache:
    def __init__(self, fast_cache_dir):
        self.fast_cache = Cache(fast_cache_dir, size_limit=DISKCACHE_SIZE_LIMIT)

    def hit_cache(self, messages, model, temperature, response_model):
        key = self.get_cache_key(messages, model, temperature, response_model)
        if key in self.fast_cache:
            try:
                cache_data = self.fast_cache[key]
            except AttributeError:
                return None
            if response_model is not None:
                if isinstance(cache_data["response"], dict):
                    response = cache_data["response"]
                    return response_model(**response)
            if isinstance(cache_data, str):
                cache_data = {
                    "response": cache_data,
                    "response_class": None,
                    "information": None,
                }
            return cache_data["response"]
        return None

    def add_to_cache(self, messages, model, temperature, response_model, response):
        key = self.get_cache_key(messages, model, temperature, response_model)
        information = json.dumps(
            self.get_cached_features(messages, model, temperature, response_model)
        )

        if isinstance(response, BaseModel):
            response_dict = response.model_dump()
            response_class = response.__class__.__name__
        else:
            response_dict = response
            response_class = None

        cache_data = {
            "response": response_dict,
            "response_class": response_class,
            "information": information,
        }

        self.fast_cache[key] = cache_data
        return key

    def get_cached_features(self, messages, model, temperature, response_model):
        if len(messages) == 2:
            return {
                "system": hashlib.sha256((messages[0]["content"]).encode()).hexdigest(),
                "user": hashlib.sha256(
                    (str(messages[1]["content"])).encode()
                ).hexdigest(),
                "model": model,
                "temperature": f"{temperature:.2f}"[:4],
            }
        elif len(messages) == 1:
            return {
                "system": hashlib.sha256("".encode()).hexdigest(),
                "user": hashlib.sha256((messages[0]["content"]).encode()).hexdigest(),
                "model": model,
                "temperature": f"{temperature:.2f}"[:4],
            }

    def close(self):
        self.fast_cache.close()

    def get_cache_key(self, messages, model, temperature, response_model):
        if not all([isinstance(msg["content"], str) for msg in messages]):
            normalized_messages = "".join([str(msg["content"]) for msg in messages])
        else:
            normalized_messages = "".join([msg["content"] for msg in messages])
        normalized_model = model
        normalized_temperature = f"{temperature:.2f}"[:4]
        normalized_response_model = (
            str(response_model.schema()) if response_model else ""
        )
        return hashlib.sha256(
            (
                normalized_messages
                + normalized_model
                + normalized_temperature
                + normalized_response_model
            ).encode()
        ).hexdigest()
