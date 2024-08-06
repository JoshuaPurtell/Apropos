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

DISKCACHE_SIZE_LIMIT = 10*1024*1024*1024  # 10 GB

@dataclass
class SafeCache:
    def __init__(self, fast_cache_dir, slow_cache_db):
        self.fast_cache = Cache(fast_cache_dir, size_limit=DISKCACHE_SIZE_LIMIT)
        self.conn = sqlite3.connect(slow_cache_db)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS cache
                              (key TEXT PRIMARY KEY, response TEXT, information TEXT)''')
        self.conn.commit()

    def get_cache_key(self, messages: List[Dict], model: str, temperature: float, response_model: Type[BaseModel]):
        normalized_messages = "".join([msg["content"] for msg in messages])
        normalized_model = model
        normalized_temperature = f"{temperature:.2f}"[:4]
        normalized_response_model = str(response_model.schema()) if response_model else ""
        return hashlib.sha256((normalized_messages+normalized_model+normalized_temperature+normalized_response_model).encode()).hexdigest()

    def hit_cache(self, messages, model, temperature, response_model):
        fast_result = self.hit_cache_fast(messages, model, temperature, response_model)
        if fast_result and isinstance(fast_result, dict):
            return fast_result["response"]
        slow_result = self.hit_cache_slow(messages, model, temperature, response_model)
        if slow_result:
            return slow_result
        return None

    def hit_cache_fast(self, messages, model, temperature, response_model):
        key = self.get_cache_key(messages, model, temperature, response_model)
        if key in self.fast_cache:
            return self.fast_cache[key]
        return None

    def hit_cache_slow(self, messages, model, temperature, response_model):
        key = self.get_cache_key(messages, model, temperature, response_model)
        self.cursor.execute("SELECT response FROM cache WHERE key = ?", (key,))
        result = self.cursor.fetchone()
        if result:
            return json.loads(result[0])
        return None
    
    def hit_cache_brute_force(self, messages, model, temperature, response_model):
        target_features = self.get_cached_features(messages, model, temperature, response_model)
        target_system = target_features["system"]
        target_user = target_features["user"]
        target_model = target_features["model"]
        
        self.cursor.execute("SELECT response, information FROM cache")
        results = self.cursor.fetchall()
        
        for response, information in results:
            info = json.loads(information)
            if (info["system"] == target_system and
                info["user"] == target_user and
                info["model"] == target_model):
                return json.loads(response)
        return None
    
    def get_cached_features(self, messages: List[Dict], model: str, temperature: float, response_model: Type[BaseModel]):
        return {
            "system":hashlib.sha256((messages[0]["content"]).encode()).hexdigest(),
            "user":hashlib.sha256((messages[1]["content"]).encode()).hexdigest(),
            "model":model,
            "temperature": f"{temperature:.2f}"[:4]
        }

    def add_to_cache(self, messages, model, temperature, response_model, response):
        key = self.get_cache_key(messages, model, temperature, response_model)
        information = json.dumps(self.get_cached_features(messages, model, temperature, response_model))

        self.fast_cache[key] = {"response": response, "information": information}
        self.cursor.execute("INSERT OR REPLACE INTO cache (key, response, information) VALUES (?, ?, ?)",
                            (key, json.dumps(response), information))
        self.conn.commit()
        return key