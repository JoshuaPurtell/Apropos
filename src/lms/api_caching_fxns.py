import hashlib
from typing import Dict, List, Type, Union
from dataclasses import dataclass
import os
import glob
from base64 import b85encode, b85decode
import zlib
import json

import loguru
from diskcache import Cache
from pydantic import BaseModel

logger = loguru.logger

def get_cache_key_legacy(messages: List[Dict], model: str, temperature: float, response_model: Type[BaseModel]):
    normalized_messages = "".join([msg["content"] for msg in messages])
    normalized_model = model
    normalized_temperature = "{:.1000g}".format(temperature)
    normalized_response_model = str(response_model.schema()) if response_model else ""
    if response_model is None:
        return hashlib.sha256((normalized_messages+normalized_model+normalized_temperature).encode()).hexdigest()
    else:
        return hashlib.sha256((normalized_messages+normalized_model+normalized_temperature+normalized_response_model).encode()).hexdigest()


def zip_info(information: Dict):
    return b85encode(zlib.compress(json.dumps(information).encode())).decode()

def response_is_valid(response, response_model):
    if response_model is None and isinstance(response, str) or isinstance(response, list):
        return True
    elif response_model and isinstance(response, Dict):
        return True
    else:
        return False

@dataclass
class SafeCache:
    def get_cache_key_modern(self, messages: List[Dict], model: str, temperature: float, response_model: Type[BaseModel]):
        normalized_messages = "".join([msg["content"] for msg in messages])
        normalized_model = model
        normalized_temperature = f"{temperature:.2f}"[:4]
        normalized_response_model = str(response_model.schema()) if response_model else ""
        return hashlib.sha256((normalized_messages+normalized_model+normalized_temperature+normalized_response_model).encode()).hexdigest()
    
    def get_cache_key(self, messages: List[Dict], model: str, temperature: float, response_model: Type[BaseModel]):
        return self.get_cache_key_modern(messages, model, temperature, response_model), get_cache_key_legacy(messages, model, temperature, response_model)

    def get_cached_features(self, messages: List[Dict], model: str, temperature: float, response_model: Type[BaseModel]):
        return {
            "system":hashlib.sha256((messages[0]["content"]).encode()).hexdigest(),
            "user":hashlib.sha256((messages[1]["content"]).encode()).hexdigest(),
            "model":model,
            "temperature": f"{temperature:.2f}"[:4]
        }

    def hit_cache(self, messages, model, temperature, response_model, cache: Cache, legacy_cache: Cache):
        key, legacy_key = self.get_cache_key(messages, model, temperature, response_model)
        
        if key in cache and "response" in cache[key] and response_is_valid(cache[key]["response"], response_model):
            if isinstance(cache[key], Dict):
                if len(cache[key]) == 2:
                    return cache[key]["response"] if isinstance(cache[key]["response"], str) or isinstance(cache[key]["response"], list) else response_model(**cache[key]["response"])
                else:
                    return None
            else:
                return cache[key]
        elif legacy_key in legacy_cache and "response" in legacy_cache[legacy_key] and isinstance(legacy_cache[legacy_key],Dict) and len(legacy_cache[legacy_key])==2 and response_is_valid(legacy_cache[legacy_key]["response"], response_model):
            print("Hit legacy key")
            if isinstance(legacy_cache[legacy_key],Dict) and len(legacy_cache[legacy_key])<2:
                return None
            elif isinstance(legacy_cache[legacy_key], Dict):
                cache[key] = legacy_cache[legacy_key]
            else:
                cache[key] = {}
                cache[key]["response"] = legacy_cache[legacy_key]
                cache[key]["information"] =  self.get_cached_features(messages, model, temperature, response_model)
            assert isinstance(cache[key], Dict), cache[key]
            assert len(cache[key]) == 2, f"Failed on {len(cache[key])}, {type(cache[key])} - {cache[key]}"
            assert 'response' in cache[key], cache[key]
            assert 'information' in cache[key], cache[key]
            return cache[key]["response"] if isinstance(cache[key]["response"], str) or isinstance(cache[key]["response"], list) else response_model(**cache[key]["response"])
        else:
            return None

    def add_to_cache(self, messages: List[Dict], model: str, temperature: float, response_model: Type[BaseModel], response: Union[str, BaseModel], cache: Cache):
        key, legacy_key= self.get_cache_key(messages, model, temperature, response_model)
        if isinstance(response, str) or isinstance(response, list):
            response_to_store = response
        else:
            response_to_store = response.dict()
        cache[key] = {
            "information": self.get_cached_features(messages, model, temperature, response_model),
            "response": response_to_store
        }
        return key


def print_cache_folder_sizes():
    cache_folders = glob.glob(".*_cache")
    for folder in cache_folders:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        print(f"{folder}: {total_size / (1024 * 1024):.2f} MB")

safecache_wrapper = SafeCache()