"""
State store abstraction for TriageSim.

This module provides:
- RedisStateStore: persistence via Redis
- InMemoryStateStore: an in-memory dictionary fallback for quick prototyping

Both implement the same interface:
    set_json(key, value)
    get_json(key)
    append_list(key, value)
    get_list(key)
    delete(key)
    flush()
"""

from typing import Any, Dict, List
import json

try:
    import redis  # type: ignore
except ImportError:
    redis = None


class StateStore:
    """
    Abstract base for a state store.
    """

    def set_json(self, key: str, value: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_json(self, key: str) -> Dict[str, Any]:
        raise NotImplementedError

    def append_list(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def get_list(self, key: str) -> List[Any]:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError


class RedisStateStore(StateStore):
    """
    Redis backed state store.

    Uses JSON encoding for structured data.

    Keys in Redis will be stored as strings, values as JSON strings,
    and lists as Redis lists of JSON strings.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        decode_responses: bool = True,
    ):
        if redis is None:
            raise ImportError(
                "The redis package is required for RedisStateStore. Install with `pip install redis`."
            )
        self.client = redis.Redis(
            host=host, port=port, db=db, decode_responses=decode_responses
        )

    def set_json(self, key: str, value: Dict[str, Any]) -> None:
        """
        Store a JSON blob at the given key.
        """
        self.client.set(key, json.dumps(value))

    def get_json(self, key: str) -> Dict[str, Any]:
        """
        Get JSON from the given key. Returns {} if not found.
        """
        raw = self.client.get(key)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def append_list(self, key: str, value: Any) -> None:
        """
        Append an element to a Redis list at key.
        The element will be JSON serialized.
        """
        self.client.rpush(key, json.dumps(value))

    def get_list(self, key: str) -> List[Any]:
        """
        Retrieve a list stored at key. Empty list if not found.
        """
        items = self.client.lrange(key, 0, -1)
        result: List[Any] = []
        for item in items:
            try:
                result.append(json.loads(item))
            except json.JSONDecodeError:
                continue
        return result

    def delete(self, key: str) -> None:
        """
        Delete a key and its contents.
        """
        self.client.delete(key)

    def flush(self) -> None:
        """
        Clear all keys in the current database.
        """
        self.client.flushdb()


class InMemoryStateStore(StateStore):
    """
    An in-memory state store for testing and quick prototyping.

    Stores keys → JSON strings in a dict, and lists → list of JSON strings.
    """

    def __init__(self):
        self._data: Dict[str, str] = {}
        self._lists: Dict[str, List[str]] = {}

    def set_json(self, key: str, value: Dict[str, Any]) -> None:
        self._data[key] = json.dumps(value)

    def get_json(self, key: str) -> Dict[str, Any]:
        raw = self._data.get(key)
        if raw is None:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def append_list(self, key: str, value: Any) -> None:
        """
        Append an element to an in-memory list.
        """
        self._lists.setdefault(key, []).append(json.dumps(value))

    def get_list(self, key: str) -> List[Any]:
        raw_list = self._lists.get(key, [])
        result: List[Any] = []
        for item in raw_list:
            try:
                result.append(json.loads(item))
            except json.JSONDecodeError:
                continue
        return result

    def delete(self, key: str) -> None:
        self._data.pop(key, None)
        self._lists.pop(key, None)

    def flush(self) -> None:
        """
        Clears all in-memory state.
        """
        self._data.clear()
        self._lists.clear()
