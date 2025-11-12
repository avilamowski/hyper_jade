from typing import Any
def keep_last(existing: Any, new: Any) -> Any:
    """Reducer that keeps the first value and discards subsequent ones"""
    # return existing if existing is not None else new
    return new

def concat(
    existing: list, new: list
) -> list:
    """Custom aggregation function for generated_prompts"""
    if existing is None:
        existing = []
    if new is None:
        new = []
    return existing + new

def merge_dicts(existing: dict, new: dict) -> dict:
    """Reducer that merges dictionaries (new values override existing)"""
    if existing is None:
        existing = {}
    if new is None:
        new = {}
    return {**existing, **new}
