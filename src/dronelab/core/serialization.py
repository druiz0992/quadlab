"""JSON encoding of DroneLab's immutable measurement and context types.

``encode`` turns frozen dataclasses (sensor payloads, ``Sample``,
``EstimatedState``, ``AlgorithmContext``, vehicle properties), NumPy arrays,
tuples, and mappings into JSON-compatible values that ``decode`` restores
exactly: floats round-trip bit-for-bit through JSON's shortest repr, arrays
keep dtype and shape, and tuples stay tuples.

Fields with ``metadata={"serialize": False}`` (callbacks such as the
context's debug log) are skipped and take their defaults when decoded.

Decoding imports dataclasses by name, so it only accepts classes from the
``dronelab`` package (a recorded file cannot name arbitrary code) and
raises ValueError otherwise.
"""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import importlib
from typing import Any

import numpy as np

ALLOWED_PREFIX = "dronelab."


def encode(value: Any) -> Any:
    """JSON-compatible form of ``value`` (see module docs)."""
    if is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        return {
            "__dataclass__": f"{cls.__module__}:{cls.__qualname__}",
            "fields": {
                f.name: encode(getattr(value, f.name))
                for f in fields(value)
                if f.metadata.get("serialize", True)
            },
        }
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": value.tolist(),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, Mapping):
        keys = list(value)
        if not all(isinstance(k, str) for k in keys):
            raise ValueError("Only string-keyed mappings can be encoded")
        return {"__mapping__": {k: encode(value[k]) for k in keys}}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"Cannot encode {type(value).__name__}")


def decode(value: Any) -> Any:
    """Restore a value produced by ``encode``; ValueError on foreign types."""
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "__dataclass__" in value:
        cls = _dataclass(value["__dataclass__"])
        return cls(**{k: decode(v) for k, v in value["fields"].items()})
    if "__ndarray__" in value:
        array = np.asarray(value["__ndarray__"], dtype=value["dtype"])
        return array.reshape(value["shape"])
    if "__tuple__" in value:
        return tuple(decode(v) for v in value["__tuple__"])
    if "__mapping__" in value:
        return {k: decode(v) for k, v in value["__mapping__"].items()}
    raise ValueError(f"Unknown encoded object with keys {sorted(value)}")


def _dataclass(name: str) -> type:
    module_name, _, qualname = name.partition(":")
    if not module_name.startswith(ALLOWED_PREFIX) or not qualname:
        raise ValueError(f"Refusing to decode {name!r}: not a dronelab type")
    cls: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        cls = getattr(cls, part)
    if not (isinstance(cls, type) and is_dataclass(cls)):
        raise ValueError(f"{name!r} is not a dataclass")
    return cls
