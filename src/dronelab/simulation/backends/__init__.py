"""Factory for the step-based flight experiment."""

from .reference import ReferenceBackend
from .rotorpy import RotorPyBackend


def make_backend(name):
    if name == "rotorpy":
        return RotorPyBackend()
    if name == "reference":
        return ReferenceBackend()
    raise ValueError(f"Unknown backend: {name}")
