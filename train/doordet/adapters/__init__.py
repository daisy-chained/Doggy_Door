"""Model families. An adapter has: model, collate(batch), loss(inputs, device), predict(inputs, device), save(path)."""
from .hf import HFAdapter

FAMILIES = {"hf": HFAdapter}


def get(family):
    if family not in FAMILIES:
        raise ValueError(f"unknown model family {family!r}; available: {', '.join(FAMILIES)}")
    return FAMILIES[family]
