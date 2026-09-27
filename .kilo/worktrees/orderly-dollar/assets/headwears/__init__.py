#!/usr/bin/env python3
"""
HEADWEARS REGISTRY
==================
Exports all headwear modules for the Face Puppet desktop pet.
"""

from .cap import CrimsonBeretHeadwear
from .christmashat import HolidaySantaHeadwear
from .hat2 import OfficerCapHeadwear
from .royalguards import RoyalGuardHeadwear

ALL_HEADWEARS = [
    CrimsonBeretHeadwear,
    HolidaySantaHeadwear,
    OfficerCapHeadwear,
    RoyalGuardHeadwear,
]


def load_all_headwears():
    """Loads and returns all active headwear instances."""
    results = []
    for cls in ALL_HEADWEARS:
        data = cls.load()
        if data is not None:
            results.append(data)
    return results
