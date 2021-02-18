# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for surface_rl_decoder.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import pytest
import numpy as np
from src.surface_rl_decoder.surface_code import SurfaceCode


@pytest.fixture
def sc():
    return SurfaceCode()


@pytest.fixture
def qbs():
    return np.zeros(shape=(8, 5, 5), dtype=np.uint8)
