"""Load the numbered pipeline scripts as modules so their functions can be tested."""

import importlib.util
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def load_step(filename):
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), SRC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def sentinel():
    return load_step("02_clean_sentinel_values.py")


@pytest.fixture(scope="session")
def cpi():
    return load_step("03_adjust_income_cpi.py")


@pytest.fixture(scope="session")
def crosswalk():
    return load_step("04_crosswalk_to_2010.py")


@pytest.fixture(scope="session")
def cleanup():
    return load_step("04d_crosswalk_cleanup.py")


@pytest.fixture(scope="session")
def stack():
    return load_step("05_stack_longitudinal.py")


@pytest.fixture(scope="session")
def hazard():
    return load_step("10_hazard_congruence.py")


@pytest.fixture(scope="session")
def verify():
    return load_step("11_verify_section_6_3.py")
