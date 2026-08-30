import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "kvarn_bf16_matmul_replay.py"
SPEC = importlib.util.spec_from_file_location("kvarn_bf16_matmul_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_shape_accepts_positive_mkn():
    assert MODULE.parse_shape("107x17408x5120") == (107, 17408, 5120)


@pytest.mark.parametrize("value", ["107x5120", "107xzeroX5120", "1x0x1"])
def test_parse_shape_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        MODULE.parse_shape(value)
