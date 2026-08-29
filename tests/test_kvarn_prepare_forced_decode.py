import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "kvarn_prepare_forced_decode.py"
FIXTURES = ROOT / "fixtures" / "kvarn-long-generation.json"
SPEC = importlib.util.spec_from_file_location("kvarn_prepare_forced_decode", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return list(text.encode())


def test_case_specs_cover_boundaries_and_4096_decode_positions():
    assert sum(case.decode_steps for case in MODULE.CASE_SPECS) == 4096
    assert {
        (case.prompt_tokens, case.prompt_tokens + 1)
        for case in MODULE.CASE_SPECS
    } >= {
        (127, 128),
        (128, 129),
        (4095, 4096),
        (16383, 16384),
        (32767, 32768),
    }


def test_load_fixtures_covers_every_case_category():
    fixtures = MODULE.load_fixtures(FIXTURES)
    assert {case.category for case in MODULE.CASE_SPECS} <= fixtures.keys()


@pytest.mark.parametrize("target", [127, 128, 4095])
def test_exact_prompt_ids_reaches_requested_length_deterministically(target):
    tokenizer = Tokenizer()
    first = MODULE.exact_prompt_ids(tokenizer, "short", "dialogue", target)
    second = MODULE.exact_prompt_ids(tokenizer, "short", "dialogue", target)
    assert first == second
    assert len(first) == target
