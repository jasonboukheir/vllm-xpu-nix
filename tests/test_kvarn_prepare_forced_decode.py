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


def test_case_specs_cover_boundaries_and_near_maximum_context():
    assert sum(case.decode_steps for case in MODULE.CASE_SPECS) == 4608
    assert {
        (case.prompt_tokens, case.prompt_tokens + 1) for case in MODULE.CASE_SPECS
    } >= {
        (127, 128),
        (128, 129),
        (4095, 4096),
        (16383, 16384),
        (32767, 32768),
        (65023, 65024),
    }
    near_max = next(
        case for case in MODULE.CASE_SPECS if case.name == "reasoning-65023"
    )
    assert near_max.prompt_tokens + near_max.decode_steps == 65535


def test_load_fixtures_covers_every_case_category():
    fixtures = MODULE.load_fixtures(FIXTURES)
    assert {case.category for case in MODULE.CASE_SPECS} <= fixtures.keys()


def test_select_case_specs_preserves_order_and_deduplicates():
    selected = MODULE.select_case_specs(
        ["reasoning-65023", "dialogue-127", "reasoning-65023"]
    )
    assert [case.name for case in selected] == ["reasoning-65023", "dialogue-127"]
    assert MODULE.select_case_specs(None) == MODULE.CASE_SPECS


@pytest.mark.parametrize("target", [127, 128, 4095])
def test_exact_prompt_ids_reaches_requested_length_deterministically(target):
    tokenizer = Tokenizer()
    first = MODULE.exact_prompt_ids(tokenizer, "short", "dialogue", target)
    second = MODULE.exact_prompt_ids(tokenizer, "short", "dialogue", target)
    assert first == second
    assert len(first) == target


def test_exact_prompt_ids_can_end_with_meaningful_prompt():
    tokenizer = Tokenizer()
    prompt = "Give a varied, internally consistent analysis."
    suffix = "\n\nFinal task after reviewing the records:\n" + prompt
    ids = MODULE.exact_prompt_ids(
        tokenizer,
        prompt,
        "reasoning",
        4096,
        trailing_prompt=True,
    )
    assert len(ids) == 4096
    assert ids[-len(suffix.encode()) :] == list(suffix.encode())


def test_exact_prompt_ids_rejects_trailing_prompt_larger_than_target():
    with pytest.raises(ValueError, match="too short"):
        MODULE.exact_prompt_ids(
            Tokenizer(),
            "long prompt",
            "reasoning",
            4,
            trailing_prompt=True,
        )
