import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "kvarn_prepare_forced_decode.py"
SPEC = importlib.util.spec_from_file_location("kvarn_prepare_forced_decode", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return list(text.encode())


def test_select_case_specs_preserves_order_and_deduplicates():
    selected = MODULE.select_case_specs(
        ["reasoning-65023", "dialogue-127", "reasoning-65023"]
    )
    assert [case.name for case in selected] == ["reasoning-65023", "dialogue-127"]
    assert MODULE.select_case_specs(None) == MODULE.CASE_SPECS


def test_exact_prompt_ids_reaches_requested_length_deterministically():
    target = 257
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


def test_materialize_service_fixtures_writes_every_selected_case(tmp_path):
    selected = (
        MODULE.CaseSpec("first", "dialogue", 3, 7),
        MODULE.CaseSpec("second", "code", 5, 9),
    )
    for case in selected:
        case_dir = tmp_path / case.name
        case_dir.mkdir()
        prompt_ids = [case.prompt_tokens] * case.prompt_tokens
        (case_dir / "prompt-token-ids.json").write_text(
            json.dumps(prompt_ids) + "\n", encoding="utf-8"
        )

    manifest = MODULE.materialize_service_fixtures(tmp_path, selected)

    assert [item["name"] for item in manifest] == ["first", "second"]
    for case in selected:
        fixture_path = tmp_path / case.name / "service-fixture.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        assert fixture == [
            {
                "id": case.name,
                "category": case.category,
                "max_tokens": case.decode_steps,
                "prompt": [case.prompt_tokens] * case.prompt_tokens,
            }
        ]
