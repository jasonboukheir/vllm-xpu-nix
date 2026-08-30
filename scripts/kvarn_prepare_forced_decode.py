#!/usr/bin/env python3
"""Prepare deterministic prompt and forced-token inputs for Kvarn logits gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, NamedTuple


class CaseSpec(NamedTuple):
    name: str
    category: str
    prompt_tokens: int
    decode_steps: int


CASE_SPECS = (
    CaseSpec("dialogue-127", "dialogue", 127, 1024),
    CaseSpec("adversarial-128", "adversarial", 128, 768),
    CaseSpec("code-4095", "code", 4095, 768),
    CaseSpec("math-16383", "math", 16383, 768),
    CaseSpec("reasoning-32767", "reasoning", 32767, 768),
    CaseSpec("reasoning-65023", "reasoning", 65023, 512),
)


def select_case_specs(names: list[str] | None) -> tuple[CaseSpec, ...]:
    if not names:
        return CASE_SPECS
    by_name = {case.name: case for case in CASE_SPECS}
    return tuple(by_name[name] for name in dict.fromkeys(names))


def load_fixtures(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError("fixtures must contain a JSON list")
    fixtures: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, dict):
            raise TypeError("each fixture must be a JSON object")
        category = item.get("category")
        prompt = item.get("prompt")
        if not isinstance(category, str) or not isinstance(prompt, str):
            raise TypeError("each fixture requires string category and prompt fields")
        if category in fixtures:
            raise ValueError(f"duplicate fixture category: {category}")
        fixtures[category] = item
    missing = sorted({case.category for case in CASE_SPECS} - fixtures.keys())
    if missing:
        raise ValueError("fixtures are missing categories: " + ", ".join(missing))
    return fixtures


def exact_prompt_ids(
    tokenizer: Any,
    prompt: str,
    category: str,
    target: int,
) -> list[int]:
    ids = tokenizer.encode(prompt)
    counter = 0
    while len(ids) < target:
        digest = hashlib.sha256(f"{category}:{counter}".encode()).hexdigest()
        record = (
            f"\nCategory {category} evidence record {counter}; "
            f"stable digest {digest}; retain its distinct facts and order."
        )
        ids.extend(tokenizer.encode(record, add_special_tokens=False))
        counter += 1
    return ids[:target]


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    # Import vLLM only in the guarded parent process. XPU uses multiprocessing
    # spawn, whose child imports this file as __mp_main__ during bootstrap.
    from vllm import LLM, SamplingParams, TokensPrompt

    case_specs = select_case_specs(args.case)
    fixtures = load_fixtures(args.fixtures)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype="bfloat16",
        quantization="compressed-tensors",
        kv_cache_dtype="auto",
        max_model_len=65536,
        max_num_seqs=1,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        enable_prefix_caching=False,
        language_model_only=True,
    )
    tokenizer = llm.get_tokenizer()

    manifest: list[dict[str, Any]] = []
    for case in case_specs:
        case_dir = args.output_dir / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        prompt_ids = exact_prompt_ids(
            tokenizer,
            fixtures[case.category]["prompt"],
            case.category,
            case.prompt_tokens,
        )
        params = SamplingParams(
            temperature=0.0,
            max_tokens=case.decode_steps,
            min_tokens=case.decode_steps,
            ignore_eos=True,
            detokenize=False,
        )
        request = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)],
            params,
            use_tqdm=False,
        )[0]
        forced_ids = list(request.outputs[0].token_ids)
        if len(prompt_ids) != case.prompt_tokens:
            raise AssertionError(f"{case.name}: wrong prompt-token count")
        if len(forced_ids) != case.decode_steps:
            raise AssertionError(f"{case.name}: wrong forced-token count")

        prompt_json = json.dumps(prompt_ids, separators=(",", ":")) + "\n"
        forced_json = json.dumps(forced_ids, separators=(",", ":")) + "\n"
        (case_dir / "prompt-token-ids.json").write_text(prompt_json, encoding="utf-8")
        (case_dir / "forced-token-ids.json").write_text(forced_json, encoding="utf-8")
        service_fixture_json = (
            json.dumps(
                [
                    {
                        "id": case.name,
                        "category": case.category,
                        "max_tokens": case.decode_steps,
                        "prompt": prompt_ids,
                    }
                ],
                separators=(",", ":"),
            )
            + "\n"
        )
        (case_dir / "service-fixture.json").write_text(
            service_fixture_json, encoding="utf-8"
        )
        manifest.append(
            {
                "name": case.name,
                "category": case.category,
                "prompt_tokens": case.prompt_tokens,
                "decode_steps": case.decode_steps,
                "prompt_token_ids_sha256": hashlib.sha256(
                    prompt_json.encode()
                ).hexdigest(),
                "forced_token_ids_sha256": hashlib.sha256(
                    forced_json.encode()
                ).hexdigest(),
                "service_fixture_sha256": hashlib.sha256(
                    service_fixture_json.encode()
                ).hexdigest(),
            }
        )

    if not args.case and sum(case["decode_steps"] for case in manifest) < 4608:
        raise AssertionError("forced-decode cases cover fewer than 4608 positions")
    (args.output_dir / "cases.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in CASE_SPECS],
        help="prepare only the selected case; may be repeated",
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
