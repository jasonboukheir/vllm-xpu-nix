"""Chunk-budget contract shared by service captures and trace attribution."""


def chunk_plan(prompt_tokens: int, budget: int) -> dict:
    if prompt_tokens <= 0 or budget <= 0:
        raise ValueError("prompt length and chunk budget must be positive")
    extents = [
        {"start": start, "end": min(start + budget, prompt_tokens)}
        for start in range(0, prompt_tokens, budget)
    ]
    return {
        "max_num_batched_tokens": budget,
        "expected_chunk_count": len(extents),
        "expected_profile_steps": len(extents) + 2,
        "expected_chunk_extents": extents,
    }


def recorded_budget(manifest: dict) -> int:
    argv = manifest["actual_argv"]
    if argv.count("--max-num-batched-tokens") != 1:
        raise ValueError("missing or ambiguous actual service chunk budget")
    budget = int(argv[argv.index("--max-num-batched-tokens") + 1])
    if budget <= 0 or manifest.get("max_num_batched_tokens", budget) != budget:
        raise ValueError("manifest and actual service chunk budgets disagree")
    return budget
