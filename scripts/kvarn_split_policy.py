"""Canonical KVarN split-policy selection and provenance contracts.

The service selects a named policy once at engine startup, but a policy may
resolve a different split count for each decode call.  Keep that distinction
explicit: a context-dependent policy must never be serialized as a nominal
batch-only split map.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

NATIVE_SPLIT_POLICIES = ("fixed", "b70_q6", "b70_q6_v2")
NAMED_SPLIT_POLICIES = frozenset({"b70_q6", "b70_q6_v2"})
SUPPORTED_HARNESS_BATCHES = (1, 4)
FACTORY_SPLIT_POLICY_EXPLICIT = "explicit"
FACTORY_SPLIT_POLICY_B70_WAVE_SWEEP = "b70_wave_sweep"
FACTORY_SPLIT_POLICIES = (
    FACTORY_SPLIT_POLICY_EXPLICIT,
    FACTORY_SPLIT_POLICY_B70_WAVE_SWEEP,
)
B70_WAVE_SWEEP_SPLITS = (8, 16, 17, 24, 32)
B70_WAVE_SWEEP_KERNEL_VARIANT = "q6_prefetch_record_cursor"
B70_WAVE_SWEEP_KERNEL_VARIANT_ID = 18
B70_Q6_SPLITS = {1: 32, 4: 8}
B70_Q6_MAX_SPLITS = 32
B70_Q6_V2_KERNEL_VARIANT = "q6_next_page_prefetch"
B70_Q6_V2_KERNEL_VARIANT_ID = 12
B70_Q6_V2_KERNEL_VARIANTS = {
    B70_Q6_V2_KERNEL_VARIANT: B70_Q6_V2_KERNEL_VARIANT_ID,
    "q6_next_page_prefetch_split_reducer": 13,
}
B70_Q6_V2_CONTEXT_THRESHOLD = 48 * 1024
B70_Q6_V2_MAX_SPLITS = 32


def _rule(*, batch: int, maximum: int | None, splits: int, minimum: int = 1) -> dict:
    return {
        "batch": batch,
        "context_tokens_minimum": minimum,
        "context_tokens_maximum_inclusive": maximum,
        "num_kv_splits": splits,
    }


_B70_Q6_CONTRACT = {
    "schema_version": 1,
    "selector": "b70_q6",
    "selection_axes": ["decode_batch_size"],
    "supported_harness_batches": list(SUPPORTED_HARNESS_BATCHES),
    "scratch_max_splits": B70_Q6_MAX_SPLITS,
    "kernel_compatibility": {"kind": "q6_xe2_dpas_family"},
    "rules": [
        _rule(batch=1, maximum=None, splits=32),
        _rule(batch=4, maximum=None, splits=8),
    ],
}

_B70_Q6_V2_CONTRACT = {
    "schema_version": 1,
    "selector": "b70_q6_v2",
    "selection_axes": ["decode_batch_size", "context_tokens"],
    "supported_harness_batches": list(SUPPORTED_HARNESS_BATCHES),
    "scratch_max_splits": B70_Q6_V2_MAX_SPLITS,
    "kernel_compatibility": {
        "kind": "exact_variants",
        "variants": [
            {"name": name, "id": variant_id}
            for name, variant_id in B70_Q6_V2_KERNEL_VARIANTS.items()
        ],
    },
    "rules": [
        _rule(batch=1, maximum=None, splits=32),
        _rule(batch=4, maximum=B70_Q6_V2_CONTEXT_THRESHOLD, splits=8),
        _rule(
            batch=4,
            minimum=B70_Q6_V2_CONTEXT_THRESHOLD + 1,
            maximum=None,
            splits=32,
        ),
    ],
}

# This is intentionally a sweep contract, not a runtime policy or a claim
# that one split count wins.  The older device-stage sweep is the source of
# the complete candidate set; the two later, matched ID18 runs establish that
# both endpoints remain relevant on the current reader.  A warmed B70 run of
# all five cells is required before a context/batch winner may be frozen.
_B70_WAVE_SWEEP_CONTRACT = {
    "schema_version": 1,
    "selector": FACTORY_SPLIT_POLICY_B70_WAVE_SWEEP,
    "selection_mode": "enumerate_all_candidates_no_winner",
    "hardware": "Intel(R) Arc(TM) Pro B70 Graphics",
    "candidate_num_kv_splits": list(B70_WAVE_SWEEP_SPLITS),
    "winner": None,
    "kernel_compatibility": {
        "kind": "exact_variant",
        "name": B70_WAVE_SWEEP_KERNEL_VARIANT,
        "id": B70_WAVE_SWEEP_KERNEL_VARIANT_ID,
    },
    "evidence": [
        {
            "artifact": (
                "benchmark-results/kvarn/"
                "20260904T002554Z-untouched-beta-device-stages/"
                "dpas-16k-65k-split-sweep.json"
            ),
            "sha256": (
                "eb307d22aba29adf68556013bf4bf1d8"
                "cf4e31e69e40967d35d56c04a0c07869"
            ),
            "scope": "candidate_set_only_pre_id18",
            "batches": [1, 4],
            "contexts": [16384, 65023],
            "num_kv_splits": list(B70_WAVE_SWEEP_SPLITS),
        },
        {
            "artifact": "benchmark-results/kvarn/factory-b70-20260904T153938Z.json",
            "sha256": (
                "ec92e73b7b1dd8aceae818dcfd32d5fff"
                "4024aa5059e5b9f52a4ff923df6c9aa"
            ),
            "scope": "matched_id18_endpoint_anchor",
            "batches": [1, 4],
            "contexts": [4096, 16384, 65536],
            "num_kv_splits": [8, 32],
        },
        {
            "artifact": "benchmark-results/kvarn/factory-b70-20260904T155813Z.json",
            "sha256": (
                "034feed1e2d15a149573cd5f2cfec905"
                "be8bc4da24d6b8e6be2048c290a21a52"
            ),
            "scope": "matched_id18_only_endpoint_anchor",
            "batches": [1, 4],
            "contexts": [4096, 16384, 65023],
            "num_kv_splits": [8, 32],
        },
    ],
}


def factory_split_policy_contract(
    selector: str, explicit_splits: list[int | None] | None = None
) -> dict[str, Any]:
    """Return a factory-enumeration contract without changing runtime policy."""
    if selector == FACTORY_SPLIT_POLICY_B70_WAVE_SWEEP:
        return deepcopy(_B70_WAVE_SWEEP_CONTRACT)
    if selector != FACTORY_SPLIT_POLICY_EXPLICIT:
        raise ValueError(f"unsupported factory split policy {selector!r}")
    if not explicit_splits:
        raise ValueError("explicit factory split policy requires split selections")
    return {
        "schema_version": 1,
        "selector": FACTORY_SPLIT_POLICY_EXPLICIT,
        "selection_mode": "caller_explicit",
        "candidate_num_kv_splits": [
            "auto" if value is None else int(value) for value in explicit_splits
        ],
        "winner": None,
        "kernel_compatibility": {"kind": "explicit_dispatch_compatible"},
        "evidence": [],
    }


def owns_runtime_selection(selector: str) -> bool:
    """Whether the named policy, rather than a fixed environment value, selects."""
    return selector in NAMED_SPLIT_POLICIES


def split_policy_contract(
    selector: str, fixed_splits: Mapping[int, int] | None = None
) -> dict[str, Any]:
    """Return the immutable, JSON-safe selection contract for one engine sweep."""
    if selector == "b70_q6":
        return deepcopy(_B70_Q6_CONTRACT)
    if selector == "b70_q6_v2":
        return deepcopy(_B70_Q6_V2_CONTRACT)
    if selector != "fixed":
        raise ValueError(f"unsupported native split policy {selector!r}")
    if (
        not fixed_splits
        or not set(fixed_splits).issubset(SUPPORTED_HARNESS_BATCHES)
    ):
        raise ValueError("fixed split policy requires explicit B1 and/or B4 selections")
    rules = [
        _rule(batch=batch, maximum=None, splits=int(fixed_splits[batch]))
        for batch in sorted(fixed_splits)
    ]
    return {
        "schema_version": 1,
        "selector": "fixed",
        "selection_axes": ["decode_batch_size"],
        "supported_harness_batches": sorted(fixed_splits),
        "scratch_max_splits": max(int(value) for value in fixed_splits.values()),
        "kernel_compatibility": {"kind": "explicit_dispatch_compatible"},
        "rules": rules,
    }


def effective_splits(
    selector: str,
    *,
    batch: int,
    context_tokens: int,
    fixed_splits: Mapping[int, int] | None = None,
) -> int:
    """Resolve one exact harness workload against the canonical policy contract."""
    if context_tokens < 1:
        raise ValueError("context tokens must be positive")
    contract = split_policy_contract(selector, fixed_splits)
    for rule in contract["rules"]:
        if rule["batch"] != batch:
            continue
        if context_tokens < rule["context_tokens_minimum"]:
            continue
        maximum = rule["context_tokens_maximum_inclusive"]
        if maximum is None or context_tokens <= maximum:
            return int(rule["num_kv_splits"])
    raise ValueError(f"{selector} has no split selection for B{batch} at {context_tokens}")


def nominal_splits_by_batch(
    selector: str, fixed_splits: Mapping[int, int] | None = None
) -> dict[str, int] | None:
    """Return a batch-only map only when it is true for every context."""
    contract = split_policy_contract(selector, fixed_splits)
    if "context_tokens" in contract["selection_axes"]:
        return None
    return {
        str(rule["batch"]): int(rule["num_kv_splits"])
        for rule in contract["rules"]
    }


def validate_kernel_compatibility(
    selector: str, kernel_variant: str, *, q6_variants: frozenset[str]
) -> None:
    if selector == "b70_q6" and kernel_variant not in q6_variants:
        raise ValueError("b70_q6 split policy requires a q6 native kernel variant")
    if (
        selector == "b70_q6_v2"
        and kernel_variant not in B70_Q6_V2_KERNEL_VARIANTS
    ):
        raise ValueError(
            "b70_q6_v2 split policy requires q6_next_page_prefetch (ID12) "
            "or q6_next_page_prefetch_split_reducer (ID13)"
        )
