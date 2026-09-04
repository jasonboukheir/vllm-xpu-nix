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
Q6_B1_SHORT_LAST_PRODUCER_VARIANT = "q6_b1_short_last_producer"
Q6_B1_SHORT_LAST_PRODUCER_VARIANT_ID = 19
Q6_B1_SHORT_LAST_PRODUCER_MAX_SEQUENCE_LENGTH = 8192
Q6_B1_SHORT_LAST_PRODUCER_FALLBACK_VARIANT = "q6_prefetch_record_cursor"
Q6_B1_SHORT_LAST_PRODUCER_FALLBACK_VARIANT_ID = 18


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


def kernel_variant_dispatch_contract(kernel_variant: str) -> dict[str, Any]:
    """Describe runtime specialization and fail-closed fallback semantics.

    Most decoder IDs own every compatible call made by an engine that selects
    them. ID19 is deliberately narrower: it fuses the reduction into the last
    producer only for short, multi-split B1 calls and otherwise executes ID18.
    This contract lets service evidence distinguish the engine selector from
    the implementation that is expected to process a particular workload.
    """
    if kernel_variant == Q6_B1_SHORT_LAST_PRODUCER_VARIANT:
        return {
            "schema_version": 1,
            "selected_variant": {
                "name": Q6_B1_SHORT_LAST_PRODUCER_VARIANT,
                "id": Q6_B1_SHORT_LAST_PRODUCER_VARIANT_ID,
            },
            "activation_scope": {
                "kind": "b1_short_multisplit",
                "decode_batch_size": 1,
                "current_sequence_length_maximum_inclusive": (
                    Q6_B1_SHORT_LAST_PRODUCER_MAX_SEQUENCE_LENGTH
                ),
                "num_kv_splits_minimum": 2,
                "requires_unrotate_output": True,
                "requires_initialized_completion_state": True,
            },
            "fallback_variant": {
                "name": Q6_B1_SHORT_LAST_PRODUCER_FALLBACK_VARIANT,
                "id": Q6_B1_SHORT_LAST_PRODUCER_FALLBACK_VARIANT_ID,
            },
        }
    return {
        "schema_version": 1,
        "selected_variant": {"name": kernel_variant},
        "activation_scope": {"kind": "all_compatible_calls"},
        "fallback_variant": None,
    }


def effective_kernel_variant(
    kernel_variant: str,
    *,
    batch: int,
    context_tokens: int,
    num_kv_splits: int,
) -> str:
    """Resolve the expected implementation for one native decode workload."""
    if batch < 1 or context_tokens < 1 or num_kv_splits < 1:
        raise ValueError("batch, context tokens, and split count must be positive")
    if kernel_variant != Q6_B1_SHORT_LAST_PRODUCER_VARIANT:
        return kernel_variant
    if (
        batch == 1
        and context_tokens <= Q6_B1_SHORT_LAST_PRODUCER_MAX_SEQUENCE_LENGTH
        and num_kv_splits > 1
    ):
        return kernel_variant
    return Q6_B1_SHORT_LAST_PRODUCER_FALLBACK_VARIANT
