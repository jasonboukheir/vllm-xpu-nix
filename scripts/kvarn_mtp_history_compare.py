"""Score unconstrained draft rollouts over audited, identical target histories."""

from __future__ import annotations


def audit_history(rows, prompt, output):
    if len(output) < 3 or not rows or not prompt:
        raise ValueError("missing prompt or complete prediction pairs")
    cursor = 0
    next_tokens, valid = [], []
    history = prompt + output[:-1]
    for index, row in enumerate(rows):
        if row["step"] != index or row["suppress"] is not True:
            raise ValueError("missing/duplicate step or proposals not suppressed")
        tokens = row["target_tokens"]
        count = len(tokens)
        if not count or tokens != history[cursor : cursor + count]:
            raise ValueError("target inputs differ from the committed history")
        if row["query_start_loc"] != [0, count] or row["seq_lens"] != [cursor + count]:
            raise ValueError("cache query/sequence lengths are misaligned")
        positions = row["target_positions"]
        if not positions or any(
            axis != list(range(cursor, cursor + count)) for axis in positions
        ):
            raise ValueError("causal target positions are misaligned")
        if row["num_rejected_tokens"] is not None:
            raise ValueError("target unexpectedly verified speculative tokens")
        hidden = row["target_hidden_states"]
        if (
            hidden["shape"] != [count, 5120]
            or hidden["dtype"] != "torch.bfloat16"
            or len(hidden["sha256"]) != 64
        ):
            raise ValueError("invalid target hidden-state identity")
        cursor += count
        expected_discard = cursor < len(prompt)
        if row["discard"] != [expected_discard]:
            raise ValueError("chunk-prefill discard boundary is misaligned")
        if not expected_discard:
            if len(row["next_token_ids"]) != 1 or len(row["draft_token_ids"]) != 2:
                raise ValueError("missing target token or two-token proposal")
            if valid and count != 1:
                raise ValueError("target did not advance one committed token")
            next_tokens.extend(row["next_token_ids"])
            valid.append(row)
    if cursor != len(history) or next_tokens != output:
        raise ValueError("incomplete history or sampled targets differ from output")
    first = second = 0
    matches = []
    for index, row in enumerate(valid[:-2]):
        one = row["draft_token_ids"][0] == output[index + 1]
        two = one and row["draft_token_ids"][1] == output[index + 2]
        first += one
        second += two
        matches.append([bool(one), bool(two)])
    count = len(matches)
    return {
        "unique_prefixes": count,
        "first_agreements": first,
        "both_agreements": second,
        "first_agreement_rate": first / count,
        "conditional_second_agreement_rate": second / first if first else None,
        "both_agreement_rate": second / count,
        "counterfactual_accepted_tokens_per_pair": (first + second) / count,
        "matches": matches,
        "scope": "On-policy two-token draft rollouts at every frozen target prefix; target advances one token. These are not actual serving verification rounds or independent repetitions of the same predictions.",
    }


def require_matched_states(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError("different target invocation counts")
    # Exclude only the draft output. This checks exact hidden-state bytes by
    # hash alongside positions, source tokens, selected next token and lengths.
    for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        left = {k: v for k, v in left.items() if k != "draft_token_ids"}
        right = {k: v for k, v in right.items() if k != "draft_token_ids"}
        if left != right:
            raise ValueError(f"target state differs at invocation {index}")
