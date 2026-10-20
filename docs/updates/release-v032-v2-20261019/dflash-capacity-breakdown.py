"""Project Qwen GDN/KVarN/DFlash pool sizes from the recorded qualification inputs."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "target-config",
        "draft-config",
        "manifest",
        "startup-record",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        name: getattr(args, name)
        for name in ("target_config", "draft_config", "manifest", "startup_record")
    }
    inputs = {name: json.loads(path.read_text()) for name, path in paths.items()}
    target = inputs["target_config"]["text_config"]
    draft = inputs["draft_config"]
    argv = inputs["manifest"]["actual_argv"]

    def option(name):
        return argv[argv.index(name) + 1]

    assert option("--kv-cache-dtype") == "kvarn_k4v2_g128_compact"
    assert option("--dtype") == "bfloat16" and target["mamba_ssm_dtype"] == "float32"
    assert json.loads(option("--speculative-config"))["kv_cache_dtype"] == "bfloat16"
    context = int(option("--max-model-len"))
    sequences = int(option("--max-num-seqs"))
    batched_tokens = int(option("--max-num-batched-tokens"))
    block_size = 128
    # V2 async eager TP1/PP1 admits two batches in flight in this captured profile.
    concurrent_batches = 2
    target_layers = target["layer_types"].count("full_attention")
    recurrent_layers = target["layer_types"].count("linear_attention")
    dim = target["head_dim"]
    # Packed K4/V2 payload plus FP16 scales, zero points and absorbed rotations.
    record_bytes = block_size * dim * 6 // 8 + 6 * (block_size + dim)
    target_page = target_layers * target["num_key_value_heads"] * record_bytes
    target_pages = (context + block_size - 1) // block_size
    draft_page = (
        draft["num_hidden_layers"]
        * 2
        * 2
        * block_size
        * draft["num_key_value_heads"]
        * draft["head_dim"]
    )
    conv_dim = (
        2 * target["linear_num_key_heads"] * target["linear_key_head_dim"]
        + target["linear_num_value_heads"] * target["linear_value_head_dim"]
    )
    ssm_bytes = (
        4
        * target["linear_num_value_heads"]
        * target["linear_value_head_dim"]
        * target["linear_key_head_dim"]
    )
    budget_gib = inputs["startup_record"]["available_gib"]
    rows = []
    for proposals in range(1, 8):
        recurrent_page = recurrent_layers * (
            2 * conv_dim * (target["linear_conv_kernel_dim"] - 1 + proposals)
            + ssm_bytes
        )
        draft_tokens = (
            draft["sliding_window"]
            - 1
            + concurrent_batches * batched_tokens
            + proposals
            + 1
        )
        draft_pages = (draft_tokens + block_size - 1) // block_size + 1
        pools = {
            "target_attention": {
                "page_bytes": target_page,
                "usable_pages": target_pages,
            },
            "recurrent": {
                "page_bytes": recurrent_page,
                "usable_pages": sequences * (2 + proposals),
            },
            "draft_sliding": {
                "page_bytes": draft_page,
                "usable_pages": sequences * draft_pages,
            },
        }
        for pool in pools.values():
            pool["usable_bytes"] = pool["page_bytes"] * pool["usable_pages"]
            pool["null_bytes"] = pool["page_bytes"]
        usable = sum(pool["usable_bytes"] for pool in pools.values())
        null = sum(pool["null_bytes"] for pool in pools.values())
        rows.append(
            {
                "proposals": proposals,
                "pools": pools,
                "usable_bytes": usable,
                "null_bytes": null,
                "gross_bytes": usable + null,
                "usable_gib": usable / 2**30,
                "fits_recorded_usable_budget": usable <= budget_gib * 2**30,
            }
        )
    assert round(rows[-1]["usable_gib"], 2) == inputs["startup_record"]["required_gib"]
    result = {
        "scope": "Capacity projection; startup, acceptance and performance not established",
        "geometry_source_commit": "ec1477a0386deba74512cc4068288040a57d6fe3",
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        },
        "context": context,
        "max_num_seqs": sequences,
        "block_size": block_size,
        "concurrent_batches": concurrent_batches,
        "max_num_batched_tokens": batched_tokens,
        "mamba_cache_mode": "align",
        "recurrent_extra_checkpoints": 0,
        "available_usable_gib": budget_gib,
        "available_budget_is_rounded": True,
        "budget_limitation": "K7 profile reused for projection; K1 may profile differently. Null pages are additional to usable bytes. Four recurrent/draft slots do not imply four maximum-length attention contexts.",
        "largest_projected_proposal_count": max(
            row["proposals"] for row in rows if row["fits_recorded_usable_budget"]
        ),
        "counts": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
