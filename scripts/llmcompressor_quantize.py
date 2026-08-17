#!/usr/bin/env python3
"""Run a combined AutoRound weight + calibrated FP8 KV-cache recipe."""

import argparse
from pathlib import Path

from auto_round.calib_dataset import get_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--scheme", default="W4A16")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset", default="NeelNanda/pile-10k")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype="auto",
        trust_remote_code=False,
    )
    dataset = get_dataset(
        tokenizer=tokenizer,
        seqlen=args.seqlen,
        nsamples=args.samples,
        dataset_name=args.dataset,
        seed=args.seed,
    )
    recipe = f"""
quant_stage:
  quant_modifiers:
    QuantizationModifier:
      kv_cache_scheme:
        num_bits: 8
        type: float
        strategy: tensor
        dynamic: false
        symmetric: true
    AutoRoundModifier:
      targets: [Linear]
      scheme: {args.scheme}
      ignore: [lm_head, 're:.*mlp.gate$']
      iters: {args.iters}
      enable_torch_compile: false
      batch_size: {args.batch_size}
"""
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args.seqlen,
        num_calibration_samples=args.samples,
        shuffle_calibration_samples=False,
        batch_size=args.batch_size,
        data_collator="truncation",
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, save_compressed=True)
    tokenizer.save_pretrained(output)


if __name__ == "__main__":
    main()
