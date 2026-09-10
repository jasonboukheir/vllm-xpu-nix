# Pure-prefill serving integration plan

The user explicitly advanced the small, repeatable native pure-prefill result
on 2026-09-09 to a production fix and companion PR. This is a new validation
stage, after the separately archived operator experiment. No minimum service
speedup is required for a correct overhead cleanup; report improvements only
when repeatable, and investigate any correctness or systematic performance
regression. Do not inherit the unrelated INT4 experiment's 3% threshold.

Use `docs/kvarn-profiling.md`, the existing service runner, and fresh evidence
under `benchmark-results/pure-prefill-serving-20260909/`.

- Implement in vLLM based on current fork main `6700b7cad8`. Build control from
  that same revision; compare it with only the candidate Python source delta.
  Preserve the full native kernel package, model checkpoint, dependencies,
  transport and service settings. Keep package version labels identical and
  attest real source hashes/commits separately.
- Enable the existing flag only for an unpadded, CPU-proved single prefill with
  query length >16. Require existing CPU prefill-state and exact CPU prefill KV
  length metadata; missing or inconsistent metadata retains existing dispatch.
  Clear the cached hint for graph capture, draft builds and draft metadata reuse.
  Forward the flag explicitly through the XPU wrapper; preserve non-XPU APIs.
- Restrict selection to ordinary causal D256 with FP16/BF16 KV, no local
  window, sinks, ALiBi or softcap. The flag changes classification and a
  skipped decode launch, not prefill arithmetic or templates. Other existing
  native prefill features keep their default routing.
- Run focused existing fork tests for CPU metadata/padding, native forwarding
  and graph/draft invalidation. Exercise actual native wrapper outputs with
  pure/default flags, output ownership and optional LSE for the selected
  FP16/BF16 configurations. Unit tests verify excluded feature routing.
- Before profiler-off timing, use a separate service diagnostic to prove that
  actual ordinary prefill calls reach the intended native route and retain
  decode dispatch. Diagnostic time is never performance evidence.
- Run auto-cache 16383/65023-token text requests, B1, eager, budget2048, MTP0,
  prefix cache off, 512 output tokens with EOS ignored: one warmup plus three
  measured repeats. Use two fresh control/candidate service pairs per length,
  reversing arm order. Compare exact requests/token IDs/output IDs, TTFT,
  total time, decode throughput, client token p95/p99, memory and preemptions.
- Run matched MTP2 image/text/state qualification for auto and compact KVarN,
  preserving within-dtype output expectations and state/repeat checks. If
  paired outputs differ, retain the failure and inspect first divergence with
  existing teacher-forced logprob replay before making an integration decision.
- Record all raw results, source/package/DSO identities, checksums and limits.
  Open a vLLM fork PR and update the workflow PR with the actual implementation
  and measured outcome. No kernel source change, release or deployment is planned.
