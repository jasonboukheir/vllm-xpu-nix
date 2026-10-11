# KVarN prefix caching on hybrid Qwen models

Enable local prefix caching with `--enable-prefix-caching` for the qualified
Qwen3.5 architecture, BF16 compute, compact K4V2, TP1/PP1, eager V1 profile.
The Brutus checkpoint is named `RedHatAI/Qwen3.8-27B-INT4`; its architecture is
Qwen3.5. Bundled MTP with one or two draft tokens is accepted; the measured
Brutus recipe uses two. See [the serving profile](kvarn-beta.md#serving-profile).

## What occupies memory

Attention KV pages hold both active requests and reusable prompt prefixes.
Prefix reuse shares those same KVarN-compressed pages, without making an
uncompressed prompt copy. Unreferenced cached pages can be evicted when active
requests need memory. KVarN's bounded recent-token and sink buffers retain
higher precision separately.

This hybrid model also needs recurrent-state checkpoints. These are not
attention K/V and are not compressed by KVarN. A useful cache hit must have both
the attention pages and the recurrent state at a matching boundary. Plenty of
free attention capacity therefore does not guarantee that an old conversation
still has a reusable checkpoint.

For the Brutus model and MTP2 recipe:

- One recurrent checkpoint across 48 layers costs 155,910,144 bytes
  (148.6875 MiB), independent of the prefix length.
- One 128-token attention block across 17 layers costs 1,827,840 bytes,
  equivalent to 14,280 bytes per token, including the compact layout metadata.
- Exchanging one checkpoint for attention capacity costs about 10,918 tokens,
  rounded to whole 128-token blocks.

## Capacity breakpoints

With four configured sequences, MTP2 and align-mode caching, 16 recurrent slots
cover the live-state reservation. The allocator reserves up to four more
checkpoint slots, spending only memory above the attention capacity required
by `max_model_len`. Each pool also has a null block.

The following capacities use a fixed 7,032,152,064-byte allocation, matching
the observed final pool allocation on Brutus. Intermediate rows are calculated
from that budget; a fresh memory profile can change a row by a page or more.
There is no separate user-facing extra-checkpoint-count option.

| Extra checkpoint slots beyond the 16 live slots | Shared attention tokens |
| ---: | ---: |
| 0 | 306,688 |
| 1 | 295,680 |
| 2 | 284,800 |
| 3 | 273,920 |
| 4 | 263,040 |
| 5, hypothetical | 252,032; cannot preserve a 262,144-token request |

With prefix caching disabled, the live-state reservation drops to 12 slots;
the observed attention capacity was 350,336 tokens. Enabling caching therefore
has a live-state cost before adding the four extra retained checkpoints.

The four-extra-slot configuration preserves 262,144 combined input/output
tokens per request, with 896 shared attention tokens to spare in the measured
allocation. It cannot hold four unrelated maximum-length requests at once.
Requests sharing the same prefix can share physical attention pages.

Changing `max_num_seqs` changes both the mandatory live reservation and the
automatic extra-checkpoint limit. At the same fixed byte budget, the calculated
capacities are:

| Maximum active sequences | Live recurrent slots | Extra checkpoint slots | Shared attention tokens |
| ---: | ---: | ---: | ---: |
| 1 | 4 | 1 | 426,752 |
| 2 | 8 | 2 | 372,096 |
| 3 | 12 | 3 | 317,568 |
| 4 | 16 | 4 | 263,040 |
| 5 | 20 | 0 | 263,040 |
| 6 | 24 | 0 | 219,264; insufficient for the configured context |

Only the four-sequence row was qualified here. Changing concurrency can also
change the profiled memory budget, so these are allocation comparisons rather
than promised startup capacities. Five sequences would spend the four spare
checkpoints on live state; six would fail the full-context capacity check at
this budget.

## Choosing the balance

Longer distinct prompts consume more attention pages. More active requests
increase the mandatory recurrent reservation through `max_num_seqs`; more
distinct recent conversations and branch points compete for retained states.
Changing speculative depth, recurrent-state precision, model dimensions,
checkpoint retention policy or available GPU memory also changes the balance.
Checkpoint slots are not a fixed number of conversations: one conversation
can retain multiple boundaries, and branches can share one boundary.

There is no workload-independent optimum. Long, frequently reused histories
save more prefill work for the same fixed-size recurrent checkpoint. Many
unrelated long requests favor more attention capacity. On the tested four-way
agent-like workload, the live-only reservation repeatedly evicted a needed
checkpoint, while four extra slots preserved every hit across three repeated
batches. The allocator's bounded reserve captures that benefit while keeping
the configured maximum context.

Prefix identity matters as much as capacity. Keep leading token IDs stable;
new timestamps or reminders at the front can invalidate reuse. The current
Qwen `/v1/messages` template moves non-leading system messages to the front,
which can change an otherwise shared conversation prefix.

Synthetic source-overlay measurements and their scope are retained in
[the investigation data](updates/prefix-cache-20261010/measurements.json).
Installed release qualification is recorded in the corresponding JSON release
manifest and release commit message.
