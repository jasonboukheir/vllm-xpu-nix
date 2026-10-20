# KVarN V2 and DFlash2 integration review

The v05 feature absorbs v06–v08 while preserving upstream v0.32 checkpoint
proofs, per-allocation zeroing flags, aligned storage and HiSparse behavior.
The prior 64-token resend checkpoint workaround is superseded by upstream's
checkpoint proof; the independent-pool resend regression remains adapted.

The new V2 path separates target builders from BF16 sliding draft builders,
allocates all groups, and recomputes kernel block sizes across both. Recurrent
and sliding reservations cover every admitted request. The in-flight bound
includes DFlash's eight lookahead writes. A real allocator regression reaches
34 resident pages at an unaligned chunk boundary, against the former bound of
33. Target context capacity remains independent of this bounded reservation.

The independent review by `/root/audit_k05` identified deferred stores running
after zeroing/COW in the first port. `prepare_cache_update` now fences outgoing
stores before any request/cache mutation. `pre_forward` performs restores after
initialization but before KVarN metadata. A content-based connector regression
checks that the old page is saved once and the restored page survives.
Initialization resets profiling cache allocations; eager measurements must
prime the actual candidate before sampling steady-state performance.

The review by `/root/audit_k07` verified the native twelve-row ABI, int32
sequence-length requirement and output-scratch aliasing. Causal virtual rows
retain physical page IDs and per-query prefixes. Sequential chunks copy each
result before reusing scratch. Mixed decode preserves its query width.
The native regression uses distinct request tables, eight-token queries,
12+4 launch splitting, page boundaries and future/rejected-tail poisoning.
It passed 72 cases against an independent attention reference. The bounded
CPU scratch regression deliberately overwrites the same scratch each call.

DFlash queries beyond max_model_len now have invalid slots even when their
position IDs are clamped. The drafter applies its own weight-name mapper and
packed projection mapping before global layer exclusions; loaded native
quantization still requires validation in the complete serving package.

The unchanged native GDN stack passed 24 rollback/continuation cases with
eight state slots, using the reproducible sibling validation patch. These
checks use the retained kernel closure and rebuilt required Python dependencies.
They do not qualify a newly built v0.32 package or establish serving performance.
