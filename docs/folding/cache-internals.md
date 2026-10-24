# Cache internals for context folding (Qwen3.8-Flash-Next)

Research notes for section 7 of "Forking mlx-serve for Context Folding". Read against `main` at
`191da71d`; line numbers refer to that tree. Sizes come from the local pack
`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-3.3bpw` (`config.json`).

## TL;DR

- **Option 4.2b already mostly exists.** The hot prefix cache saves GDN state (recurrent + conv), the
  PLE n-gram window and the QSA indexer history as `SSMCheckpoint`s, matches the longest token
  prefix that has a checkpoint, restores it, cuts attention KV back and prefills only the new tail.
  A harness that resends history with the investigation cut out and the summary appended is
  already served from cache today. The fork mostly needs checkpoint *retention* (pinning branch
  points), not a new mechanism.
- **There is no live per-session state between requests.** Every request gets a fresh `Slot`; at
  the end the slot's state is committed to the prefix cache and the slot is freed. So the plan's
  "slot / timing / copy-on-write" worries (4.5) are already handled by the existing request-level
  restore, and explicit `/branch` `/return` endpoints (4.2) would be the harder path.
- **A GDN checkpoint is not small:** about **115 MB** each (36 GDN layers x 48 x 128 x 128 x f32
  = 113 MB, plus about 2 MB conv window). That is fine on SSD, but it limits how many fit in RAM.
- **QSA indexer:** the pooled key bank is a prefix buffer, so it is cut back like attention.
  Only a 32-row raw ring is saved per checkpoint. Already handled.
- **MTP:** the draft head keeps its own committed-history KV (plus a QSA half). It is cut back,
  not snapshotted, and rides the cache entry. A miss costs draft acceptance, never wrong tokens.
- **PR #895 bug is fixed** and is in the fork's `main` (merged as `3ec2db45`).
- **Expert streaming from SSD exists only in the `lib/mlx-stream` plugin** (DeepSeek-V4.1, MiMo,
  GLM-5.3), not for `qwen4_exp`. No per-expert counters exist for Flash Next.
- **Two gotchas for the harness ideas:** (1) `enable_thinking` / `reasoning_effort` and the tool
  list are rendered at the very top of the system prompt, so toggling them per turn misses the
  whole cache. (2) With MTP on, two Flash Next MTP slots cannot decode at the same time.

## 1. Where per-request caches live

| What | Type / file | Owner |
|---|---|---|
| Attention KV (12 full-attention layers) | `KVCache` `src/transformer.zig:8725` | `Slot.cache` `src/scheduler.zig:466` |
| GDN recurrent + conv state, PLE window, QSA history | `SSMCacheEntry` `src/transformer.zig:9750`, one per layer | `Slot.ssm_entries` `src/scheduler.zig:468` |
| Absolute position | `Slot.moe_seq_offset` (`KVCache.step` stays 0 on a GDN trunk, AGENTS.md rule) | `Slot` |
| MTP head history | `Qwen4MtpState` `src/transformer.zig:25078` (own `KVCache` + `SSMCacheEntry`) | the request's `Generator` |
| Stacked batch state | `PersistentSsmGroup` `src/transformer.zig:10643` | the batched decode tick |

A slot lives for one request only. After the last token, `commitSlotIfApplicable`
(`src/scheduler.zig:6858`) builds `prompt ++ generated` tokens, appends a decode-end checkpoint
(`appendDecodeEndCheckpoint`, `:6816`), hands the newest checkpoint the live QSA history
(`handoffQsaHistoryToLatest`, `src/transformer.zig:14673`) and the MTP committed history, then calls
`HotPrefixCache.commitWithMediaState` (`src/prefix_cache.zig:1691`). Pad-only and errored generations
are not committed. Prompts with images skip checkpointing (`shouldCheckpointSsmPrefill`,
`src/generate.zig:1423`).

## 2. How the prefix cache is organised

`HotPrefixCache` (`src/prefix_cache.zig:469`) is a per-model LRU of `Entry` (`:160`):

- `tokens` (prompt ++ generated), `has_tools`, media spans, `quant_config`, `cache_key`.
- `snapshot`: a `KVCacheSnapshot` (refcount-shared handles, no copy).
- `ssm_checkpoints`: sorted `SSMCheckpoint` list (`src/transformer.zig:14036`), each the full
  per-layer state after `pos` tokens.
- `dflash` / `mtp`: draft-side snapshots (best effort).

**Lookup** (`findBestRestorableMatch`, `src/prefix_cache.zig:1079`) is a linear token compare over
all entries (default cap 32, `server.zig:1003`). For a hybrid model it ranks by the *highest
checkpoint at or below the raw match*, not by the raw match. Then `lookupAndRestoreWithMedia`
(`:1212`) restores KV to `cp.pos`, restores GDN/PLE/QSA from the checkpoint and the caller
prefills `prompt[cp.pos..]`. A restore without QSA history counts as a miss on this arch
(`qsa_history_required`). A hash chain (4.2b) is not needed at this scale: 32 entries x one memcmp is
microseconds.

**Commit**: if an existing entry is a prefix of the new tokens it is *replaced* (extended in place,
checkpoints merged then thinned). Otherwise a *new* entry is created that **inherits the donor's
checkpoints up to the shared prefix** (`:1968`, refcount-shared via `shareSsmCheckpoint`). A folded
request is the second case: the investigation entry stays (until LRU), and the folded entry starts
with every checkpoint below the branch point at no copy cost.

**Copy-on-write** already holds: snapshots and checkpoints are shared MLX handles. MLX ops are
functional, and the extra reference stops in-place donation, so a restoring slot never writes into
the shared copy. "Restore by move" (`checkoutEligible`, `:1553`) hands the buffers over only on a
full-prefix hit, and hands them back if the request ends before writing.

**Eviction** is fair across `cache_key` (`requestCacheKey`, `src/server.zig:9615`): `prompt_cache_key`,
then `metadata.user_id`, then a hash of the system prompt. **The harness should send
`prompt_cache_key = <worker session id>`.** That is the session ID of section 4.1, already there.

**SSD tier** (`src/kv_disk_cache.zig`, `--prefix-cache-disk`, off by default): one folder per entry with
`tokens.bin`, KV chunks `c######.safetensors` (1024 tokens), GDN checkpoints `s<pos>.safetensors`
(immutable, keyed by position) and an MTP/DFlash `spec.safetensors`. Writes go through a background
thread (`kv_disk_writer.zig`). On `qwen4_exp`, turning the disk tier on also turns on **SSD-first**:
idle entries leave RAM past `--prefix-cache-mem` and come back from disk on the next match. That is
most of idea 12.2 (parking): a worker waiting on a tool has no live slot, only a cache entry, and
SSD-first already spills it.

## 3. Where checkpoints are taken, and how many survive

Captured (`captureSsmCheckpoint`, `src/transformer.zig:14067`, a real copy, evaluated in one batch):

1. During prefill, at every multiple of the stride. `--ssm-checkpoint-stride` defaults to 256
   (`server.zig:1096`) but is raised to at least the prefill chunk (`effectiveSsmCheckpointStride`,
   `src/generate.zig:1397`), so in practice it is the chunk width (thousands of tokens on this MoE).
   Flash Next also gets dense low anchors every 8192 tokens up to 64k (`generate.zig:1351`).
2. `SSM_SNAPSHOT_BACKOFF` = 30 tokens before the prompt end (`src/generate.zig:1329`), because the
   generation-prompt suffix renders differently once the turn becomes history.
3. At decode end (`appendDecodeEndCheckpoint`), so the next turn resumes after the generated tail.

Kept: `--ssm-checkpoint-max` 16 per RAM entry (`server.zig:1101`) and 16 per disk entry
(`SSM_DISK_MAX_PER_ENTRY`). Thinning on Flash Next is `min_span_recency` (`scheduler.zig:5380`): keep
the first and newest, drop the interior point that widens gaps least, keep a denser newest quarter.

**What this means for folding.** In an agent loop every turn already leaves a checkpoint at about
prompt-end minus 30 and one at decode end. If the harness keeps the worker's `start_investigation`
call and puts the summary in as its tool result, the folded prompt shares exactly up to a decode-end
checkpoint. Then only the summary is prefilled. If it also drops the call, it falls back about 30
tokens. The real gap is **retention**: in a long session an entry extended every turn thins its
checkpoints to 16, and the branch point can be dropped. A restore then falls back to an older
checkpoint and re-prefills the gap. That is still correct, just slow.

## 4. What a snapshot must hold (answers 4.3)

Per GDN layer (`SSMCacheEntry`): `ssm_state` `[B, 48, 128, 128]` (f32 on this pack, `mamba_ssm_dtype`;
`MLX_SERVE_GDN_STATE_F32=0` halves it) and `conv_state` `[B, 3, 10240]`. Also, which the plan did not
list:

- **PLE n-gram state** (`aux_state` on the PLE layer + `ple_prev` tokens): a dilated-conv window and
  the last `ngram_size-1` tokens. Per position, so it cannot be cut back; it is saved in every
  checkpoint already.
- **QSA indexer** (`aux_state` on attention layers + `qsa_pooled`): the raw key history is a
  32-row ring (`QSA_RING_ROWS`, `transformer.zig:9819`). The pooled block keys `[B, nb, 128]` are a
  prefix buffer. A checkpoint stores the partial-block leftover at `pos` (`qsaLeftoverAt`, `:9951`).
  The pooled bank is attached once to the newest checkpoint and sliced down for older ones
  (`sliceQsaHistoryOntoCheckpoint`, `:14848`), never copied per stride. So QSA is cut back like
  attention and needs no extra snapshot. Roughly 0.75 KB per token for the pooled bank
  (12 layers x 128 x 2 B / ratio 4, inferred).
- **Speculative scratch** (`spec_state_seq`, `spec_conv_input`, `spec_state_in`, `spec_ple_*`): verify
  captures, freed every round. Every speculative round rolls the GDN state back before it returns, so
  nothing speculative crosses a request boundary.

Hyper-connection streams carry no cross-token state (inferred from `forwardQwen4With`; nothing is
cached for them).

**MTP draft state:** the in-checkpoint head keeps a committed-history `KVCache` and its own QSA entry
(`Qwen4MtpState`). On commit, the speculative draft tail is truncated off first. The history is stored
as `Entry.mtp` and restored when `base + step` matches (`specAdoptPlan`, `prefix_cache.zig:263`). It is
cut back, never snapshotted. A missed or declined restore starts the head blind: slower drafts, same
tokens. Nothing needs resetting on return beyond what the cache restore already does.

## 5. Batching and state slots (answers 4.5)

- `qwen4_exp` batch-decodes (`configBatchesDecode`, `scheduler.zig:2570`). Each slot keeps its own
  state. The tick stacks it into a `PersistentSsmGroup`, guarded by `member_gen`: any write outside the
  group bumps the generation, and the group refuses the slot until rebuilt. `ssmRestore` patches the
  group row (`ssmPatchGroupRow`, `transformer.zig:11183`).
- Restores happen only at admission, before the slot's prefill, never mid-tick. A slot joining
  mid-generation drains lazy pipeline state first (AGENTS.md).
- **MTP limits parallel workers.** The Flash Next head's module state is one per model, so a slot that
  drives MTP is exclusive (`slotExclusiveDecode`, `scheduler.zig:2751`). Plain slots batch beside it,
  but **two MTP slots serialize** until one releases the head. Grouped MTP verify is opt-in
  (`MLX_SERVE_MTP_BATCHED_QWEN4`). With several workers, measure MTP-on against batched MTP-off.
- Batched attention: past 1024 KV tokens each slot attends to its own view (`perSlotBatchedAttn`).
  Below that, and on QSA reads, padding waste caps the stacked group (`MAX_PAD_WASTE` 1.5).

## 6. Gaps the fork has to close for 4.2b

1. **Pin branch-point checkpoints.** Let a marker (a request field, or a message the harness flags)
   make the commit tag the checkpoint at that position as pinned, so `min_span_recency` thinning and
   disk thinning skip it. Release the pin on return or on session end. Optionally force a capture at
   an exact token position (`nextChunkEnd` already cuts chunks at stride and anchor points,
   `generate.zig:1266`).
2. **Budget per session.** About 115 MB per pinned checkpoint. RAM budget is `--prefix-cache-mem`, and
   the hybrid cache keeps about 3.4x what it reports (AGENTS.md rule), so turn on the disk tier and let
   SSD-first hold the rest.
3. **Exactness test (6.1) must use the right bar.** On this arch a warm restore is not byte-identical
   to a cold prefill when chunk boundaries differ. The repo's own restore-parity bar is "the chunking
   class, about 0.3 nats top-5" (AGENTS.md). Byte-identical holds when the restored tail fits the
   backoff window (`tests/test_hybrid_reuse_equivalence.sh`). Compare against a cold run with matching
   chunking, or use the logit bar.
4. **Archive and observability.** Log which checkpoint a folded request restored from and how many
   tokens it re-forwarded (the `[hot-cache]` lines already carry most of this).
5. The header comment of `src/prefix_cache.zig` still says hybrids are excluded. That is stale.

## 7. Harness constraints found on the way

- **Thinking and tools sit at the top of the prompt.** The Flash Next template puts the
  `reasoning_effort` sentence and the whole `<tools>` block at the start of the system message. So
  per-turn thinking control (section 8) or loading tool definitions (section 9) changes token 1 and
  misses the whole cache. Keep `reasoning_effort`, `enable_thinking` and the tool list fixed per
  session, and steer thinking another way, for example with a short instruction at the end of the
  turn. Patching the template is another option.
- `preserve_thinking` defaults to true in this template, so past reasoning stays in history and
  re-renders identically **if the harness echoes `reasoning_content` back**. If it drops it, every
  past assistant turn changes and the match falls back to the start of that turn.
- Send `prompt_cache_key` per worker (see above).

## 8. SSD expert streaming (for 12.2b)

- The in-tree MoE for `qwen4_exp` keeps every expert resident. ModelOpt NVFP4 packs read routed experts
  off disk into banks at load only (`modelopt_pack.zig`). Sushi EXL3 is a weight encoding, not
  streaming.
- The streamer lives in the `lib/mlx-stream` plugin (`ddalcu/mlx-stream`; the submodule is not checked
  out on macstudio1): `expert_stream.zig` (per-layer slot pools, routes, deferred release, lookahead),
  `expert_admission.zig`, `expert_bank.zig` (`experts.bin`), with aggregate hit/miss/eviction stats.
  It serves only DeepSeek-V4.1, MiMo-V2.6-Flash and GLM-5.3. It owns the whole forward, runs one request
  at a time and turns the prefix cache off, so it cannot be reused for Flash Next as is. Its
  slot-pool and lookahead design is the reference to borrow.
- No per-expert counters exist. Flash Next's routing has several arms that would all need the counter:
  the compiled routing closure (`transformer.zig:18914`, chain at `:37560`), `moeRouterLogits`
  (`:33292`), the fused router (`moeRouterFusedEnabled`, `:38493`), batched decode
  (`forwardMoeBatchedDecode`, `:22636`), verify rows (`moeVerifyRows`, `:33449`) and tensor-parallel
  (`moeTensorParallel`, `:33543`).

## 9. PR #895 (batched MTP, tensor mode, two streams)

Fixed. The reviewer's finding (grouped verify reached `moeMLP` on rank 0's half-experts) was fixed
during review: `qwen4VerifyRowsEligible` declines while `dist_tp` is set (`transformer.zig:26054`), with
a test. The PR merged on 2026-10-18 as `3ec2db45` and is in the fork's `main`. It only ever affected
two-Mac tensor mode with the opt-in `MLX_SERVE_MTP_BATCHED_QWEN4=1`.

One thing matters more for the second Mac: in **pipeline** mode rank 0 keeps **no prefix cache and no
MTP head** (`docs/reference.md`, Thunderbolt cluster), which would turn folding off. **Tensor** mode
keeps both. A two-Mac setup for folding has to use tensor mode.
