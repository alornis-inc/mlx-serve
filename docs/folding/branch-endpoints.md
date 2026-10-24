# Branch endpoints for context folding

Sections 4.1, 4.2, 10 and 11 of "Forking mlx-serve for Context Folding", built on the
snapshot store from `feat/gdn-snapshots`. Code: `src/branches.zig` (bookkeeping, no mlx) and
`handleBranchRoute` in `src/server.zig`.

## Model

- **Session** = the `prompt_cache_key` the worker's chat requests already send. The server
  hashes it the same way it hashes the cache-eviction workload key (`requestCacheKey`), so
  `/v1/chat/completions`, `/v1/messages` and `/v1/responses` are unchanged.
- **Branch** = a *pinned* checkpoint at a token position of that session's history. The pin
  stops LRU eviction, checkpoint thinning and SSD drops from removing it. Nothing is copied:
  the prefix cache already holds the GDN, conv, PLE and QSA state at each checkpoint, and it
  shares the attention KV by refcount.
- **Return** doesn't restore anything eagerly. The harness resends the folded history (the
  prefix plus the summary). The prefix cache then restores the pinned checkpoint and prefills
  only the summary, which is the same work the plan's `/return` would do, done on the next
  request. Until that request arrives, the branch's pin stays held as the session's *anchor*.

There is no live per-session slot between requests: every request gets a fresh slot that is
committed to the prefix cache when it ends (`docs/folding/cache-internals.md`). That's why the
endpoints only manage pins and leave restoring to the normal request path.

## Endpoints

All bodies are JSON. `GET` and `DELETE` also read `session_id` and `branch_id` from the query string.

| Route | Body | Result |
|---|---|---|
| `POST /branch` | `session_id`, optional `label` | `{"id":"br_N","object":"branch","position":P,"depth":0,"created":T,"label":…}`. P is the end of the session's last turn (prompt + generated), or the anchor after a `/return`. |
| `POST /return` | `session_id`, `branch_id` | `{"id":"br_N","object":"branch.return","position":P,"closed":K}`. Closes this branch and every branch opened after it, and releases K pins. The harness then sends `history[0..P] + summary`. |
| `GET /branches?session_id=` | | `{"object":"list","session_id":…,"data":[…]}`, outermost first, with `depth` set. |
| `DELETE /branch` | `session_id`, `branch_id` | `{"id":"br_N","object":"branch","deleted":true}`. Unpins without folding. |

Errors: 400 when `session_id` or `branch_id` is missing; 404 for an unknown branch, including
one that belongs to another session (ownership is never revealed); 409 when the session has no
turn yet or there is no checkpoint at that position; 429 past 32 open branches per session or
256 sessions; 501 until the snapshot store serves pins.

## Harness flow

1. Every worker request carries `prompt_cache_key: <worker id>`.
2. `start_investigation` → `POST /branch`. Keep the returned `position` together with the
   matching message index in the harness's history.
3. The worker explores with normal chat turns.
4. `finish_investigation(summary)` → `POST /return`. Archive the dropped messages to a file,
   replace them with one summary message (keep the `start_investigation` call and put the
   summary in as its tool result, so the prompt shares the prefix up to the decode-end
   checkpoint), and continue the conversation.

**Checkpoint jump-back (10)** comes for free: `/return` accepts any open branch, not only the
innermost one. **Fork/join (11)** needs no endpoint: N concurrent chat requests that share
the pinned prefix each restore it into their own slot (copy-on-write), and the join is one
`/return` followed by a request with every branch's summary appended in a fixed order. Give each
fork its own `prompt_cache_key` (for example `<worker>/fork-k`), so eviction stays fair.

## Still to do

- `branchStoreFor` returns null until `HotPrefixCache.sessionHead / pin / unpin` land
  (owned by the snapshot thread). Those calls then hop onto the inference thread through a
  scheduler queue, like `unloadModel` does.
- The registry is per server, not per model. A session that switches models would pin
  against the wrong cache.
- A session that returns and never branches again holds its anchor until the server restarts.
  An idle timeout should release it.
