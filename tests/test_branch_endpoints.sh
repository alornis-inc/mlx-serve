#!/bin/bash
# test_branch_endpoints.sh — POST /branch, POST /return, GET /branches, DELETE /branch.
#
# A branch pins the session's history (session = prompt_cache_key); the folded turn that
# follows a /return must restore at or above the branch point even after other sessions'
# turns pushed the hot cache past its entry cap. Also: nesting and jump-back, ownership
# (another session's branch id is a 404), and the 400/404/409 refusals.
#
#  1. Refusals before any turn: no session_id 400, no history 409, empty list.
#  2. Branch after R1, nested branch after R2, list shows depths 0 and 1.
#  3. Another session cannot return to or drop them (404).
#  4. Unrelated sessions churn the cache; /return to the OUTER branch closes both.
#  5. The folded turn reuses at least the branch position (cached_n).
#  6. DELETE drops a branch without folding; the list is empty again.
#
# Env: MLX_FOLD_MODEL (default Qwen3.8-Flash-Next), PORT (19094), BINARY.
# Pinning off (the routes answering 501, or pin() a no-op) turns check 5 red: with
# --prefix-cache-entries 2 the churn evicts the session's entry.

set -uo pipefail

source "$(dirname "$0")/_lib_models.sh"
MODEL="${MLX_FOLD_MODEL:-$(find_model ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-3.3bpw mlx-community/Qwen3.5-4B-MLX-4bit)}"
PORT="${PORT:-19094}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
BASE="http://127.0.0.1:$PORT"

[ -d "$MODEL" ] || { echo "SKIP: model dir not found: $MODEL"; exit 0; }
[ -x "$BIN" ]   || { echo "fail: build mlx-serve first"; exit 1; }
command -v jq >/dev/null || { echo "needs jq"; exit 1; }

WORK="$(mktemp -d)"
LOG="$WORK/server.log"
SERVER_PID=""
trap '[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; wait 2>/dev/null; rm -rf "$WORK"; true' EXIT

mkdir -p "$WORK/home"
HOME="$WORK/home" "$BIN" --model "$MODEL" --serve --port "$PORT" --ctx-size 16384 \
    --prefix-cache-entries 2 --no-mtp --log-level info >> "$LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 600); do
    curl -sf --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"' && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: server died:"; tail -30 "$LOG"; exit 1; }
    sleep 0.5
done

EC=0
ok()   { echo "  ok: $1"; }
fail() { echo "  FAIL: $1"; EC=1; }
# req <method> <path> <json-body> -> "<status>\t<body>"
req() {
    local out code
    out="$(curl -s --max-time 600 -w '\n%{http_code}' -X "$1" "$BASE$2" -H 'Content-Type: application/json' ${3:+-d "$3"})"
    code="${out##*$'\n'}"
    printf "%s\t%s\n" "$code" "${out%$'\n'*}"
}
expect() { # name want-status "<status>\t<body>"
    local got="${3%%$'\t'*}"
    if [ "$got" == "$2" ]; then ok "$1 -> $2"; else fail "$1 -> $got (want $2): ${3#*$'\t'}"; fi
}
body() { printf "%s" "${1#*$'\t'}"; }

SYSTEM="You are a careful coding worker. Answer in at most three sentences."
TASK="Task: explain what a ring buffer is and when it beats a linked list."
JUNK="$(python3 -c "print('Investigation log: ' + ' '.join(f'file_{i}.zig line {i*7}: no match;' for i in range(260)))")"
JUNK2="$(python3 -c "print('Nested search: ' + ' '.join(f'grep hit {i} in src/x{i}.zig;' for i in range(180)))")"
SUM1="Summary of the investigation: the codebase has no ring buffer yet; a fixed array is enough."

msgs() { # role content role content ...
    local out="[]"
    while [ $# -ge 2 ]; do
        out="$(jq -c --arg r "$1" --arg c "$2" '. + [{role:$r,content:$c}]' <<< "$out")"
        shift 2
    done
    echo "$out"
}
chat() { # <session> <messages-json>
    curl -sf --max-time 600 -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg k "$1" --argjson m "$2" \
            '{messages:$m,max_tokens:48,temperature:0,seed:1,stream:false,enable_thinking:false,prompt_cache_key:$k}')"
}
S=worker-1

echo "[1] refusals before any turn"
expect "POST /branch without session_id" 400 "$(req POST /branch '{}')"
expect "POST /branch with no history" 409 "$(req POST /branch "{\"session_id\":\"$S\"}")"
r="$(req GET "/branches?session_id=$S")"; expect "GET /branches" 200 "$r"
[ "$(body "$r" | jq '.data | length')" == 0 ] && ok "no open branches" || fail "list not empty: $(body "$r")"

echo "[2] branch, nested branch, list"
R1="$(chat $S "$(msgs system "$SYSTEM" user "$TASK")")" || { echo "fail: R1"; exit 1; }
A1="$(jq -r '.choices[0].message.content' <<< "$R1")"
r="$(req POST /branch "{\"session_id\":\"$S\",\"label\":\"read logs\"}")"; expect "POST /branch outer" 200 "$r"
B1="$(body "$r" | jq -r .id)"; P1="$(body "$r" | jq -r .position)"
R2="$(chat $S "$(msgs system "$SYSTEM" user "$TASK" assistant "$A1" user "$JUNK")")" || { echo "fail: R2"; exit 1; }
A2="$(jq -r '.choices[0].message.content' <<< "$R2")"
chat $S "$(msgs system "$SYSTEM" user "$TASK" assistant "$A1" user "$JUNK" assistant "$A2" user "$JUNK2")" > /dev/null
r="$(req POST /branch "{\"session_id\":\"$S\",\"label\":\"grep\"}")"; expect "POST /branch nested" 200 "$r"
B2="$(body "$r" | jq -r .id)"; P2="$(body "$r" | jq -r .position)"
[ "$P2" -gt "$P1" ] && ok "nested position $P2 > $P1" || fail "nested position $P2 <= $P1"
r="$(req GET "/branches?session_id=$S")"
[ "$(body "$r" | jq -c '[.data[] | [.id, .depth]]')" == "[[\"$B1\",0],[\"$B2\",1]]" ] && ok "list outermost first with depths" \
    || fail "list: $(body "$r")"

echo "[3] ownership"
expect "return from another session" 404 "$(req POST /return "{\"session_id\":\"other\",\"branch_id\":\"$B1\"}")"
expect "delete from another session" 404 "$(req DELETE /branch "{\"session_id\":\"other\",\"branch_id\":\"$B1\"}")"

echo "[4] churn, then jump back to the outer branch"
for k in churn-a churn-b churn-c; do chat $k "$(msgs system "Session $k." user "Count to five.")" > /dev/null; done
r="$(req POST /return "{\"session_id\":\"$S\",\"branch_id\":\"$B1\"}")"; expect "POST /return outer" 200 "$r"
[ "$(body "$r" | jq -r .position)" == "$P1" ] && ok "return position = branch position" || fail "return: $(body "$r")"
expect "inner branch closed with it" 404 "$(req POST /return "{\"session_id\":\"$S\",\"branch_id\":\"$B2\"}")"

echo "[5] the folded turn restores from the pinned branch point"
R3="$(chat $S "$(msgs system "$SYSTEM" user "$TASK" assistant "$A1" user "$SUM1")")" || { echo "fail: R3"; exit 1; }
C3="$(jq -r '.timings.cached_n' <<< "$R3")"
P1_PROMPT="$(jq -r '.usage.prompt_tokens' <<< "$R1")"
[ "$C3" -ge "$P1_PROMPT" ] && ok "folded turn cached_n=$C3 >= R1 prompt ($P1_PROMPT), branch at $P1" \
    || fail "folded turn cached_n=$C3 < R1 prompt $P1_PROMPT (pin lost to eviction?)"

echo "[6] DELETE without folding"
r="$(req POST /branch "{\"session_id\":\"$S\"}")"; expect "POST /branch" 200 "$r"
B3="$(body "$r" | jq -r .id)"
r="$(req DELETE /branch "{\"session_id\":\"$S\",\"branch_id\":\"$B3\"}")"; expect "DELETE /branch" 200 "$r"
[ "$(body "$r" | jq -r .deleted)" == true ] && ok "deleted" || fail "delete: $(body "$r")"
expect "DELETE again" 404 "$(req DELETE /branch "{\"session_id\":\"$S\",\"branch_id\":\"$B3\"}")"
r="$(req GET "/branches?session_id=$S")"
[ "$(body "$r" | jq '.data | length')" == 0 ] && ok "no open branches" || fail "list: $(body "$r")"

grep -E "\[hot-cache\] pinned|\[branches\]" "$LOG" | head -5
[ $EC -eq 0 ] && echo "PASS: branch endpoints" || { echo "FAIL: branch endpoints"; tail -20 "$LOG"; }
exit $EC
