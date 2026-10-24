#!/bin/bash
# test_fold_reuse.sh — context folding on a hybrid (GDN) trunk: a resent history with an
# investigation cut out restores at the message boundary where it started, and decodes
# EXACTLY as if the investigation had never happened.
#
# The bar is the run without the investigation, not a cold prefill of the folded prompt: a
# cold prefill chunks the history differently from the turns that built it (decode steps vs
# prefill chunks), and on a hybrid that is the chunking class (AGENTS.md), not a state bug.
# The reference turns build the same prefix the same way, so greedy bytes must match.
#
#  1. Exactness: R1, R2 (investigation), R3 folded  ==  R1, R3 on a fresh server.
#  2. Nesting:   R1, inv1, inv1+inv2, fold inv2, fold inv1  ==  the references.
#  3. Parallel:  two sessions fold concurrently; each matches its serial reference.
#  4. Disk:      R1, R2 with the SSD tier, restart, R3 folded == reference (from disk).
# Every folded turn must also reuse at least the reference's prefix (cached_n).
#
# Env: MLX_FOLD_MODEL (default Qwen3.8-Flash-Next), PORT (19093), BINARY.
# Reverting the message-boundary capture in generate.zig turns the cached_n checks red
# (the restore falls back below the boundary).

set -uo pipefail

source "$(dirname "$0")/_lib_models.sh"
MODEL="${MLX_FOLD_MODEL:-$(find_model ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-3.3bpw mlx-community/Qwen3.5-4B-MLX-4bit)}"
PORT="${PORT:-19093}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
BASE="http://127.0.0.1:$PORT"

[ -d "$MODEL" ] || { echo "SKIP: model dir not found: $MODEL"; exit 0; }
[ -x "$BIN" ]   || { echo "fail: build mlx-serve first"; exit 1; }
command -v jq >/dev/null || { echo "needs jq"; exit 1; }

WORK="$(mktemp -d)"
LOG="$WORK/server.log"
SERVER_PID=""
trap 'stop_server; rm -rf "$WORK"; true' EXIT

# Own HOME: the SSD tier's root (~/.mlx-serve/kv-cache/<fingerprint>) must never be a
# running server's, and model settings stay the defaults.
mkdir -p "$WORK/home"
start_server() { # extra flags...
    HOME="$WORK/home" "$BIN" --model "$MODEL" --serve --port "$PORT" --ctx-size 16384 \
        --prefix-cache-entries 8 --no-mtp --log-level info "$@" >> "$LOG" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 600); do
        curl -sf --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"' && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: server died:"; tail -30 "$LOG"; exit 1; }
        sleep 0.5
    done
    echo "fail: server never became healthy"; tail -30 "$LOG"; exit 1
}
stop_server() {
    [ -n "$SERVER_PID" ] || return 0
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
}

SYSTEM="You are a careful coding worker. Answer in at most three sentences."
TASK="Task: explain what a ring buffer is and when it beats a linked list."
JUNK="$(python3 -c "print('Investigation log: ' + ' '.join(f'file_{i}.zig line {i*7}: no match;' for i in range(260)))")"
JUNK2="$(python3 -c "print('Nested search: ' + ' '.join(f'grep hit {i} in src/x{i}.zig;' for i in range(180)))")"
SUM1="Summary of the investigation: the codebase has no ring buffer yet; a fixed array is enough."
SUM2="Summary of the nested search: no hits outside src/x*.zig."

# chat <cache_key> <messages-json> -> response json (greedy, no thinking)
chat() {
    curl -sf --max-time 600 -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg k "$1" --argjson m "$2" \
            '{messages:$m,max_tokens:48,temperature:0,seed:1,stream:false,enable_thinking:false,prompt_cache_key:$k}')"
}
text_of()   { jq -r '.choices[0].message.content'; }
cached_of() { jq -r '.timings.cached_n'; }

msgs() { # role content role content ...
    local out="[]"
    while [ $# -ge 2 ]; do
        out="$(jq -c --arg r "$1" --arg c "$2" '. + [{role:$r,content:$c}]' <<< "$out")"
        shift 2
    done
    echo "$out"
}

EC=0
check_eq() { # name expected actual
    if [ "$2" == "$3" ]; then echo "  ok: $1 byte-identical"; else
        echo "  FAIL: $1 diverged"; diff <(printf "%s\n" "$2") <(printf "%s\n" "$3") | head -10; EC=1; fi
}
check_ge() { # name got floor
    if [ "$2" -ge "$3" ]; then echo "  ok: $1 cached_n=$2 >= $3"; else
        echo "  FAIL: $1 cached_n=$2 < $3 (restore fell below the fold point)"; EC=1; fi
}

# The flow for one session key; prints "A1<TAB>R3text<TAB>R3cached<TAB>R5text<TAB>R5cached".
session_flow() { # key
    local k="$1" r a1 m r3 r5
    r="$(chat "$k" "$(msgs system "$SYSTEM" user "$TASK")")" || return 1
    a1="$(text_of <<< "$r")"
    # Investigation 1 opens, then a nested investigation 2 inside it.
    chat "$k" "$(msgs system "$SYSTEM" user "$TASK" assistant "$a1" user "$JUNK")" > "$WORK/$k.r2" || return 1
    local a2; a2="$(text_of < "$WORK/$k.r2")"
    chat "$k" "$(msgs system "$SYSTEM" user "$TASK" assistant "$a1" user "$JUNK" assistant "$a2" user "$JUNK2")" > /dev/null || return 1
    # Fold 2: back inside investigation 1, its summary appended.
    r3="$(chat "$k" "$(msgs system "$SYSTEM" user "$TASK" assistant "$a1" user "$JUNK" assistant "$a2" user "$SUM2")")" || return 1
    # Fold 1: back to the task, its summary appended.
    r5="$(chat "$k" "$(msgs system "$SYSTEM" user "$TASK" assistant "$a1" user "$SUM1")")" || return 1
    printf "%s\t%s\t%s\t%s\t%s\n" "$a1" "$(text_of <<< "$r3" | tr '\n\t' '  ')" "$(cached_of <<< "$r3")" "$(text_of <<< "$r5" | tr '\n\t' '  ')" "$(cached_of <<< "$r5")"
}

# References: the same turns with the investigations never sent. Prints the same columns.
reference_flow() { # key
    local k="$1" r a1 r2 a2 r3 r5
    r="$(chat "$k" "$(msgs system "$SYSTEM" user "$TASK")")" || return 1
    a1="$(text_of <<< "$r")"
    r5="$(chat "$k" "$(msgs system "$SYSTEM" user "$TASK" assistant "$a1" user "$SUM1")")" || return 1
    r2="$(chat "$k" "$(msgs system "$SYSTEM" user "$TASK" assistant "$a1" user "$JUNK")")" || return 1
    a2="$(text_of <<< "$r2")"
    r3="$(chat "$k" "$(msgs system "$SYSTEM" user "$TASK" assistant "$a1" user "$JUNK" assistant "$a2" user "$SUM2")")" || return 1
    printf "%s\t%s\t%s\t%s\t%s\n" "$a1" "$(text_of <<< "$r3" | tr '\n\t' '  ')" "$(cached_of <<< "$r3")" "$(text_of <<< "$r5" | tr '\n\t' '  ')" "$(cached_of <<< "$r5")"
}

compare_rows() { # name folded-row reference-row
    local f r; IFS=$'\t' read -r -a f <<< "$2"; IFS=$'\t' read -r -a r <<< "$3"
    check_eq "$1 turn 1" "${r[0]}" "${f[0]}"
    check_eq "$1 nested fold" "${r[1]}" "${f[1]}"
    check_ge "$1 nested fold" "${f[2]}" "${r[2]}"
    check_eq "$1 outer fold" "${r[3]}" "${f[3]}"
    check_ge "$1 outer fold" "${f[4]}" "${r[4]}"
}

echo "--- reference (fresh server, no investigations) ---"
start_server
REF="$(reference_flow ref)" || { echo "fail: reference flow"; tail -30 "$LOG"; exit 1; }
stop_server

echo "--- 1+2. exactness and nesting (one session) ---"
start_server
ROW="$(session_flow s1)" || { echo "fail: session flow"; tail -30 "$LOG"; exit 1; }
compare_rows "serial" "$ROW" "$REF"
grep -E "\[hot-cache\] (restored|hybrid)" "$LOG" | tail -6 | sed 's/^/    log: /'

echo "--- 3. parallel (two sessions folding at once) ---"
session_flow p1 > "$WORK/p1.row" &
P1=$!
session_flow p2 > "$WORK/p2.row" &
P2=$!
wait $P1; wait $P2
compare_rows "parallel p1" "$(cat "$WORK/p1.row")" "$REF"
compare_rows "parallel p2" "$(cat "$WORK/p2.row")" "$REF"
stop_server

echo "--- 4. disk (restart between the investigation and the fold) ---"
start_server --prefix-cache-disk 4GB
R1="$(chat d1 "$(msgs system "$SYSTEM" user "$TASK")")"
A1="$(text_of <<< "$R1")"
chat d1 "$(msgs system "$SYSTEM" user "$TASK" assistant "$A1" user "$JUNK")" > /dev/null
stop_server   # SIGTERM persists the owed slices
start_server --prefix-cache-disk 4GB
R5="$(chat d1 "$(msgs system "$SYSTEM" user "$TASK" assistant "$A1" user "$SUM1")")"
IFS=$'\t' read -r -a REFCOL <<< "$REF"
check_eq "disk outer fold" "${REFCOL[3]}" "$(text_of <<< "$R5" | tr '\n\t' '  ')"
check_ge "disk outer fold" "$(cached_of <<< "$R5")" "${REFCOL[4]}"
stop_server

[ $EC -eq 0 ] && echo "PASS" || echo "FAIL"
exit $EC
