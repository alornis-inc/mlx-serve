//! Pinned prefix-cache checkpoints (context folding).
//!
//! A pin names a token prefix whose SSM checkpoint and KV must survive: a worker that opened an
//! investigation at that prefix will resend the history folded back to it. Pins are
//! CONTENT-addressed (position + a digest of the tokens before it), never entry-addressed: a
//! RAM entry is replaced on every commit and a disk record outlives the process's entry ids,
//! but the same prefix keeps the same digest on both tiers.
//!
//! Pure Zig, no MLX: the thinning and eviction sites ask `pinnedAt` / `holds`, nothing here
//! owns cache state. Inference thread only, like the cache it guards.

const std = @import("std");

/// Cap per cache key: nested investigations are a few deep; a harness that never unpins must
/// not keep a whole session's checkpoints (115 MB each on Qwen3.8-Flash-Next) forever.
pub const MAX_PINS_PER_KEY: usize = 16;

pub const PinId = u64;

pub const Pin = struct {
    id: PinId,
    cache_key: u64,
    /// The checkpoint position: the state after `pos` tokens.
    pos: usize,
    digest: u64,
};

pub fn prefixDigest(tokens: []const u32) u64 {
    return std.hash.Wyhash.hash(0x6d6c782d70696e, std.mem.sliceAsBytes(tokens));
}

pub const PinSet = struct {
    allocator: std.mem.Allocator,
    pins: std.ArrayList(Pin) = .empty,
    next_id: PinId = 1,

    pub fn init(allocator: std.mem.Allocator) PinSet {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *PinSet) void {
        self.pins.deinit(self.allocator);
    }

    /// Pin `tokens` (the state after all of them) for `cache_key`. Re-pinning the same prefix
    /// returns the existing id. Over the per-key cap the key's OLDEST pin goes.
    pub fn add(self: *PinSet, cache_key: u64, tokens: []const u32) !PinId {
        const digest = prefixDigest(tokens);
        for (self.pins.items) |p| {
            if (p.cache_key == cache_key and p.pos == tokens.len and p.digest == digest) return p.id;
        }
        if (self.countKey(cache_key) >= MAX_PINS_PER_KEY) {
            for (self.pins.items, 0..) |p, i| if (p.cache_key == cache_key) {
                _ = self.pins.orderedRemove(i);
                break;
            };
        }
        const id = self.next_id;
        try self.pins.append(self.allocator, .{ .id = id, .cache_key = cache_key, .pos = tokens.len, .digest = digest });
        self.next_id += 1;
        return id;
    }

    pub fn remove(self: *PinSet, id: PinId) bool {
        for (self.pins.items, 0..) |p, i| if (p.id == id) {
            _ = self.pins.orderedRemove(i);
            return true;
        };
        return false;
    }

    /// Session end: drop every pin of `cache_key`. Returns how many went.
    pub fn removeKey(self: *PinSet, cache_key: u64) usize {
        var n: usize = 0;
        var i: usize = 0;
        while (i < self.pins.items.len) {
            if (self.pins.items[i].cache_key == cache_key) {
                _ = self.pins.orderedRemove(i);
                n += 1;
            } else i += 1;
        }
        return n;
    }

    pub fn countKey(self: *const PinSet, cache_key: u64) usize {
        var n: usize = 0;
        for (self.pins.items) |p| n += @intFromBool(p.cache_key == cache_key);
        return n;
    }

    /// Is the state after `tokens[0..pos]` pinned?
    pub fn pinnedAt(self: *const PinSet, tokens: []const u32, pos: usize) bool {
        if (pos > tokens.len) return false;
        var digest: ?u64 = null;
        for (self.pins.items) |p| {
            if (p.pos != pos) continue;
            if (digest == null) digest = prefixDigest(tokens[0..pos]);
            if (p.digest == digest.?) return true;
        }
        return false;
    }

    /// Does a record of `tokens` cover `pin` (a restore at the pin can come from it)?
    pub fn covers(pin: Pin, tokens: []const u32) bool {
        return pin.pos <= tokens.len and pin.digest == prefixDigest(tokens[0..pin.pos]);
    }

    /// Is the record `records[idx]` some pin's KEEPER: the shortest record covering it (the
    /// lowest index on a tie)? Only keepers are preferred over other eviction victims; a longer
    /// record of the same session (the investigation itself) is ordinary prey. `tokensOf` maps a
    /// record to its tokens.
    pub fn keeps(self: *const PinSet, records: anytype, idx: usize, comptime tokensOf: anytype) bool {
        const mine = tokensOf(&records[idx]);
        for (self.pins.items) |p| {
            if (!covers(p, mine)) continue;
            var shortest = true;
            for (records, 0..) |*r, j| {
                if (j == idx) continue;
                const other = tokensOf(r);
                if (other.len > mine.len or (other.len == mine.len and j > idx)) continue;
                if (covers(p, other)) {
                    shortest = false;
                    break;
                }
            }
            if (shortest) return true;
        }
        return false;
    }

    /// `out[i]` = is `positions[i]` pinned for `tokens`. `positions` may be any integer slice.
    pub fn mask(self: *const PinSet, tokens: []const u32, positions: anytype, out: []bool) void {
        std.debug.assert(out.len == positions.len);
        for (positions, out) |p, *o| o.* = self.pinnedAt(tokens, @intCast(p));
    }
};

fn sliceOf(r: *const []const u32) []const u32 {
    return r.*;
}

test "PinSet: content-addressed, per-key cap, session drop" {
    const t = std.testing;
    var set = PinSet.init(t.allocator);
    defer set.deinit();
    const toks = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const a = try set.add(7, toks[0..4]);
    try t.expectEqual(a, try set.add(7, toks[0..4]));
    try t.expect(set.pinnedAt(&toks, 4));
    try t.expect(!set.pinnedAt(&toks, 3));
    // Same position, different prefix: not pinned.
    const other = [_]u32{ 1, 2, 9, 4, 5, 6 };
    try t.expect(!set.pinnedAt(&other, 4));
    try t.expect(PinSet.covers(set.pins.items[0], &toks));
    try t.expect(!PinSet.covers(set.pins.items[0], &other));
    try t.expect(!PinSet.covers(set.pins.items[0], toks[0..3]));
    // Keeper = the shortest record covering the pin; a longer one of the session is not.
    const records = [_][]const u32{ &toks, toks[0..5], &other };
    try t.expect(!set.keeps(&records, 0, sliceOf));
    try t.expect(set.keeps(&records, 1, sliceOf));
    try t.expect(!set.keeps(&records, 2, sliceOf));
    var m: [3]bool = undefined;
    set.mask(&toks, &[_]u32{ 2, 4, 6 }, &m);
    try t.expectEqualSlices(bool, &.{ false, true, false }, &m);
    try t.expect(set.remove(a));
    try t.expect(!set.remove(a));
    try t.expect(!set.keeps(&records, 1, sliceOf));

    var long: [MAX_PINS_PER_KEY + 2]u32 = undefined;
    for (&long, 0..) |*x, i| x.* = @intCast(i);
    for (1..MAX_PINS_PER_KEY + 2) |n| _ = try set.add(3, long[0..n]);
    _ = try set.add(4, long[0..1]);
    try t.expectEqual(MAX_PINS_PER_KEY, set.countKey(3));
    try t.expect(set.pinnedAt(&long, MAX_PINS_PER_KEY + 1));
    try t.expectEqual(MAX_PINS_PER_KEY, set.removeKey(3));
    // Key 3's oldest (position 1) went at the cap; only key 4's pin is left, at position 1.
    try t.expectEqual(@as(usize, 1), set.pins.items.len);
    try t.expect(set.pinnedAt(&long, 1));
    try t.expect(!set.pinnedAt(&long, 2));
}
