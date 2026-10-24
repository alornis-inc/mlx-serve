//! Context-folding branches: `POST /branch`, `POST /return`, `GET /branches`, `DELETE /branch`.
//!
//! A branch is a PINNED PREFIX of one session's token history. The session id is the
//! request's `prompt_cache_key` (hashed exactly like `server.requestCacheKey`), so chat
//! requests need no new field. `/return` restores nothing eagerly: it keeps the branch's
//! pin as the session's ANCHOR, and the next chat turn on the folded history (prefix +
//! summary) restores the pinned checkpoint and prefills only the summary. Returning to an
//! outer branch closes every branch opened after it (jump-back to any checkpoint).
//!
//! This file is pure bookkeeping (no mlx): the snapshot side is the `Store` vtable,
//! implemented on `HotPrefixCache` and called on the inference thread only.

const std = @import("std");

/// The snapshot store's opaque handle for one pinned checkpoint (`cache_pins.PinId`).
/// Pinning the same prefix twice returns the same id, so a pin two branches share is
/// unpinned only when neither holds it.
pub const PinId = u64;

pub const Store = struct {
    ctx: *anyopaque,
    /// A copy (in `gpa`, caller frees) of the session's last committed turn, prompt ++
    /// generated; null when the store holds none in RAM.
    sessionHead: *const fn (ctx: *anyopaque, gpa: std.mem.Allocator, key: u64) anyerror!?[]u32,
    /// error.NoCheckpointAt when no checkpoint sits exactly at `tokens.len`.
    pin: *const fn (ctx: *anyopaque, key: u64, tokens: []const u32) anyerror!PinId,
    unpin: *const fn (ctx: *anyopaque, pin: PinId) void,
};

pub const Error = error{
    /// Also for a branch id of ANOTHER session: ownership is never an oracle.
    UnknownBranch,
    UnknownSession,
    /// The session has no committed turn to branch from.
    NoHistory,
    TooManyBranches,
    TooManySessions,
};

/// The store keeps 16 pins per session and drops the oldest past that: 15 branches + the anchor.
pub const MAX_OPEN_BRANCHES: usize = 15;
pub const MAX_SESSIONS: usize = 256;

pub fn sessionKey(session_id: []const u8) u64 {
    return std.hash.Wyhash.hash(1, session_id);
}

pub const Branch = struct {
    id: u64,
    /// Token position of the branch point: the pinned prefix is `head[0..pos]`.
    pos: usize,
    pin: PinId,
    label: []u8,
    created_ms: i64,
};

const Session = struct {
    /// Open branches in the order they were opened; positions never decrease.
    open: std.ArrayList(Branch) = .empty,
    /// Pin of the branch last returned to, held until the session branches or returns
    /// again, so the next turn's restore cannot lose it to LRU eviction.
    anchor: ?PinId = null,
    /// Length of the history `/return` folded back to; the store's head is stale until
    /// the next commit, so a branch taken before then starts here.
    anchor_pos: usize = 0,

    fn deinit(self: *Session, gpa: std.mem.Allocator) void {
        for (self.open.items) |b| gpa.free(b.label);
        self.open.deinit(gpa);
    }

    fn indexOf(self: *const Session, id: u64) ?usize {
        for (self.open.items, 0..) |b, i| if (b.id == id) return i;
        return null;
    }

    fn holds(self: *const Session, pin: PinId) bool {
        if (self.anchor == pin) return true;
        for (self.open.items) |b| if (b.pin == pin) return true;
        return false;
    }
};

/// Pins the caller must `Store.unpin`, owned by the caller (`gpa.free`).
pub const Released = []PinId;

pub const Registry = struct {
    mu: std.Io.Mutex = .init,
    sessions: std.AutoHashMapUnmanaged(u64, Session) = .{},
    next_id: u64 = 1,
    gpa: std.mem.Allocator,
    io: std.Io,

    pub fn init(io: std.Io, gpa: std.mem.Allocator) Registry {
        return .{ .io = io, .gpa = gpa };
    }

    pub fn deinit(self: *Registry) void {
        var it = self.sessions.valueIterator();
        while (it.next()) |s| s.deinit(self.gpa);
        self.sessions.deinit(self.gpa);
    }

    /// The position a new branch would pin: the anchor while one is set, else `head_len`.
    pub fn branchPos(self: *Registry, key: u64, head_len: usize) usize {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const s = self.sessions.getPtr(key) orelse return head_len;
        return if (s.anchor != null) @min(s.anchor_pos, head_len) else head_len;
    }

    /// Record a branch the caller already pinned at `pos`. On error the caller unpins `pin`
    /// unless `held` says the session already holds it. On success `released` is the old
    /// anchor, when nothing else holds it.
    pub fn open(self: *Registry, key: u64, pin: PinId, pos: usize, label: []const u8, now_ms: i64, released: *?PinId) !Branch {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const gop = try self.sessions.getOrPut(self.gpa, key);
        if (!gop.found_existing) {
            if (self.sessions.count() > MAX_SESSIONS) {
                self.sessions.removeByPtr(gop.key_ptr);
                return Error.TooManySessions;
            }
            gop.value_ptr.* = .{};
        }
        const s = gop.value_ptr;
        if (s.open.items.len >= MAX_OPEN_BRANCHES) return Error.TooManyBranches;
        const owned = try self.gpa.dupe(u8, label);
        errdefer self.gpa.free(owned);
        const b: Branch = .{ .id = self.next_id, .pos = pos, .pin = pin, .label = owned, .created_ms = now_ms };
        try s.open.append(self.gpa, b);
        self.next_id += 1;
        const old = s.anchor;
        s.anchor = null;
        released.* = if (old) |a| (if (s.holds(a)) null else a) else null;
        return b;
    }

    /// Fold back to branch `id`: it and every branch opened after it close, its pin becomes
    /// the anchor. Returns the closed branch (label owned by the caller) and the pins to unpin.
    pub fn returnTo(self: *Registry, key: u64, id: u64, released: *std.ArrayList(PinId)) !Branch {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const s = self.sessions.getPtr(key) orelse return Error.UnknownBranch;
        const i = s.indexOf(id) orelse return Error.UnknownBranch;
        try released.ensureUnusedCapacity(self.gpa, s.open.items.len - i);
        const target = s.open.items[i];
        const old_anchor = s.anchor;
        var closed: std.ArrayList(Branch) = .fromOwnedSlice(try self.gpa.dupe(Branch, s.open.items[i + 1 ..]));
        defer closed.deinit(self.gpa);
        s.open.shrinkRetainingCapacity(i);
        s.anchor = target.pin;
        s.anchor_pos = target.pos;
        for (closed.items) |b| {
            self.gpa.free(b.label);
            if (!s.holds(b.pin) and std.mem.indexOfScalar(PinId, released.items, b.pin) == null) released.appendAssumeCapacity(b.pin);
        }
        if (old_anchor) |a| if (!s.holds(a) and std.mem.indexOfScalar(PinId, released.items, a) == null) released.appendAssumeCapacity(a);
        return target;
    }

    /// Discard branch `id` without folding (keep the investigation). Returns its pin when
    /// nothing else in the session holds it.
    pub fn drop(self: *Registry, key: u64, id: u64) !?PinId {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const s = self.sessions.getPtr(key) orelse return Error.UnknownBranch;
        const i = s.indexOf(id) orelse return Error.UnknownBranch;
        const b = s.open.orderedRemove(i);
        self.gpa.free(b.label);
        const held = s.holds(b.pin);
        self.forgetIfIdle(key, s);
        return if (held) null else b.pin;
    }

    /// Does `key` hold `pin` (an idempotent re-pin of a prefix it already pinned)?
    pub fn holds(self: *Registry, key: u64, pin: PinId) bool {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const s = self.sessions.getPtr(key) orelse return false;
        return s.holds(pin);
    }

    fn forgetIfIdle(self: *Registry, key: u64, s: *Session) void {
        if (s.open.items.len > 0 or s.anchor != null) return;
        s.deinit(self.gpa);
        _ = self.sessions.remove(key);
    }

    /// `GET /branches` body: `{"session_id":…,"branches":[…]}`, outermost first.
    pub fn writeList(self: *Registry, key: u64, session_id_json: []const u8, w: *std.Io.Writer) !void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        try w.print("{{\"object\":\"list\",\"session_id\":{s},\"data\":[", .{session_id_json});
        if (self.sessions.getPtr(key)) |s| for (s.open.items, 0..) |b, depth| {
            if (depth > 0) try w.writeByte(',');
            try writeBranch(w, b, depth);
        };
        try w.writeAll("]}");
    }
};

pub fn writeBranch(w: *std.Io.Writer, b: Branch, depth: usize) !void {
    try w.print("{{\"id\":\"br_{d}\",\"object\":\"branch\",\"position\":{d},\"depth\":{d},\"created\":{d},\"label\":", .{ b.id, b.pos, depth, @divFloor(b.created_ms, 1000) });
    try std.json.Stringify.value(b.label, .{}, w);
    try w.writeByte('}');
}

/// `"br_12"` → 12; anything else is an unknown branch.
pub fn parseBranchId(s: []const u8) ?u64 {
    if (!std.mem.startsWith(u8, s, "br_")) return null;
    const id = std.fmt.parseInt(u64, s[3..], 10) catch return null;
    return if (id == 0) null else id;
}

// ── tests ──

const testing = std.testing;

const FakeStore = struct {
    head: []const u32 = &.{},
    pinned: std.ArrayList(PinId) = .empty,

    fn store(self: *FakeStore) Store {
        return .{ .ctx = self, .sessionHead = sessionHeadFn, .pin = pinFn, .unpin = unpinFn };
    }
    fn sessionHeadFn(ctx: *anyopaque, gpa: std.mem.Allocator, _: u64) anyerror!?[]u32 {
        const self: *FakeStore = @ptrCast(@alignCast(ctx));
        return if (self.head.len > 0) try gpa.dupe(u32, self.head) else null;
    }
    /// Idempotent per prefix, like the real store: the id is the prefix length.
    fn pinFn(ctx: *anyopaque, _: u64, tokens: []const u32) anyerror!PinId {
        const self: *FakeStore = @ptrCast(@alignCast(ctx));
        const p: PinId = tokens.len;
        if (std.mem.indexOfScalar(PinId, self.pinned.items, p) == null) try self.pinned.append(testing.allocator, p);
        return p;
    }
    fn unpinFn(ctx: *anyopaque, pin: PinId) void {
        const self: *FakeStore = @ptrCast(@alignCast(ctx));
        for (self.pinned.items, 0..) |p, i| if (p == pin) {
            _ = self.pinned.swapRemove(i);
            return;
        };
        @panic("unpin of a pin that is not held");
    }
};

/// What the handlers do, minus HTTP: pin first, then record.
fn branchVia(reg: *Registry, st: Store, key: u64, label: []const u8) !Branch {
    const head = try st.sessionHead(st.ctx, testing.allocator, key) orelse return Error.NoHistory;
    defer testing.allocator.free(head);
    const pos = reg.branchPos(key, head.len);
    const pin = try st.pin(st.ctx, key, head[0..pos]);
    const held = reg.holds(key, pin);
    var old: ?PinId = null;
    const b = reg.open(key, pin, pos, label, 0, &old) catch |err| {
        if (!held) st.unpin(st.ctx, pin);
        return err;
    };
    if (old) |p| st.unpin(st.ctx, p);
    return b;
}

fn returnVia(reg: *Registry, st: Store, key: u64, id: u64) !usize {
    var released: std.ArrayList(PinId) = .empty;
    defer released.deinit(testing.allocator);
    const b = try reg.returnTo(key, id, &released);
    testing.allocator.free(b.label);
    for (released.items) |p| st.unpin(st.ctx, p);
    return b.pos;
}

test "branches: nested branch and return releases every pin but the anchor" {
    var reg = Registry.init(testing.io, testing.allocator);
    defer reg.deinit();
    var fs: FakeStore = .{};
    defer fs.pinned.deinit(testing.allocator);
    const st = fs.store();
    const key = sessionKey("worker-1");

    const toks = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    fs.head = toks[0..4];
    const outer = try branchVia(&reg, st, key, "read repo");
    fs.head = toks[0..7];
    const inner = try branchVia(&reg, st, key, "grep");
    fs.head = toks[0..10];
    try testing.expectEqual(@as(usize, 2), fs.pinned.items.len);

    // Jump straight back past the inner branch: both close, the outer pin anchors.
    try testing.expectEqual(@as(usize, 4), try returnVia(&reg, st, key, outer.id));
    try testing.expectEqual(@as(usize, 1), fs.pinned.items.len);
    try testing.expectEqual(@as(PinId, 4), fs.pinned.items[0]);
    try testing.expectError(Error.UnknownBranch, returnVia(&reg, st, key, inner.id));

    // Before the next commit the store's head is stale: a new branch starts at the anchor,
    // and the store hands back the anchor's own pin, which must stay held.
    const again = try branchVia(&reg, st, key, "retry");
    try testing.expectEqual(@as(usize, 4), again.pos);
    try testing.expectEqual(@as(usize, 1), fs.pinned.items.len);
}

test "branches: a branch id never resolves in another session" {
    var reg = Registry.init(testing.io, testing.allocator);
    defer reg.deinit();
    var fs: FakeStore = .{ .head = &.{ 1, 2, 3 } };
    defer fs.pinned.deinit(testing.allocator);
    const st = fs.store();
    const b = try branchVia(&reg, st, sessionKey("a"), "");
    try testing.expectError(Error.UnknownBranch, returnVia(&reg, st, sessionKey("b"), b.id));
    try testing.expectError(Error.UnknownBranch, reg.drop(sessionKey("b"), b.id));
    st.unpin(st.ctx, (try reg.drop(sessionKey("a"), b.id)).?);
    try testing.expectEqual(@as(usize, 0), fs.pinned.items.len);
    try testing.expectEqual(@as(u32, 0), reg.sessions.count());
}

test "branches: the open-branch cap refuses and leaves the pin with the caller" {
    var reg = Registry.init(testing.io, testing.allocator);
    defer reg.deinit();
    var fs: FakeStore = .{};
    defer fs.pinned.deinit(testing.allocator);
    const st = fs.store();
    const key = sessionKey("w");
    const toks: [MAX_OPEN_BRANCHES + 2]u32 = @splat(0);
    for (0..MAX_OPEN_BRANCHES) |i| {
        fs.head = toks[0 .. i + 1];
        _ = try branchVia(&reg, st, key, "x");
    }
    fs.head = toks[0 .. MAX_OPEN_BRANCHES + 1];
    try testing.expectError(Error.TooManyBranches, branchVia(&reg, st, key, "x"));
    try testing.expectEqual(MAX_OPEN_BRANCHES, fs.pinned.items.len);
    var released: std.ArrayList(PinId) = .empty;
    defer released.deinit(testing.allocator);
    const first = try reg.returnTo(key, 1, &released);
    testing.allocator.free(first.label);
    try testing.expectEqual(MAX_OPEN_BRANCHES - 1, released.items.len);
}

test "branches: list JSON is outermost first with depth and escaped labels" {
    var reg = Registry.init(testing.io, testing.allocator);
    defer reg.deinit();
    var fs: FakeStore = .{ .head = &.{ 1, 2, 3 } };
    defer fs.pinned.deinit(testing.allocator);
    const st = fs.store();
    const key = sessionKey("s");
    _ = try branchVia(&reg, st, key, "say \"hi\"");
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    try reg.writeList(key, "\"s\"", &out.writer);
    try testing.expectEqualStrings(
        \\{"object":"list","session_id":"s","data":[{"id":"br_1","object":"branch","position":3,"depth":0,"created":0,"label":"say \"hi\""}]}
    , out.written());
}

test "branches: parseBranchId" {
    try testing.expectEqual(@as(?u64, 12), parseBranchId("br_12"));
    try testing.expectEqual(@as(?u64, null), parseBranchId("br_0"));
    try testing.expectEqual(@as(?u64, null), parseBranchId("12"));
    try testing.expectEqual(@as(?u64, null), parseBranchId("br_x"));
}

test "branches: two branches at one position share a pin until both are gone" {
    var reg = Registry.init(testing.io, testing.allocator);
    defer reg.deinit();
    var fs: FakeStore = .{ .head = &.{ 1, 2, 3 } };
    defer fs.pinned.deinit(testing.allocator);
    const st = fs.store();
    const key = sessionKey("s");
    const a = try branchVia(&reg, st, key, "a");
    const b = try branchVia(&reg, st, key, "b");
    try testing.expectEqual(a.pin, b.pin);
    try testing.expectEqual(@as(?PinId, null), try reg.drop(key, b.id));
    try testing.expectEqual(@as(?PinId, a.pin), try reg.drop(key, a.id));
}
