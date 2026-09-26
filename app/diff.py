"""Impact comparison between two finalized revisions of one document.

Given two already-finalized revision numbers of the *same* document, this
module reads each revision's body and its immutable per-page data only --
nothing is re-planned and no finalization is modified.

The comparison proceeds in two stages:

1. **Minimum-edit alignment on Unicode grapheme clusters.**  The two
   (NFC-normalized) bodies are split with the same UAX#29-core splitter
   the planner uses, then aligned with unit costs for substitution,
   deletion (present only in the old revision) and insertion (present only
   in the new one).  Only the grapheme **text** is compared -- the
   codepoint offsets carried by the clusters never take part in the
   equality test, so an insertion/deletion that shifts every later offset
   (old ``aa`` vs new ``acaba`` is the boundary case) cannot turn an
   identical grapheme into a phantom substitution.  The resulting
   operation stream carries the real per-side cluster indices and
   consumes every grapheme at most once on each side.  Among equal-cost
   alignments the lexicographically first operation stream is chosen,
   ordered ``match > substitute > delete > insert``; the same
   suffix-DP-plus-greedy-walk construction as the readback verifier is
   used, so repeated text yields a stable alignment.

   For tables within :data:`MAX_TABLE_CELLS` the exact suffix table is
   built directly.  Larger pairs are handled in bounded memory without
   changing the stream: the same suffix DP and greedy walk run only
   inside diagonal bands ``|i-j| <= w`` of growing width until the
   resulting stream costs at most ``w`` -- an exact certificate that the
   canonical optimal walk never left the band (a walk that has spent
   ``t`` cost at point ``(i, j)`` obeys ``|i-j| <= t``) and therefore
   that the stream is the untrimmed one.  Total cell work is
   ``O((n+m)*D)`` with ``O(n*D)`` memory; near-identical revisions of
   long bodies (e.g. 3163 repeated ``a`` with only the first grapheme
   changed, where the common suffix cannot be trimmed because its text
   reappears in the middle) thus align in a few thousand cells, while
   genuinely unrelated large rewrites are still rejected once the
   required band exceeds the guard.

2. **Per-paired-grapheme classification.**  Every matched grapheme is
   examined on both sides:

   * masks (and page/line/column positions) equal  -> ``unchanged``;
   * the braille mask sequences differ              -> ``retranslated``
                                                       (盲文重译);
   * the masks are equal but positions move         -> ``pagination_shift``
                                                       (仅分页位移).

   Substitution / deletion / insertion operations are respectively
   ``body_substituted`` / ``body_deleted`` / ``body_inserted``.

Adjacent changes are merged into diff segments carrying the old/new
codepoint ranges and page ranges.  The report also lists, for each side,
the affected pages split into pages with real content changes
(``content_changed`` -- embossed dots actually differ) and pages touched
only by chained reflow (``cascade_reflow`` -- identical dots, shifted
placement).  A pure deletion owns no grapheme on the new side, but the
gap it leaves still changes the new version: the segment and the
new-side page list therefore also record the **landing page** (落点页) --
the page of the first new-side grapheme after the deletion point (the
content that pulled up into the gap), or of the last retained grapheme
when the deletion runs to the end of the body.
"""
from array import array

from .errors import InputError
from .graphemes import graphemes

# Alignment operation codes, ordered for the equal-cost tie break:
# match > substitute > delete (old only) > insert (new only).
MATCH = "match"
SUBSTITUTE = "substitute"
DELETE = "delete"
INSERT = "insert"

# Classification of every aligned position.
BODY_SUBSTITUTED = "body_substituted"   # 正文改
BODY_DELETED = "body_deleted"           # 正文删
BODY_INSERTED = "body_inserted"         # 正文增
RETRANSLATED = "retranslated"           # 盲文重译
PAGINATION_SHIFT = "pagination_shift"   # 仅分页位移
UNCHANGED = "unchanged"                 # 完全不变

# Marks that change embossed dots (vs. marks that only move them).
_CONTENT_KINDS = frozenset({
    BODY_SUBSTITUTED, BODY_DELETED, BODY_INSERTED, RETRANSLATED})
_CHANGE_KINDS = frozenset({
    BODY_SUBSTITUTED, BODY_DELETED, BODY_INSERTED,
    RETRANSLATED, PAGINATION_SHIFT})

_OP_COST = {MATCH: 0, SUBSTITUTE: 1, DELETE: 1, INSERT: 1}

# Resource guard for grapheme alignment.  Small pairs are aligned with
# the exact quadratic suffix-DP table; pairs above this budget use an
# exact banded suffix-DP construction whose diagonal band widens until
# it provably contains the canonical optimal walk.  The final band has
# at most (n+1)*(2*D+1) entries (D = true edit distance); the widening
# rounds together stay under twice that, so near-identical revisions of
# bodies up to the grapheme cap cost only O((n+m)*D) cells, while
# pathological full-document rewrites of two unrelated huge texts still
# fail with InputError.
MAX_TABLE_CELLS = 10_000_000
# "infinity" for out-of-band DP cells: safely above any real suffix cost
# (each body has at most MAX_GRAPHEMES graphemes).
_INF = MAX_TABLE_CELLS * 2 + 1


# --------------------------------------------------------------------------
# Grapheme alignment (minimum edit distance, deterministic tie break)
# --------------------------------------------------------------------------
def align_graphemes(old_clusters, new_clusters):
    """Align two grapheme-cluster lists with minimum edit cost.

    Returns a list of ``(op, old_index_or_None, new_index_or_None)``
    triples.  Equal-cost alignments are resolved with the
    ``match > substitute > delete > insert`` lexicographic preference.

    Only the grapheme **text** decides equality; the codepoint offsets in
    the clusters are ignored here.  The returned indices are the real
    per-side cluster indices (each side keeps its own), and every
    grapheme is consumed at most once per side.

    The common prefix is matched for free first: equal heads are always
    matched by the greedy walk, so trimming them cannot change the
    resulting stream.  The remaining middle is aligned by
    :func:`_align_core`, which builds either the exact suffix table or,
    when that would exceed the cell guard, an exact diagonal-band
    alignment of growing width.  Either way the emitted stream is
    exactly the one the untrimmed suffix-DP-plus-greedy-walk would
    produce; the band only keeps the table small for near-identical
    revisions -- even when a common suffix cannot be forced because its
    text reappears in the middle (old ``aa`` vs new ``acaba``, or 3163
    repeated ``a`` with one changed head).
    """
    old_texts = [cluster[0] for cluster in old_clusters]
    new_texts = [cluster[0] for cluster in new_clusters]
    n, m = len(old_texts), len(new_texts)

    pre = 0
    while pre < n and pre < m and old_texts[pre] == new_texts[pre]:
        pre += 1

    ops = [(MATCH, i, i) for i in range(pre)]
    ops.extend(_align_core(old_texts, new_texts, pre, n, pre, m))
    return ops


def _align_core(a, b, lo_n, hi_n, lo_m, hi_m):
    """Suffix DP + greedy walk on ``a[lo_n:hi_n]`` / ``b[lo_m:hi_m]``.

    ``a`` and ``b`` are grapheme-*text* lists; returned indices are
    offsets into them (i.e. the real per-side cluster indices).

    The exact quadratic suffix table is built when its cell count is
    within :data:`MAX_TABLE_CELLS`.  Larger pairs run the identical
    DP+walk inside growing diagonal bands (:func:`_align_large`); the
    emitted stream is the same in both cases.
    """
    n, m = hi_n - lo_n, hi_m - lo_m
    if n * m > MAX_TABLE_CELLS:
        return _align_large(a, b, lo_n, hi_n, lo_m, hi_m)

    rest = [array("i", [0]) * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        rest[i][m] = n - i
    for j in range(m + 1):
        rest[n][j] = m - j
    for i in range(n - 1, -1, -1):
        row, nxt = rest[i], rest[i + 1]
        ai = a[lo_n + i]
        for j in range(m - 1, -1, -1):
            pair = 0 if ai == b[lo_m + j] else 1
            row[j] = min(nxt[j + 1] + pair, nxt[j] + 1, row[j + 1] + 1)

    ops = []
    i = j = 0
    while i < n or j < m:
        candidates = []
        if i < n and j < m:
            op = MATCH if a[lo_n + i] == b[lo_m + j] else SUBSTITUTE
            candidates.append((op, i + 1, j + 1, i, j))
        if i < n:
            candidates.append((DELETE, i + 1, j, i, None))
        if j < m:
            candidates.append((INSERT, i, j + 1, None, j))
        total = rest[i][j]
        chosen = None
        for opname, ni, nj, oi, nj2 in candidates:
            if _OP_COST[opname] + rest[ni][nj] == total:
                chosen = (opname, oi, nj2)
                break
        assert chosen is not None
        opname, oi, ni = chosen
        ops.append((opname,
                    None if oi is None else lo_n + oi,
                    None if ni is None else lo_m + ni))
        if opname in (MATCH, SUBSTITUTE, DELETE):
            i += 1
        if opname in (MATCH, SUBSTITUTE, INSERT):
            j += 1
    return ops

def _align_large(a, b, lo_n, hi_n, lo_m, hi_m):
    """Exact alignment for pairs whose full table exceeds the cell guard.

    The suffix DP + greedy walk is run inside diagonal bands
    ``|i - j| <= w`` (starting at the unavoidable length offset) until
    the stream the walk produces costs at most ``w``.  That test is an
    exact certificate:

    * if the emitted cost ``C`` obeys ``C <= w``, the canonical optimal
      walk never leaves the band -- at point ``(i, j)`` it has spent
      ``C - rest[i][j] <= C <= w``, and ``|i - j|`` cannot exceed the
      spent cost -- so every in-band value the walk used was the true
      suffix value, hence ``C`` is the true distance and the stream is
      exactly the untrimmed one;
    * if ``C > w`` (or the band cannot even reach the opposite corner),
      the band was too narrow and ``C`` is itself an upper bound on the
      true distance (it is the cost of one concrete edit stream), so the
      next band widens directly to ``C`` instead of doubling blindly.

    Total cell work stays O((n+m)*D): the widths climb monotonically
    toward D and overshoot it by less than a factor of two, and an exact
    final attempt at the observed cost admits every pair whose true band
    fits :data:`MAX_TABLE_CELLS`.  When that band would exceed the guard
    the pair is genuinely unrelated and is rejected with
    :class:`InputError` -- the same resource protection as the quadratic
    guard.
    """
    n, m = hi_n - lo_n, hi_m - lo_m
    width = max(abs(n - m), 1)
    last_cost = None
    while True:
        if (n + 1) * (2 * width + 1) > MAX_TABLE_CELLS:
            if (last_cost is not None
                    and last_cost > width
                    and (n + 1) * (2 * last_cost + 1)
                    <= MAX_TABLE_CELLS):
                width = last_cost
            else:
                raise InputError(
                    "revisions differ too much to align "
                    f"({n} x {m} graphemes needs a diagonal band beyond "
                    f"the {MAX_TABLE_CELLS}-cell limit)")
        ops, cost = _align_banded(
            a, b, lo_n, hi_n, lo_m, hi_m, width)
        if cost <= width:
            return ops
        last_cost = cost
        width = cost if cost > width else min(max(n, m), width * 2)


def _align_banded(a, b, lo_n, hi_n, lo_m, hi_m, dist):
    """Suffix DP + greedy walk restricted to the band ``|i-j| <= dist``.

    Returns ``(ops, cost)`` where ``ops`` uses real per-side cluster
    indices and ``cost`` is the stream's edit cost.  The returned stream
    is the canonical optimal one exactly when ``cost <= dist``
    (certificate described in :func:`_align_large`); with a too-narrow
    band the walk may be forced through unreachable cells, in which case
    the returned cost exceeds ``dist`` and signals that a wider band is
    required.

    The table is stored as variable-width rows in band-local indexing;
    its cell count is bounded by ``(n+1)*(2*dist+1)``.
    """
    n, m = hi_n - lo_n, hi_m - lo_m
    aa = a[lo_n:hi_n]
    bb = b[lo_m:hi_m]
    inf = _INF
    d = dist
    rest = []

    # Bottom-up suffix DP.  Row i spans j in [jmin, jmax]; the next row
    # i+1 starts at jmin_next = max(0, i+1-dist), and below() maps a
    # column to the next row's band-local index (out of range -> inf).
    for i in range(n, -1, -1):
        jmin = i - d if i > d else 0
        jmax = i + d if i + d < m else m
        size = jmax - jmin + 1
        row = array("i", [inf]) * size
        if i == n:
            # bottom boundary: only inserts remain; (n, m) is in band
            # whenever dist >= |n - m|.
            for li in range(size):
                row[li] = m - (jmin + li)
        else:
            nxt = rest[-1]
            nlen = len(nxt)
            # column-to-local-index mapping for the next row
            jmin_next = i + 1 - d if i + 1 > d else 0

            def below(jj, off=jmin_next):
                li = jj - off
                return nxt[li] if 0 <= li < nlen else inf
            ai = aa[i]
            right = inf
            li = size - 1
            for j in range(jmax, jmin - 1, -1):
                if j == m:
                    val = n - i  # right boundary: only deletes remain
                else:
                    diag = below(j + 1)
                    pair = 0 if ai == bb[j] else 1
                    v1 = diag + pair
                    v2 = below(j) + 1
                    v3 = right + 1
                    val = v1 if v1 < v2 else v2
                    if v3 < val:
                        val = v3
                row[li] = val
                right = val
                li -= 1
        rest.append(row)
    rest.reverse()

    def value(r, c):
        cmin = r - d if r > d else 0
        cmax = r + d if r + d < m else m
        if not (cmin <= c <= cmax):
            return inf
        return rest[r][c - cmin]

    ops = []
    i = j = 0
    cost = 0
    while i < n or j < m:
        total = value(i, j)
        if total >= inf:
            # Current cell unreachable within this band.  Advance in the
            # same tie-break order just to terminate; the certificate
            # (cost <= dist) will reject this band.
            if i < n and j < m:
                opname = MATCH if aa[i] == bb[j] else SUBSTITUTE
                oi, ni = i, j
            elif i < n:
                opname, oi, ni = DELETE, i, None
            else:
                opname, oi, ni = INSERT, None, j
        else:
            opname = oi = ni = None
            if i < n and j < m:
                step_cost = 0 if aa[i] == bb[j] else 1
                cand = MATCH if step_cost == 0 else SUBSTITUTE
                if step_cost + value(i + 1, j + 1) == total:
                    opname, oi, ni = cand, i, j
            if opname is None and i < n and value(i + 1, j) + 1 == total:
                opname, oi, ni = DELETE, i, None
            if opname is None and j < m and value(i, j + 1) + 1 == total:
                opname, oi, ni = INSERT, None, j
            assert opname is not None
        cost += _OP_COST[opname]
        ops.append((opname,
                    None if oi is None else lo_n + oi,
                    None if ni is None else lo_m + ni))
        if opname in (MATCH, SUBSTITUTE, DELETE):
            i += 1
        if opname in (MATCH, SUBSTITUTE, INSERT):
            j += 1
    return ops, cost


# --------------------------------------------------------------------------
# Per-grapheme braille data from finalized pages
# --------------------------------------------------------------------------
def grapheme_slots(layout):
    """Map every grapheme index of a finalized layout to its emitted cells.

    Returns a list indexed by grapheme index; each entry is a list of
    ``(page, line, col, mask)`` tuples in reading order (pages are
    emitted in order and cells within a line are column-sorted).  Space
    atoms dropped at a line boundary simply contribute no cells.
    """
    slots = []

    def ensure(idx):
        while len(slots) <= idx:
            slots.append([])

    for pg in layout.get("pages", []):
        page_no = pg["page"]
        for ln in pg.get("lines", []):
            line_no = ln["line"]
            for cell in ln.get("cells", []):
                gs, ge = cell["grapheme_start"], cell["grapheme_end"]
                ensure(ge - 1)
                for g in range(gs, ge):
                    slots[g].append(
                        (page_no, line_no, cell["col"], cell["mask"]))
    return slots


def _masks(slots, g):
    return [t[3] for t in slots[g]] if g < len(slots) else []


def _positions(slots, g):
    return [(t[0], t[1], t[2]) for t in slots[g]] if g < len(slots) else []


def _pages(slots, g):
    return {t[0] for t in slots[g]} if g < len(slots) else set()


# --------------------------------------------------------------------------
# Mark construction and segment merging
# --------------------------------------------------------------------------
def _classify(op, oi, ni, old_slots, new_slots):
    if op == SUBSTITUTE:
        return BODY_SUBSTITUTED
    if op == DELETE:
        return BODY_DELETED
    if op == INSERT:
        return BODY_INSERTED
    # matched grapheme: compare braille, then placement
    if _masks(old_slots, oi) != _masks(new_slots, ni):
        return RETRANSLATED
    if _positions(old_slots, oi) != _positions(new_slots, ni):
        return PAGINATION_SHIFT
    return UNCHANGED


def _mark(kind, oi, ni, old_clusters, new_clusters,
          old_slots, new_slots):
    """One changed aligned position, with old/new ranges and page spans."""
    rec = {"kind": kind}
    if oi is not None:
        _, cs, ce = old_clusters[oi]
        rec["old"] = {
            "grapheme_index": oi, "char_start": cs, "char_end": ce,
            "text": old_clusters[oi][0],
            "pages": sorted(_pages(old_slots, oi)),
        }
    if ni is not None:
        _, cs, ce = new_clusters[ni]
        rec["new"] = {
            "grapheme_index": ni, "char_start": cs, "char_end": ce,
            "text": new_clusters[ni][0],
            "pages": sorted(_pages(new_slots, ni)),
        }
    return rec


def _page_range(pages):
    return {"start": min(pages), "end": max(pages)} if pages else None


def compare_finalizations(old_rev, new_rev):
    """Compare two stored, finalized revisions (read-only, no re-planning).

    Each argument is a revision dict as returned by ``Store.get_revision``
    with ``status == "ok"``.  Returns the impact report; raises
    :class:`InputError` for non-ok layouts or revisions too divergent to
    align within the table-size guard.
    """
    for label, rev in (("old", old_rev), ("new", new_rev)):
        if rev.get("status") != "ok" or rev.get("layout", {}).get(
                "status") != "ok":
            raise InputError(
                f"the {label} revision is not in a finalized 'ok' state")

    old_body, new_body = old_rev["request"]["body"], new_rev["request"]["body"]
    old_clusters = graphemes(old_body)
    new_clusters = graphemes(new_body)
    old_slots = grapheme_slots(old_rev["layout"])
    new_slots = grapheme_slots(new_rev["layout"])

    alignment = align_graphemes(old_clusters, new_clusters)
    edit_cost = sum(_OP_COST[op] for op, _, _ in alignment)

    # New-side neighbours of every aligned position.  A deletion owns no
    # new-side grapheme, but the gap it leaves still lands on a new-side
    # page: the page of the first new grapheme after the deletion point
    # (the content that pulled up into the gap), or of the last retained
    # grapheme when the deletion runs to the end of the body.
    prev_new = [None] * len(alignment)
    next_new = [None] * len(alignment)
    last = None
    for k, (_, _, ni) in enumerate(alignment):
        prev_new[k] = last
        if ni is not None:
            last = ni
    last = None
    for k in range(len(alignment) - 1, -1, -1):
        next_new[k] = last
        if alignment[k][2] is not None:
            last = alignment[k][2]

    def landing_pages(k):
        ni = next_new[k]
        if ni is None:
            ni = prev_new[k]
        return _pages(new_slots, ni) if ni is not None else set()

    counts = {BODY_SUBSTITUTED: 0, BODY_DELETED: 0, BODY_INSERTED: 0,
              RETRANSLATED: 0, PAGINATION_SHIFT: 0, UNCHANGED: 0}
    marks = []
    segments = []

    # Page impact sets; content changes win over chained reflow.
    old_content, new_content = set(), set()
    old_cascade, new_cascade = set(), set()

    cur = None  # open segment accumulator

    def note_pages(kind, oi, ni, landing=frozenset()):
        opages = _pages(old_slots, oi) if oi is not None else set()
        npages = (_pages(new_slots, ni) if ni is not None else set())
        npages = npages | landing
        if kind in _CONTENT_KINDS:
            old_content.update(opages)
            new_content.update(npages)
        else:  # pagination shift
            # Identical dots on different (or partly different) pages:
            # every touched page is only re-embossed due to chained flow.
            old_cascade.update(opages - npages)
            new_cascade.update(npages - opages)
            common = opages & npages
            old_cascade.update(common)
            new_cascade.update(common)

    def flush(seg):
        if seg is None:
            return
        old_pages, new_pages = seg.pop("_pages_old"), seg.pop("_pages_new")
        seg["old_page_range"] = _page_range(old_pages)
        seg["new_page_range"] = _page_range(new_pages)
        segments.append(seg)

    for k, (op, oi, ni) in enumerate(alignment):
        kind = _classify(op, oi, ni, old_slots, new_slots)
        counts[kind] += 1
        if kind == UNCHANGED:
            flush(cur)
            cur = None
            continue
        landing = landing_pages(k) if op == DELETE else set()
        note_pages(kind, oi, ni, landing)
        marks.append(_mark(kind, oi, ni, old_clusters, new_clusters,
                           old_slots, new_slots))
        if cur is None:
            cur = {"kinds": [], "mark_count": 0,
                   "old_char_start": None, "old_char_end": None,
                   "new_char_start": None, "new_char_end": None,
                   "_pages_old": set(), "_pages_new": set()}
        cur["kinds"].append(kind)
        cur["mark_count"] += 1
        if oi is not None:
            if cur["old_char_start"] is None:
                cur["old_char_start"] = old_clusters[oi][1]
            cur["old_char_end"] = old_clusters[oi][2]
            cur["_pages_old"].update(_pages(old_slots, oi))
        if ni is not None:
            if cur["new_char_start"] is None:
                cur["new_char_start"] = new_clusters[ni][1]
            cur["new_char_end"] = new_clusters[ni][2]
            cur["_pages_new"].update(_pages(new_slots, ni))
        # A pure deletion still lands on a new-side page (see above).
        cur["_pages_new"].update(landing)
    flush(cur)

    # Content pages are a superset of the cascade signal; keep lists
    # disjoint with content changes taking precedence.
    old_cascade -= old_content
    new_cascade -= new_content

    change_count = sum(counts[k] for k in _CHANGE_KINDS)
    old_page_count = old_rev["layout"].get("page_count", 0)
    new_page_count = new_rev["layout"].get("page_count", 0)
    return {
        "status": "compared",
        "identical": change_count == 0,
        "old_revision_no": old_rev["revision_no"],
        "new_revision_no": new_rev["revision_no"],
        "old_request_sha256": old_rev["request_sha256"],
        "new_request_sha256": new_rev["request_sha256"],
        "old_layout_sha256": old_rev["layout_sha256"],
        "new_layout_sha256": new_rev["layout_sha256"],
        "old_page_count": old_page_count,
        "new_page_count": new_page_count,
        "old_grapheme_count": len(old_clusters),
        "new_grapheme_count": len(new_clusters),
        "edit_cost": edit_cost,
        "counts": counts,
        "change_count": change_count,
        "segments": segments,
        "changes": marks,
        "old_affected_pages": {
            "content_changed": sorted(old_content),
            "cascade_reflow": sorted(old_cascade),
            "count": len(old_content) + len(old_cascade),
        },
        "new_reemboss_pages": {
            "content_changed": sorted(new_content),
            "cascade_reflow": sorted(new_cascade),
            "count": len(new_content) + len(new_cascade),
        },
    }
