"""Tests for finalized-revision impact comparison (stdlib unittest)."""
import json
import os
import pathlib
import random
import shutil
import tempfile
import threading
import unittest
import urllib.request

from app.diff import (
    BODY_DELETED, BODY_INSERTED, BODY_SUBSTITUTED, DELETE, INSERT, MATCH,
    PAGINATION_SHIFT, RETRANSLATED, SUBSTITUTE, UNCHANGED, _align_core,
    align_graphemes, compare_finalizations, grapheme_slots,
)
from app.errors import InputError
from app.graphemes import graphemes
from app.planner import plan
from app.server import build_server, get_store


def _rev(no, body, cpl=12, lpp=3, rules=None):
    payload = {"body": body,
               "paper": {"cells_per_line": cpl, "lines_per_page": lpp},
               "rules": {"rules": rules or []}}
    layout = plan(payload)
    assert layout["status"] == "ok", layout.get("conflict")
    return {"revision_no": no, "request_sha256": f"r{no}",
            "layout_sha256": f"l{no}", "request": {"body": body},
            "status": "ok", "layout": layout}, payload


def rev(*args, **kwargs):
    return _rev(*args, **kwargs)[0]


class AlignmentTests(unittest.TestCase):
    def test_match_substitute_delete_insert_tie_break(self):
        # repeated text: deletions come from the tail, never the middle
        ops = align_graphemes(graphemes("abab"), graphemes("ab"))
        self.assertEqual(
            ops, [(MATCH, 0, 0), (MATCH, 1, 1),
                  (DELETE, 2, None), (DELETE, 3, None)])
        ops = align_graphemes(graphemes("aaaa"), graphemes("aa"))
        self.assertEqual(
            ops, [(MATCH, 0, 0), (MATCH, 1, 1),
                  (DELETE, 2, None), (DELETE, 3, None)])
        # inserts follow the matched prefix
        ops = align_graphemes(graphemes("ab"), graphemes("abab"))
        self.assertEqual(
            ops, [(MATCH, 0, 0), (MATCH, 1, 1),
                  (INSERT, None, 2), (INSERT, None, 3)])

    def test_substitute_preferred_to_delete_insert_at_equal_cost(self):
        ops = align_graphemes(graphemes("abc"), graphemes("axc"))
        self.assertEqual(ops, [(MATCH, 0, 0), (SUBSTITUTE, 1, 1),
                               (MATCH, 2, 2)])

    def test_grapheme_clusters_are_aligned_whole(self):
        # 'e' + combining acute is one cluster; caf + e-combining is four
        # clusters, so appending 'x' yields four matches then one insertion.
        base = "caf" + "é"
        self.assertEqual(len(graphemes(base)), 4)
        ops = align_graphemes(graphemes(base), graphemes(base + "x"))
        self.assertEqual(
            [o[0] for o in ops],
            [MATCH, MATCH, MATCH, MATCH, INSERT])

    def test_empty_side(self):
        ops = align_graphemes([], graphemes("ab"))
        self.assertEqual([o[0] for o in ops], [INSERT, INSERT])
        ops = align_graphemes(graphemes("ab"), [])
        self.assertEqual([o[0] for o in ops], [DELETE, DELETE])

    def test_alignment_compares_grapheme_text_not_offsets(self):
        # Boundary case: insertions shift every later codepoint offset,
        # but the retained 'a' graphemes must still pair up as matches.
        ops = align_graphemes(graphemes("aa"), graphemes("acaba"))
        self.assertEqual(
            ops, [(MATCH, 0, 0), (INSERT, None, 1), (MATCH, 1, 2),
                  (INSERT, None, 3), (INSERT, None, 4)])
        ops = align_graphemes(graphemes("acaba"), graphemes("aa"))
        self.assertEqual(
            ops, [(MATCH, 0, 0), (DELETE, 1, None), (MATCH, 2, 1),
                  (DELETE, 3, None), (DELETE, 4, None)])

    def test_suffix_matches_keep_each_sides_own_index(self):
        # Common suffix with different body lengths: the paired indices
        # are per side, not a shared absolute position.
        ops = align_graphemes(graphemes("xabc"), graphemes("abc"))
        self.assertEqual(
            ops, [(DELETE, 0, None), (MATCH, 1, 0),
                  (MATCH, 2, 1), (MATCH, 3, 2)])

    def test_affix_trimming_never_changes_the_stream(self):
        # The prefix/suffix trimming is a table-size optimization only:
        # the emitted stream must equal the untrimmed DP+walk, even on
        # heavily repeated text (and multi-codepoint clusters).
        rng = random.Random(20260926)
        for alphabet in ("ab", "aab", "ab\u0301"):
            for _ in range(500):
                old = "".join(rng.choice(alphabet)
                              for _ in range(rng.randrange(10)))
                new = "".join(rng.choice(alphabet)
                              for _ in range(rng.randrange(10)))
                trimmed = align_graphemes(graphemes(old), graphemes(new))
                old_t = [c[0] for c in graphemes(old)]
                new_t = [c[0] for c in graphemes(new)]
                reference = _align_core(old_t, new_t,
                                        0, len(old_t), 0, len(new_t))
                self.assertEqual(trimmed, reference, (old, new))


class CompareTests(unittest.TestCase):
    def test_identical_revisions(self):
        r1 = rev(1, "hello world test")
        r2 = rev(2, "hello world test")
        rep = compare_finalizations(r1, r2)
        self.assertTrue(rep["identical"])
        self.assertEqual(rep["edit_cost"], 0)
        self.assertEqual(rep["change_count"], 0)
        self.assertEqual(rep["segments"], [])
        self.assertEqual(rep["changes"], [])
        self.assertEqual(rep["old_affected_pages"]["count"], 0)
        self.assertEqual(rep["new_reemboss_pages"]["count"], 0)

    def test_body_substitution_with_ranges_and_pages(self):
        r1 = rev(1, "hello world test")
        r2 = rev(2, "hallo world test")
        rep = compare_finalizations(r1, r2)
        self.assertFalse(rep["identical"])
        self.assertEqual(rep["counts"][BODY_SUBSTITUTED], 1)
        self.assertEqual(rep["edit_cost"], 1)
        self.assertEqual(len(rep["segments"]), 1)
        seg = rep["segments"][0]
        self.assertEqual(seg["kinds"], [BODY_SUBSTITUTED])
        self.assertEqual(seg["old_char_start"], 1)
        self.assertEqual(seg["old_char_end"], 2)
        self.assertEqual(seg["new_char_start"], 1)
        self.assertEqual(seg["new_char_end"], 2)
        self.assertIsNotNone(seg["old_page_range"])
        mark = rep["changes"][0]
        self.assertEqual(mark["kind"], BODY_SUBSTITUTED)
        self.assertEqual(mark["old"]["text"], "e")
        self.assertEqual(mark["new"]["text"], "a")
        # substitution changes content on both sides
        self.assertIn(1, rep["old_affected_pages"]["content_changed"])
        self.assertIn(1, rep["new_reemboss_pages"]["content_changed"])
        self.assertEqual(rep["old_affected_pages"]["cascade_reflow"], [])
        self.assertEqual(rep["new_reemboss_pages"]["cascade_reflow"], [])

    def test_deletion_segment_ranges(self):
        # A suffix truncation is a pure deletion run (any remaining common
        # characters would pair up as equal-cost substitutions instead).
        r1 = rev(1, "hello world qqq")
        r2 = rev(2, "hello")
        rep = compare_finalizations(r1, r2)
        self.assertGreaterEqual(rep["counts"][BODY_DELETED], 1)
        self.assertEqual(rep["counts"][BODY_INSERTED], 0)
        self.assertEqual(rep["counts"][BODY_SUBSTITUTED], 0)
        seg = rep["segments"][0]
        self.assertTrue(all(k == BODY_DELETED for k in seg["kinds"]))
        self.assertIsNotNone(seg["old_char_start"])
        self.assertIsNotNone(seg["old_char_end"])
        self.assertIsNone(seg["new_char_start"])
        self.assertIsNone(seg["new_char_end"])
        self.assertIsNotNone(seg["old_page_range"])
        # pure deletion still lands on a new-side page: the last retained
        # grapheme sits on new page 1, whose content is now shorter
        self.assertEqual(seg["new_page_range"], {"start": 1, "end": 1})
        self.assertEqual(rep["new_reemboss_pages"]["content_changed"], [1])
        self.assertEqual(rep["old_affected_pages"]["content_changed"], [1])

    def test_insertion_segment_ranges(self):
        r1 = rev(1, "hello")
        r2 = rev(2, "hello world qqq")
        rep = compare_finalizations(r1, r2)
        self.assertGreaterEqual(rep["counts"][BODY_INSERTED], 1)
        self.assertEqual(rep["counts"][BODY_DELETED], 0)
        self.assertEqual(rep["counts"][BODY_SUBSTITUTED], 0)
        seg = rep["segments"][0]
        self.assertTrue(all(k == BODY_INSERTED for k in seg["kinds"]))
        self.assertIsNone(seg["old_char_start"])
        self.assertIsNone(seg["old_char_end"])
        self.assertIsNotNone(seg["new_char_start"])
        self.assertIsNotNone(seg["new_char_end"])
        self.assertIsNone(seg["old_page_range"])
        self.assertIsNotNone(seg["new_page_range"])

    def test_coordinate_shifting_insertion_keeps_identical_graphemes(self):
        # Boundary example: old "aa" vs new "acaba".  The retained 'a'
        # must pair as a match (a pure pagination shift), not be
        # misjudged as a substitution because its codepoint offset moved.
        r1 = rev(1, "aa")
        r2 = rev(2, "acaba")
        rep = compare_finalizations(r1, r2)
        self.assertEqual(rep["edit_cost"], 3)
        self.assertEqual(rep["counts"][BODY_SUBSTITUTED], 0)
        self.assertEqual(rep["counts"][BODY_INSERTED], 3)
        self.assertEqual(rep["counts"][UNCHANGED], 1)
        self.assertEqual(rep["counts"][PAGINATION_SHIFT], 1)
        shift = [m for m in rep["changes"]
                 if m["kind"] == PAGINATION_SHIFT]
        self.assertEqual(len(shift), 1)
        self.assertEqual(shift[0]["old"]["grapheme_index"], 1)
        self.assertEqual(shift[0]["new"]["grapheme_index"], 2)
        self.assertEqual(shift[0]["old"]["text"], "a")
        self.assertEqual(shift[0]["new"]["text"], "a")
        # the inserted dots are real new content on page 1
        self.assertEqual(rep["new_reemboss_pages"]["content_changed"], [1])

    def test_middle_deletion_reports_new_side_landing_page(self):
        # Deleting "world " closes the gap in the new version; the page
        # the following "test" pulls up to is the re-emboss landing page.
        r1 = rev(1, "hello world test")
        r2 = rev(2, "hello test")
        rep = compare_finalizations(r1, r2)
        self.assertEqual(rep["counts"][BODY_DELETED], 6)
        self.assertEqual(rep["counts"][BODY_INSERTED], 0)
        self.assertEqual(rep["counts"][BODY_SUBSTITUTED], 0)
        self.assertEqual(rep["old_affected_pages"]["content_changed"], [1])
        self.assertEqual(rep["new_reemboss_pages"]["content_changed"], [1])
        seg = rep["segments"][0]
        self.assertEqual(seg["old_page_range"], {"start": 1, "end": 1})
        self.assertEqual(seg["new_page_range"], {"start": 1, "end": 1})

    def test_pagination_only_shift_is_cascade(self):
        # same text and dots, tighter paper moves cells to a second page
        r1 = rev(1, "hello world singing test", cpl=20, lpp=3)
        r2 = rev(2, "hello world singing test", cpl=8, lpp=3)
        self.assertLess(r1["layout"]["page_count"],
                        r2["layout"]["page_count"])
        rep = compare_finalizations(r1, r2)
        self.assertEqual(rep["counts"][RETRANSLATED], 0)
        self.assertGreater(rep["counts"][PAGINATION_SHIFT], 0)
        self.assertEqual(rep["counts"][BODY_SUBSTITUTED], 0)
        self.assertEqual(rep["old_affected_pages"]["content_changed"], [])
        self.assertEqual(rep["new_reemboss_pages"]["content_changed"], [])
        self.assertIn(1, rep["old_affected_pages"]["cascade_reflow"])
        self.assertIn(2, rep["new_reemboss_pages"]["cascade_reflow"])

    def test_retranslation_without_body_change(self):
        rules = [{"id": "ing", "pattern": "ing", "cells": ["346"]}]
        r1 = rev(1, "singing test", cpl=20, lpp=3, rules=rules)
        r2 = rev(2, "singing test", cpl=20, lpp=3, rules=[])
        rep = compare_finalizations(r1, r2)
        self.assertGreater(rep["counts"][RETRANSLATED], 0)
        self.assertEqual(rep["counts"][BODY_SUBSTITUTED], 0)
        self.assertEqual(rep["counts"][BODY_INSERTED], 0)
        self.assertEqual(rep["counts"][BODY_DELETED], 0)
        # mask changes are real content changes, not mere reflow
        self.assertIn(1, rep["old_affected_pages"]["content_changed"])
        self.assertIn(1, rep["new_reemboss_pages"]["content_changed"])

    def test_insertion_pages_are_content_and_lists_stay_disjoint(self):
        # Prefix insertion: the new dots on page 1 are real content; the
        # retained graphemes pair up as matches (text decides equality,
        # not the shifted codepoint offsets) and are marked as pagination
        # shifts -- chained flow.  Either way the two page buckets must
        # be disjoint.
        r1 = rev(1, "the quick brown fox jumps over lazy dog final end",
                 cpl=8, lpp=2)
        r2 = rev(2, "zzz the quick brown fox jumps over lazy dog final end",
                 cpl=8, lpp=2)
        rep = compare_finalizations(r1, r2)
        self.assertGreaterEqual(rep["counts"][BODY_INSERTED], 1)
        for side in ("old_affected_pages", "new_reemboss_pages"):
            pages = rep[side]
            self.assertFalse(
                set(pages["content_changed"]) & set(pages["cascade_reflow"]))
            self.assertEqual(
                pages["count"],
                len(pages["content_changed"]) + len(pages["cascade_reflow"]))
        self.assertIn(1, rep["new_reemboss_pages"]["content_changed"])

    def test_adjacent_changes_are_merged_into_one_segment(self):
        r1 = rev(1, "hello world test")
        r2 = rev(2, "heXXlo world tesZ")
        rep = compare_finalizations(r1, r2)
        changed = [m for m in rep["changes"] if m["kind"] != UNCHANGED]
        self.assertEqual(len(rep["segments"]), 1)
        self.assertEqual(rep["segments"][0]["mark_count"], len(changed))

    def test_segments_split_at_unchanged_region(self):
        r1 = rev(1, "hello world test")
        r2 = rev(2, "hallo warld test")
        rep = compare_finalizations(r1, r2)
        # two substitutions separated by unchanged characters
        self.assertEqual(rep["counts"][BODY_SUBSTITUTED], 2)
        self.assertEqual(len(rep["segments"]), 2)

    def test_grapheme_slots_cover_every_cluster(self):
        r, _ = _rev(1, "hello world")
        slots = grapheme_slots(r["layout"])
        self.assertEqual(len(slots), len(graphemes("hello world")))
        for g, entries in enumerate(slots):
            for page, line, col, mask in entries:
                self.assertGreaterEqual(page, 1)
                self.assertGreaterEqual(line, 1)
                self.assertGreaterEqual(col, 1)
                self.assertIsInstance(mask, int)

    def test_non_ok_revision_rejected(self):
        r1 = rev(1, "hello")
        bad = {"revision_no": 2, "request_sha256": "x", "layout_sha256": "y",
               "request": {"body": "hello"}, "status": "conflict",
               "layout": {"status": "conflict"}}
        with self.assertRaises(InputError):
            compare_finalizations(bad, r1)
        with self.assertRaises(InputError):
            compare_finalizations(r1, bad)


class HttpEndpointTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        import app.server as server
        from app import config as cfg
        cfg.DATA_DIR = pathlib.Path(self.dir)
        cfg.DB_PATH = os.path.join(self.dir, "t.db")
        server._STORE = None
        self.store = get_store()
        self.store.create_document("d")

        def finalize(payload):
            layout = plan(payload)
            self.assertEqual(layout["status"], "ok")
            no, _ = self.store.add_revision("d", payload, layout)
            from app.exporter import export_all
            stored = self.store.get_revision("d", no)
            art = export_all(
                {"doc_id": "d", "revision_no": no,
                 "request_sha256": stored["request_sha256"],
                 "layout_sha256": stored["layout_sha256"]},
                stored["layout"], payload)
            self.store.finalize("d", no, art["json"], art["svg"],
                                art["replay"])
            return no

        self.n1 = finalize({"body": "hello world test foo",
                            "paper": {"cells_per_line": 12, "lines_per_page": 2},
                            "rules": {"rules": []}})
        self.n2 = finalize({"body": "hello world test bar",
                            "paper": {"cells_per_line": 12, "lines_per_page": 2},
                            "rules": {"rules": []}})

        self.server = build_server(port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def _post(self, path, obj):
        data = json.dumps(obj).encode("utf-8")
        req = urllib.request.Request(
            self._url(path), data=data, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_compare_two_finalizations(self):
        status, rep = self._post(
            "/documents/d/finalizations/diff/compare",
            {"old_revision": self.n1, "new_revision": self.n2})
        self.assertEqual(status, 200)
        self.assertEqual(rep["status"], "compared")
        self.assertEqual(rep["document_id"], "d")
        self.assertEqual(rep["old_revision_no"], self.n1)
        self.assertEqual(rep["new_revision_no"], self.n2)
        self.assertFalse(rep["identical"])
        self.assertIn("segments", rep)
        self.assertIn("old_affected_pages", rep)
        self.assertIn("new_reemboss_pages", rep)

    def test_compare_unfinalized_revision_is_404(self):
        # append but do not finalize
        no, _ = self.store.add_revision(
            "d", {"body": "another fine revision",
                  "paper": {"cells_per_line": 12, "lines_per_page": 2},
                  "rules": {"rules": []}},
            plan({"body": "another fine revision",
                  "paper": {"cells_per_line": 12, "lines_per_page": 2},
                  "rules": {"rules": []}}))
        status, rep = self._post(
            "/documents/d/finalizations/diff/compare",
            {"old_revision": self.n1, "new_revision": no})
        self.assertEqual(status, 404)
        self.assertEqual(rep["error"], "not_found")

    def test_compare_unknown_revision_is_404(self):
        status, rep = self._post(
            "/documents/d/finalizations/diff/compare",
            {"old_revision": self.n1, "new_revision": 999})
        self.assertEqual(status, 404)

    def test_compare_same_revision_is_400(self):
        status, rep = self._post(
            "/documents/d/finalizations/diff/compare",
            {"old_revision": self.n1, "new_revision": self.n1})
        self.assertEqual(status, 400)
        self.assertEqual(rep["error"], "invalid_request")

    def test_compare_bad_payload_is_400(self):
        for payload in ({"old_revision": "1", "new_revision": self.n2},
                        {"old_revision": 0, "new_revision": self.n2},
                        {"old_revision": self.n1},
                        ["not", "an", "object"]):
            status, rep = self._post(
                "/documents/d/finalizations/diff/compare", payload)
            self.assertEqual(status, 400, payload)
            self.assertEqual(rep["error"], "invalid_request")

    def test_compare_missing_body_is_400(self):
        req = urllib.request.Request(
            self._url("/documents/d/finalizations/diff/compare"),
            data=b"", method="POST")
        try:
            urllib.request.urlopen(req, timeout=5)
            self.fail("expected 400")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)

    def test_compare_unknown_document_is_404(self):
        status, rep = self._post(
            "/documents/nope/finalizations/diff/compare",
            {"old_revision": self.n1, "new_revision": self.n2})
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
