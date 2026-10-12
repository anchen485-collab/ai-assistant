"""治理回归测试：纯本地，不请求模型、不接触真实向量库。"""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.rag import governance as policy


class GovernancePolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / policy.METADATA_FILE_NAME
        self.today = date(2026, 9, 27)

    def write(self, content):
        self.path.write_text(json.dumps(content), encoding="utf-8")
        return policy.load_metadata(self.root)

    def test_missing_config_stops_instead_of_reactivating(self):
        with self.assertRaises(ValueError):
            policy.load_metadata(self.root)

    def test_broken_json_stops(self):
        self.path.write_text('{"default":', encoding="utf-8")
        with self.assertRaises(ValueError):
            policy.load_metadata(self.root)

    def test_invalid_utf8_stops(self):
        self.path.write_bytes(b"\xff\xfe\xff")
        with self.assertRaises(ValueError):
            policy.load_metadata(self.root)

    def test_utf8_bom_supported(self):
        self.path.write_text('{"default": {"status": "inactive"}}', encoding="utf-8-sig")
        catalog = policy.load_metadata(self.root)
        self.assertEqual(catalog["default"]["status"], "inactive")

    def test_duplicate_json_keys_stops(self):
        self.path.write_text('{"default": {"status": "inactive", "status": "active"}}')
        with self.assertRaises(ValueError):
            policy.load_metadata(self.root)

    def test_bad_root_and_sections(self):
        for content in [[], None, {"typo": {}}, {"default": []}, {"documents": []},
                        {"documents": {"a.txt": None}}]:
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.write(content)

    def test_unknown_fields_rejected(self):
        for fields in [{"stats": "inactive"}, {"authority_rnak": 100}, {"nested": {}}]:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.write({"default": fields})

    def test_default_inheritance_and_override(self):
        catalog = self.write({
            "default": {"owner": "team", "authority_rank": 10, "version": "1.0"},
            "documents": {"a.txt": {"knowledge_id": "a", "authority_rank": 20}},
        })
        a = policy.metadata_for_document(Path("a.txt"), catalog)
        b = policy.metadata_for_document(Path("b.txt"), catalog)
        self.assertEqual((a["authority_rank"], b["authority_rank"]), (20, 10))
        self.assertEqual(a["owner"], "team")
        self.assertEqual(b["knowledge_id"], "b.txt")
        self.assertTrue(all(type(v) in (str, int) for v in a.values()))

    def test_old_business_document_id_migrates_without_internal_collision(self):
        catalog = self.write({"documents": {"a.txt": {"document_id": "readable"}}})
        meta = policy.metadata_for_document(Path("a.txt"), catalog)
        self.assertEqual(meta["knowledge_id"], "readable")
        self.assertNotIn("document_id", meta)

    def test_conflicting_alias_rejected(self):
        with self.assertRaises(ValueError):
            self.write({"documents": {"a.txt": {"document_id": "old", "knowledge_id": "new"}}})

    def test_duplicate_knowledge_id_rejected(self):
        with self.assertRaises(ValueError):
            self.write({"documents": {"a.txt": {"knowledge_id": "same"},
                                      "b.txt": {"knowledge_id": "same"}}})

    def test_shared_business_id_rejected(self):
        with self.assertRaises(ValueError):
            self.write({"default": {"knowledge_id": "shared"}})

    def test_filename_keys_are_flat_document_names(self):
        for filename in ["a", "../a.txt", "x/a.docx", "x\\a.txt", "meta.json"]:
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                self.write({"documents": {filename: {}}})

    def test_rank_is_nonnegative_integer_not_bool_or_string(self):
        for rank in [-1, True, 1.5, "100", None, [], {}]:
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                self.write({"default": {"authority_rank": rank}})

    def test_other_fields_are_strings(self):
        for field in ["owner", "version", "status", "source_url", "region", "effective_to"]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.write({"default": {field: None}})

    def test_status_is_normalized_and_unknown_rejected(self):
        catalog = self.write({"default": {"status": " INACTIVE "}})
        self.assertEqual(catalog["default"]["status"], "inactive")
        for status in ["", "activ", "deleted"]:
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.write({"default": {"status": status}})

    def test_only_active_records_are_eligible(self):
        for status in ["inactive", "archived", "draft", "", None]:
            with self.subTest(status=status):
                self.assertFalse(policy.is_effective({"status": status}, self.today))
        self.assertFalse(policy.is_effective({}, self.today))
        self.assertTrue(policy.is_effective({"status": "active"}, self.today))

    def test_effective_boundaries_are_inclusive(self):
        metadata = {"status": "active", "effective_from": "2026-09-27", "effective_to": "2026-09-27"}
        self.assertTrue(policy.is_effective(metadata, self.today))
        self.assertFalse(policy.is_effective(metadata, date(2026, 9, 26)))
        self.assertFalse(policy.is_effective(metadata, date(2026, 9, 28)))

    def test_invalid_dates_fail_closed_in_config_and_records(self):
        for value in ["2026-02-30", "2026-13-01", "yesterday", "20260927", "2026-9-27", None, 0]:
            for field in ["effective_from", "effective_to"]:
                with self.subTest(value=value, field=field):
                    self.assertFalse(policy.is_effective({"status": "active", field: value}, self.today))
                    with self.assertRaises(ValueError):
                        self.write({"default": {field: value}})

    def test_reverse_date_range_rejected_after_inheritance(self):
        with self.assertRaises(ValueError):
            self.write({"default": {"effective_to": "2026-01-01"},
                        "documents": {"a.txt": {"effective_from": "2026-02-01"}}})

    def test_reverse_default_range_rejected(self):
        with self.assertRaises(ValueError):
            self.write({"default": {"effective_from": "2026-02-01", "effective_to": "2026-01-01"}})

    def test_malformed_unused_document_is_also_validated(self):
        with self.assertRaises(ValueError):
            self.write({"documents": {"unused.docx": {"effective_from": "bad"}}})

    def test_date_timezone_is_explicit(self):
        self.assertEqual(policy.BUSINESS_TIMEZONE.utcoffset(None).total_seconds(), 8 * 3600)
        self.assertIsInstance(policy.business_date(), date)

    def test_versions_compare_numerically_and_equivalent_forms_tie(self):
        self.assertGreater(policy.version_key("v1.10"), policy.version_key("1.2"))
        self.assertEqual(policy.version_key("1.0"), policy.version_key("V1.0.0"))
        self.assertGreater(policy.version_key("2"), policy.version_key("1.99"))

    def test_unsupported_versions_are_not_silently_guessed(self):
        for version in ["latest", "1.0-rc1", "1..2", 1, None]:
            with self.subTest(version=version), self.assertRaises(ValueError):
                self.write({"default": {"version": version}})

    def test_invalid_stored_ordering_fields_fail_closed(self):
        for field, value in [("authority_rank", "bad"), ("authority_rank", True), ("version", "latest")]:
            self.assertFalse(policy.is_effective({"status": "active", field: value}, self.today))

    def test_authority_then_version_then_effective_date(self):
        records = [
            {"authority_rank": 10, "version": "100"},
            {"authority_rank": 100, "version": "1.0"},
            {"authority_rank": 100, "version": "1.1", "effective_from": "2026-09-01"},
            {"authority_rank": 100, "version": "1.1", "effective_from": "2026-09-02"},
        ]
        self.assertEqual(sorted(records, key=policy.governance_sort_key, reverse=True), list(reversed(records)))

    def test_source_url_validation(self):
        for url in ["javascript:alert(1)", "not a link", "https://", "https://["]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.write({"default": {"source_url": url}})
        for url in ["", "https://example.com/document"]:
            self.write({"default": {"source_url": url}})

    def test_restoring_status_requires_explicit_config_change(self):
        self.write({"documents": {"a.txt": {"status": "inactive"}}})
        metadata = policy.metadata_for_document(Path("a.txt"), policy.load_metadata(self.root))
        self.assertFalse(policy.is_effective(metadata, self.today))
        self.write({"documents": {"a.txt": {"status": "active"}}})
        metadata = policy.metadata_for_document(Path("a.txt"), policy.load_metadata(self.root))
        self.assertTrue(policy.is_effective(metadata, self.today))


if __name__ == "__main__":
    unittest.main()
