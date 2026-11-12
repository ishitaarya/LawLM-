"""Tests for cross-page legal provision assembly."""

import unittest

from src.document.legal_section_assembler import assemble_sections


class LegalSectionAssemblerTests(unittest.TestCase):
    def test_section_continuation_is_joined_across_pages(self):
        pages = [
            {
                "page_number": 10,
                "text": "It is hereby enacted as follows:\n"
                "1. Short title.—This Act may be called the Indian Contract Act.",
            },
            {"page_number": 11, "text": "It extends to the whole of India."},
        ]
        result = assemble_sections(pages, body_start_page=10)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["section_number"], "1")
        self.assertIn("whole of India", result[0]["text"])
        self.assertEqual(result[0]["source_pages"], [10, 11])
        self.assertEqual(result[0]["end_page_number"], 11)

    def test_amendment_is_kept_out_of_central_section_text(self):
        pages = [{
            "page_number": 12,
            "text": "5. Revocation of proposals and acceptances.—A proposal "
            "may be revoked.\nSTATE AMENDMENT\nUttar Pradesh\n"
            "Amendment of section 5 of Act (9 of 1872).—Substitute text.\n"
            "6. Revocation how made.—A proposal is revoked.",
        }]
        result = assemble_sections(pages, body_start_page=12)
        section_five = next(item for item in result if item["section_number"] == "5")
        self.assertIn("A proposal may be revoked", section_five["text"])
        self.assertNotIn("Uttar Pradesh", section_five["text"])
        self.assertTrue(any("Uttar Pradesh" in item["text"] for item in section_five["amendments"]))
        self.assertEqual(result[1]["section_number"], "6")

    def test_footnotes_are_not_added_to_provision_body(self):
        pages = [{
            "page_number": 10,
            "text": "1. Short title.—This Act may be called the Act.\n"
            "1. For the Statement of Objects and Reasons, see Gazette.",
        }]
        result = assemble_sections(pages, body_start_page=10)
        self.assertEqual(len(result), 1)
        self.assertNotIn("Statement of Objects", result[0]["text"])
        self.assertEqual(len(result[0]["footnotes"]), 1)

    def test_repeal_status_requires_explicit_verified_input(self):
        pages = [{"page_number": 20, "text": "76. Repealed provision.—Old text."}]
        default = assemble_sections(pages, body_start_page=20)
        marked = assemble_sections(pages, body_start_page=20, repealed_sections={"76"})
        self.assertEqual(default[0]["section_status"], "unknown")
        self.assertEqual(marked[0]["section_status"], "repealed")

    def test_fused_footnote_section_number_is_preserved_for_review(self):
        pages = [{
            "page_number": 44,
            "text": "150. Previous provision.—Existing text.\\n"
            "1151. Special provision.—Text belonging to possible Section 151.",
        }]
        result = assemble_sections(pages, body_start_page=44)
        self.assertEqual(len(result), 1)
        self.assertNotIn("1151.", result[0]["text"])
        self.assertEqual(len(result[0]["ocr_anomalies"]), 1)
        self.assertIn("manual review", result[0]["ocr_anomalies"][0]["reason"])

    def test_front_matter_is_skipped_only_when_body_start_is_known(self):
        pages = [
            {"page_number": 1, "text": "1. Contents.—Section list."},
            {"page_number": 10, "text": "1. Short title.—Actual provision."},
        ]
        result = assemble_sections(pages, body_start_page=10)
        self.assertEqual(len(result), 1)
        self.assertIn("Actual provision", result[0]["text"])


if __name__ == "__main__":
    unittest.main()
