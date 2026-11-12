"""Unit tests for the conservative legal document structure parser."""

import unittest

from src.document.legal_parser import (
    analyze_page,
    detect_amendment_markers,
    detect_chapter_headings,
    detect_section_candidates,
    find_body_start_page,
)


class LegalParserTests(unittest.TestCase):
    def test_body_start_is_detected_from_enactment_text(self):
        pages = [
            {"page_number": 1, "text": "ARRANGEMENT OF SECTIONS"},
            {
                "page_number": 10,
                "text": "PREAMBLE—WHEREAS it is expedient.\n"
                "It is hereby enacted as follows:—",
            },
        ]
        self.assertEqual(find_body_start_page(pages), 10)

    def test_front_matter_is_not_classified_using_fixed_page_limit(self):
        record = {"page_number": 9, "text": "ARRANGEMENT OF SECTIONS"}
        result = analyze_page(record, body_start_page=10)
        self.assertEqual(result.page_type, "table_of_contents_or_front_matter")

    def test_missing_body_marker_is_not_assumed_to_be_contents(self):
        record = {"page_number": 1, "text": "Unknown legal document"}
        result = analyze_page(record, body_start_page=None)
        self.assertEqual(result.page_type, "unclassified")

    def test_chapter_heading_is_detected(self):
        self.assertEqual(
            detect_chapter_headings("CHAPTER I\nOF CONTRACTS\nCHAPTER IV"),
            ["CHAPTER I", "CHAPTER IV"],
        )

    def test_section_heading_and_alphanumeric_number_are_preserved(self):
        text = (
            "19A. Power to set aside contract induced by undue influence.—"
            "The court may set aside."
        )
        found = detect_section_candidates(text, page_number=15)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["section_number"], "19A")
        self.assertEqual(
            found[0]["heading_text"],
            "Power to set aside contract induced by undue influence",
        )
        self.assertEqual(found[0]["page_number"], 15)

    def test_three_digit_alphanumeric_section_is_preserved(self):
        found = detect_section_candidates(
            "178A. Special provision.—Text of the provision.",
            page_number=42,
        )
        self.assertEqual(found[0]["section_number"], "178A")

    def test_footnote_is_not_mistaken_for_section_heading(self):
        text = (
            "1. For the Statement of Objects and Reasons, see Gazette of India.\n"
            "2. Interpretation-clause.—In this Act..."
        )
        found = detect_section_candidates(text, page_number=10)
        self.assertEqual([item["section_number"] for item in found], ["2"])

    def test_observed_indian_contract_act_heading_formats(self):
        examples = [
            ("1. Short title.—This Act may be called the Act.", "1"),
            ("2. Interpretation-clause.—In this Act...", "2"),
            ("3.Communication, acceptance and revocation of proposals. —The communication...", "3"),
            ("4. Communication when complete .—The communication...", "4"),
            ("5. Revocation of proposals and acceptances .—A proposal...", "5"),
            ("8. Acceptance by performing conditions, or receiving consideration .—Performance...", "8"),
            ("19A. Power to set aside contract.—Text.", "19A"),
            ("178A. Special provision.—Text.", "178A"),
        ]
        for line, expected_number in examples:
            with self.subTest(line=line):
                found = detect_section_candidates(line, page_number=10)
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0]["section_number"], expected_number)

    def test_ocr_footnote_prefix_is_not_silently_repaired(self):
        # In the scanned Act, 151/161 may appear as 1151./1161. due to
        # a footnote marker. These must be flagged for review, not guessed.
        self.assertEqual(
            detect_section_candidates("1151. Special provision.—Text.", 44),
            [],
        )
        self.assertEqual(
            detect_section_candidates("1161. Special provision.—Text.", 45),
            [],
        )

    def test_amendment_markers_are_detected(self):
        text = (
            "STATE AMENDMENT\n"
            "Uttar Pradesh\n"
            "Amendment of section 5 of Act (9 of 1872).—..."
        )
        markers = detect_amendment_markers(text)
        self.assertEqual(
            markers,
            [
                "STATE AMENDMENT",
                "Amendment of section 5 of Act (9 of 1872).—...",
            ],
        )


if __name__ == "__main__":
    unittest.main()
