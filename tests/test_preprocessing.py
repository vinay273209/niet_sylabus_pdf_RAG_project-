from __future__ import annotations

import unittest

from src.chunking import chunk_pages
from src.metadata_extractor import extract_chunk_metadata, extract_folder_metadata
from src.qdrant_manager import point_id


class MetadataTests(unittest.TestCase):
    def test_extracts_folder_metadata_and_variant(self) -> None:
        metadata = extract_folder_metadata(
            "2026-27/CSE (IoT)/2nd Year/"
            "B.Tech CSE (IoT) Collaboration Twining Program.pdf"
        )
        self.assertEqual(metadata["academic_session"], "2026-27")
        self.assertEqual(metadata["branch"], "CSE (IoT)")
        self.assertEqual(metadata["year"], 2)
        self.assertIn("Collaboration", str(metadata["program_variant"]))

    def test_unrecognized_year_folder_is_not_guessed(self) -> None:
        metadata = extract_folder_metadata(
            "2025-26/Integrated MBA/forth-year/example.pdf"
        )
        self.assertIsNone(metadata["year"])

    def test_extracts_explicit_chunk_metadata(self) -> None:
        metadata = extract_chunk_metadata(
            "Course Code: CCSIOT0303\n"
            "Course Name: Sensors and its Applications\n"
            "Semester IV\nUnit 3"
        )
        self.assertEqual(metadata["course_code"], "CCSIOT0303")
        self.assertEqual(metadata["subject"], "Sensors and its Applications")
        self.assertEqual(metadata["semester"], 4)
        self.assertEqual(metadata["unit"], 3)

    def test_ambiguous_metadata_is_unknown(self) -> None:
        metadata = extract_chunk_metadata(
            "Semester III and Semester IV. Unit 1 and Unit 2."
        )
        self.assertIsNone(metadata["semester"])
        self.assertIsNone(metadata["unit"])


class ChunkingTests(unittest.TestCase):
    def test_chunks_preserve_page_and_overlap(self) -> None:
        text = "one two three four five six seven eight nine ten"
        chunks = chunk_pages((text, "second page"), chunk_size=20, overlap=5)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(chunk.page == 1 for chunk in chunks[:-1]))
        self.assertEqual(chunks[-1].page, 2)
        self.assertEqual(
            [chunk.chunk_index for chunk in chunks],
            list(range(len(chunks))),
        )

    def test_chunk_configuration_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            chunk_pages(("text",), chunk_size=10, overlap=10)


class PointIdTests(unittest.TestCase):
    def test_ids_are_stable_but_content_versioned(self) -> None:
        document_id = "120aab3e-a67a-5e13-8c31-5e2d6c93d18c"
        first = point_id(document_id, "a" * 64, 0)
        self.assertEqual(first, point_id(document_id, "a" * 64, 0))
        self.assertNotEqual(first, point_id(document_id, "b" * 64, 0))


if __name__ == "__main__":
    unittest.main()
