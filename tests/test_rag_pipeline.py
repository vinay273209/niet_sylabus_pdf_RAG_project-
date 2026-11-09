from __future__ import annotations

import unittest
from unittest.mock import Mock
from unittest.mock import patch

from google.genai.errors import APIError
from qdrant_client import models

from config import Settings
from src.chatbot import ChatbotSession
from src.rag_pipeline import (
    GeminiDailyQuotaError,
    RAGChatbot,
    _generate_with_retry,
    _infer_query_filters,
    _retrieval_query,
    format_citations,
)
from src.retriever import PdfRetriever, SearchResult, build_qdrant_filter


class FilterTests(unittest.TestCase):
    def test_builds_exact_and_multi_value_filters(self) -> None:
        query_filter = build_qdrant_filter(
            {"academic_session": ["2025-26", "2026-27"], "year": 2}
        )
        self.assertIsInstance(query_filter, models.Filter)
        self.assertEqual(len(query_filter.must), 2)
        self.assertEqual(query_filter.must[0].key, "academic_session")
        self.assertIsInstance(query_filter.must[0].match, models.MatchAny)
        self.assertEqual(query_filter.must[1].match.value, 2)

    def test_rejects_unknown_and_invalid_filters(self) -> None:
        with self.assertRaises(ValueError):
            build_qdrant_filter({"unknown": "value"})
        with self.assertRaises(ValueError):
            build_qdrant_filter({"year": "second"})


class QueryParsingTests(unittest.TestCase):
    def test_sensor_course_query_uses_course_and_explicit_scope(self) -> None:
        question = (
            "Give me the unit-wise syllabus topics for Sensors and its Applications, "
            "CSE (IoT), Semester 3, 2026-27. Include every topic from the PDF."
        )

        self.assertEqual(
            _retrieval_query(question, None),
            "Sensors and its Applications CSE IoT",
        )
        self.assertEqual(
            _infer_query_filters(question),
            {"branch": "CSE (IoT)", "academic_session": "2026-27"},
        )

    def test_unit_follow_up_uses_prior_user_context(self) -> None:
        history = [("CSE IoT Sensors and its Applications syllabus", "prior answer")]

        query = _retrieval_query("Explain Unit 3", history)

        self.assertIn("CSE IoT Sensors and its Applications syllabus", query)
        self.assertIn("Explain Unit 3", query)


class RetrieverTests(unittest.TestCase):
    def test_search_uses_filter_and_does_not_fetch_vectors(self) -> None:
        client = Mock()
        point = Mock()
        point.payload = {
            "text": "Course content",
            "source": "CSE.pdf",
            "page": 7,
            "branch": "CSE",
            "academic_session": "2026-27",
            "year": 2,
            "semester": 4,
            "subject": "Data Structures",
            "course_code": "CCSE0401",
            "program_variant": None,
        }
        point.score = 0.9
        client.query_points.return_value.points = [point]
        embedder = Mock()
        embedder.encode.return_value = [[0.1] * 384]
        settings = Settings(
            qdrant_url="https://example.invalid",
            qdrant_api_key="not-a-secret",
        )
        retriever = PdfRetriever(settings, client=client, embedder=embedder)

        results = retriever.search_pdf(
            "Explain databases",
            filters={"branch": "CSE", "year": 2},
            top_k=5,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "CSE.pdf")
        self.assertEqual(results[0].page, 7)
        kwargs = client.query_points.call_args.kwargs
        self.assertEqual(kwargs["limit"], 5)
        self.assertFalse(kwargs["with_vectors"])
        self.assertEqual(len(kwargs["query_filter"].must), 2)

    def test_page_chunks_are_restored_in_document_order(self) -> None:
        client = Mock()
        points = []
        for chunk_index in (2, 1):
            point = Mock()
            point.payload = {
                "text": f"chunk {chunk_index}",
                "document_id": "document-1",
                "chunk_index": chunk_index,
                "source": "CSE.pdf",
                "page": 7,
            }
            point.score = 0.9
            points.append(point)
        client.scroll.return_value = (points, None)
        retriever = PdfRetriever(Settings(), client=client, embedder=Mock())
        result = SearchResult(
            text="matching chunk",
            score=0.9,
            source="CSE.pdf",
            page=7,
            academic_session="2026-27",
            branch="CSE",
            year=2,
            semester=None,
            subject=None,
            course_code="CCSE0201",
            program_variant=None,
            document_id="document-1",
            chunk_index=1,
        )

        chunks = retriever.get_page_chunks(result)

        self.assertEqual([chunk.text for chunk in chunks], ["chunk 1", "chunk 2"])


class CitationTests(unittest.TestCase):
    def test_citations_include_filename_and_page(self) -> None:
        result = SearchResult(
            text="Supported evidence",
            score=0.9,
            source="CSE.pdf",
            page=7,
            academic_session="2026-27",
            branch="CSE",
            year=2,
            semester=4,
            subject="Data Structures",
            course_code="CCSE0401",
            program_variant=None,
        )

        answer, sources = format_citations("The answer is supported [S1].", [result])

        self.assertIn("CSE.pdf, page 7", answer)
        self.assertEqual(sources[0]["source_id"], "S1")
        self.assertEqual(sources[0]["page"], 7)

    def test_grouped_citations_keep_every_referenced_source(self) -> None:
        results = [
            SearchResult(
                text=f"Evidence {index}",
                score=0.9,
                source=f"source-{index}.pdf",
                page=index,
                academic_session="2026-27",
                branch="CSE",
                year=2,
                semester=4,
                subject=None,
                course_code=None,
                program_variant=None,
            )
            for index in range(1, 4)
        ]

        answer, sources = format_citations("Claim [S1, S3].", results)

        self.assertIn("source-1.pdf, page 1", answer)
        self.assertIn("source-3.pdf, page 3", answer)
        self.assertEqual(
            [source["source_id"] for source in sources],
            ["S1", "S3"],
        )


class ChatbotTests(unittest.TestCase):
    def test_every_answer_retrieves_and_uses_history_only_as_context(self) -> None:
        retriever = Mock()
        retriever.search_pdf.return_value = [
            SearchResult(
                text="Unit 3 covers normalization.",
                score=0.9,
                source="CSE.pdf",
                page=7,
                academic_session="2026-27",
                branch="CSE",
                year=2,
                semester=4,
                subject="Database Management Systems",
                course_code="CCSE0401",
                program_variant=None,
            )
        ]
        gemini = Mock()
        gemini.models.generate_content.return_value.text = (
            "Unit 3 covers normalization [S1]."
        )
        settings = Settings(
            gemini_api_key="not-a-secret",
            qdrant_url="https://example.invalid",
            qdrant_api_key="not-a-secret",
        )
        chatbot = RAGChatbot(settings, retriever=retriever, gemini_client=gemini)
        history = [
            ("Tell me about second-year CSE.", "The subject is DBMS."),
        ]

        response = chatbot.ask_pdf(
            "Explain Unit 3",
            history=history,
            filters={"branch": "CSE"},
        )

        self.assertEqual(response.retrieved_count, 1)
        self.assertIn("CSE.pdf, page 7", response.answer)
        retriever.search_pdf.assert_called_once()
        retrieval_query = retriever.search_pdf.call_args.args[0]
        self.assertIn("Explain Unit 3", retrieval_query)
        self.assertIn("second-year CSE", retrieval_query)
        prompt = gemini.models.generate_content.call_args.kwargs["contents"]
        self.assertIn("Unit 3 covers normalization.", prompt)
        self.assertIn(
            "CONVERSATION HISTORY (reference resolution only; not evidence)",
            prompt,
        )

    def test_no_retrieved_evidence_skips_gemini(self) -> None:
        retriever = Mock()
        retriever.search_pdf.return_value = []
        gemini = Mock()
        chatbot = RAGChatbot(
            Settings(),
            retriever=retriever,
            gemini_client=gemini,
        )

        response = chatbot.ask_pdf("A question")

        self.assertEqual(response.retrieved_count, 0)
        self.assertEqual(response.sources, [])
        gemini.models.generate_content.assert_not_called()

    def test_temporary_gemini_error_retries_with_bounded_backoff(self) -> None:
        gemini = Mock()
        gemini.models.generate_content.side_effect = [
            APIError(503, {}),
            Mock(text="Grounded answer [S1]."),
        ]
        settings = Settings(gemini_model="gemini-test")

        with patch("src.rag_pipeline.time.sleep") as sleep:
            answer = _generate_with_retry(gemini, settings, "prompt")

        self.assertEqual(answer, "Grounded answer [S1].")
        self.assertEqual(gemini.models.generate_content.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_daily_gemini_quota_error_is_not_retried(self) -> None:
        gemini = Mock()
        gemini.models.generate_content.side_effect = APIError(
            429,
            {
                "error": {
                    "message": (
                        "Generate content requests per day limit exceeded on "
                        "Free Tier."
                    ),
                    "status": "RESOURCE_EXHAUSTED",
                }
            },
        )

        with patch("src.rag_pipeline.time.sleep") as sleep:
            with self.assertRaisesRegex(
                GeminiDailyQuotaError,
                "daily request quota",
            ):
                _generate_with_retry(gemini, Settings(), "prompt")

        gemini.models.generate_content.assert_called_once()
        sleep.assert_not_called()

    def test_empty_gemini_response_is_reported(self) -> None:
        gemini = Mock()
        gemini.models.generate_content.return_value.text = None

        with self.assertRaisesRegex(RuntimeError, "empty answer"):
            _generate_with_retry(gemini, Settings(), "prompt")

    def test_chat_session_keeps_and_clears_its_own_history(self) -> None:
        chatbot = Mock()
        chatbot.ask_pdf.return_value.answer = "Evidence-backed reply [S1]."
        session = ChatbotSession(chatbot=chatbot)

        session.ask("Tell me about CSE IoT second year.")
        session.ask("Explain Unit 3.")

        self.assertEqual(len(session.history), 2)
        self.assertEqual(
            chatbot.ask_pdf.call_args_list[1].kwargs["history"],
            [("Tell me about CSE IoT second year.", "Evidence-backed reply [S1].")],
        )
        session.clear()
        self.assertEqual(session.history, [])


if __name__ == "__main__":
    unittest.main()
