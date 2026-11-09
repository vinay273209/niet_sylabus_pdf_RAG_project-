from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import gradio as gr

from app import _clear_chat, _submit, build_app
from src.rag_pipeline import GeminiDailyQuotaError, RAGAnswer


class AppTests(unittest.TestCase):
    def test_build_app_supports_installed_gradio_chatbot_api(self) -> None:
        demo = build_app()

        self.assertIsInstance(demo, gr.Blocks)
        self.assertFalse(
            any(isinstance(block, gr.Dropdown) for block in demo.blocks.values())
        )

    def test_submit_keeps_chat_history_and_searches_without_filters(self) -> None:
        session = Mock()
        session.ask.return_value = RAGAnswer(
            answer="Check [S1].",
            sources=[],
            filters={"branch": "CSE"},
            retrieved_count=1,
        )
        history = [{"role": "user", "content": "Earlier question"}]
        result = _submit(
            "Explain Unit 3",
            history,
            session,
        )
        self.assertEqual(result[0], "")
        self.assertEqual(len(result[1]), 3)
        self.assertEqual(result[1][-1]["content"], "Check [S1].")
        session.ask.assert_called_once_with("Explain Unit 3")
        self.assertIs(result[2], session)

    def test_submit_reports_user_facing_error(self) -> None:
        session = Mock()
        session.ask.side_effect = RuntimeError("private diagnostic detail")
        with patch("app.logger.exception"):
            result = _submit(
                "Question",
                [],
                session,
            )
        self.assertIn("couldn't complete", result[1][-1]["content"])
        self.assertNotIn("private diagnostic detail", result[1][-1]["content"])

    def test_submit_shows_actionable_gemini_quota_error(self) -> None:
        session = Mock()
        session.ask.side_effect = GeminiDailyQuotaError(
            "The Gemini API daily request quota has been reached."
        )

        result = _submit("Question", [], session)

        self.assertIn("daily request quota", result[1][-1]["content"])

    def test_clear_resets_messages_and_conversation_state(self) -> None:
        self.assertEqual(_clear_chat(), ([], None))


if __name__ == "__main__":
    unittest.main()
