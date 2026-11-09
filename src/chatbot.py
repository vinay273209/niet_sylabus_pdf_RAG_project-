from __future__ import annotations

from typing import Any

from config import Settings
from src.rag_pipeline import RAGAnswer, RAGChatbot


class ChatbotSession:
    """Conversation state for one user/session; create a separate instance per chat."""

    def __init__(
        self,
        chatbot: RAGChatbot | None = None,
        settings: Settings | None = None,
        *,
        top_k: int = 7,
    ) -> None:
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")
        self.chatbot = chatbot or RAGChatbot(settings)
        self.top_k = top_k
        self._history: list[tuple[str, str]] = []

    @property
    def history(self) -> list[tuple[str, str]]:
        return list(self._history)

    def ask(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
    ) -> RAGAnswer:
        response = self.chatbot.ask_pdf(
            question,
            history=self.history,
            filters=filters,
            top_k=self.top_k,
        )
        self._history.append((question, response.answer))
        return response

    def clear(self) -> None:
        self._history.clear()

    def close(self) -> None:
        self.chatbot.close()
