from __future__ import annotations

import inspect
import logging

import gradio as gr

from src.chatbot import ChatbotSession
from src.rag_pipeline import GeminiDailyQuotaError

logger = logging.getLogger("syllabus_app")

APP_CSS = """
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
}
#app-shell {
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    height: 100vh;
    margin: 0 auto;
    max-width: 920px;
    padding: 18px 24px 16px;
}
#app-header {
    border-bottom: 1px solid var(--border-color-primary);
    padding-bottom: 12px;
}
#chat-window {
    flex: 1 1 auto;
    min-height: 0;
}
#chat-window .message {
    border-radius: 18px;
}
#composer {
    border: 1px solid var(--border-color-primary);
    border-radius: 22px;
    padding: 10px;
}
#composer textarea {
    border: 0 !important;
    box-shadow: none !important;
}
#send-button {
    align-self: end;
}
@media (max-width: 640px) {
    #app-shell {
        height: 100dvh;
        padding: 12px;
    }
}
"""


def _submit(
    question: str,
    history: list[dict[str, str]] | None,
    session: ChatbotSession | None,
) -> tuple[str, list[dict[str, str]], ChatbotSession | None]:
    if not question or not question.strip():
        return "", history or [], session
    try:
        if session is None:
            session = ChatbotSession()
        response = session.ask(question.strip())
        next_history = list(history or [])
        next_history.extend(
            [
                {"role": "user", "content": question.strip()},
                {"role": "assistant", "content": response.answer},
            ]
        )
        return "", next_history, session
    except GeminiDailyQuotaError as exc:
        next_history = list(history or [])
        next_history.extend(
            [
                {"role": "user", "content": question.strip()},
                {"role": "assistant", "content": str(exc)},
            ]
        )
        return "", next_history, session
    except Exception as exc:
        logger.exception("Chat request failed (%s).", type(exc).__name__)
        message = (
            "I couldn't complete that request. Check the API/database configuration "
            "or try again."
        )
        next_history = list(history or [])
        next_history.extend(
            [
                {"role": "user", "content": question.strip()},
                {"role": "assistant", "content": message},
            ]
        )
        return "", next_history, session


def _clear_chat() -> tuple[list[dict[str, str]], ChatbotSession | None]:
    return [], None


def build_app() -> gr.Blocks:
    blocks_options = {"title": "Smart Academic Syllabus Assistant"}
    if int(gr.__version__.split(".", maxsplit=1)[0]) < 6:
        blocks_options.update(theme=gr.themes.Soft(), css=APP_CSS)
    with gr.Blocks(**blocks_options) as demo:
        with gr.Column(elem_id="app-shell"):
            with gr.Row(elem_id="app-header"):
                gr.Markdown(
                    "### Smart Academic Syllabus Assistant\n"
                    "Ask about any branch, year, semester, or syllabus version.",
                    scale=8,
                )
                clear = gr.Button("New chat", scale=1, min_width=100)

            chatbot_options = {
                "label": "Conversation",
                "height": "65vh",
                "placeholder": (
                    "Ask about subjects, units, or compare syllabus versions."
                ),
                "show_label": False,
            }
            chatbot_parameters = inspect.signature(gr.Chatbot).parameters
            if "type" in chatbot_parameters:
                chatbot_options["type"] = "messages"
            if "buttons" in chatbot_parameters:
                chatbot_options["buttons"] = ["copy"]
            else:
                chatbot_options["show_copy_button"] = True
            chatbot = gr.Chatbot(**chatbot_options, elem_id="chat-window")
            chat_state = gr.State(value=None)

            with gr.Row(elem_id="composer"):
                question = gr.Textbox(
                    label="Message",
                    placeholder="Message the syllabus assistant...",
                    lines=1,
                    max_lines=6,
                    show_label=False,
                    scale=8,
                    autofocus=True,
                )
                submit = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    min_width=78,
                    elem_id="send-button",
                )

        inputs = [question, chatbot, chat_state]
        outputs = [question, chatbot, chat_state]
        submit.click(_submit, inputs=inputs, outputs=outputs)
        question.submit(_submit, inputs=inputs, outputs=outputs)
        clear.click(_clear_chat, outputs=[chatbot, chat_state])

    return demo


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app = build_app()
    if int(gr.__version__.split(".", maxsplit=1)[0]) >= 6:
        app.launch(theme=gr.themes.Soft(), css=APP_CSS)
    else:
        app.launch()
