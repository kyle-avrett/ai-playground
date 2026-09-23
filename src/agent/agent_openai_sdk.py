"""Gradio chat app for a career-focused digital twin."""

import os
import sys
from pathlib import Path
from typing import Any, cast

import gradio as gr
from agents import Agent, Runner, function_tool, trace
from agents.items import TResponseInputItem

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.system_prompt import SYSTEM_PROMPT
from settings import settings

MODEL = "gpt-5.4-mini"
AGENT_DIR = Path(__file__).parent
EMAIL_LOG = AGENT_DIR / "emails.txt"

if settings.OPENAI_API_KEY:
    os.environ.setdefault("OPENAI_API_KEY", settings.OPENAI_API_KEY)


@function_tool(
    description_override="Use this tool to record that a user provided their email address"
)
def record_email_tool(email: str) -> str:
    email = email.strip()
    if not email:
        return "Email missing"

    print(f"Tool called to record an email: {email}")
    with EMAIL_LOG.open("a", encoding="utf-8") as file:
        file.write(email + "\n")
    return "Email received"


CAREER_AGENT = Agent(
    name="Career digital twin",
    instructions=SYSTEM_PROMPT,
    model=MODEL,
    tools=[record_email_tool],
)


def ask_agent(messages: list[TResponseInputItem]) -> str:
    with trace("Career digital twin chat", metadata={"model": MODEL}):
        result = Runner.run_sync(CAREER_AGENT, messages)
    return str(result.final_output or "")


def message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return str(content)


def chat(message: str, history: list[dict[str, Any]]) -> str:
    messages = [
        cast(
            TResponseInputItem,
            {"role": item["role"], "content": message_text(item.get("content", ""))},
        )
        for item in history
        if item.get("role") in {"user", "assistant", "system"}
    ]
    messages.append(cast(TResponseInputItem, {"role": "user", "content": message}))
    return ask_agent(messages)


def main() -> None:
    gr.ChatInterface(chat).launch(inbrowser=True)


if __name__ == "__main__":
    main()
