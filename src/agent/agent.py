"""Gradio chat app for a career-focused digital twin."""

import json
from pathlib import Path
from typing import cast

import gradio as gr
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from pypdf import PdfReader

from settings import settings

MODEL = "gpt-5.4-mini"
AGENT_DIR = Path(__file__).parent
EMAIL_LOG = AGENT_DIR / "emails.txt"

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def load_pdf_text(path: Path) -> str:
    pages = (page.extract_text() for page in PdfReader(path).pages)
    return "\n".join(page for page in pages if page)


def build_system_prompt(summary: str, linkedin: str) -> str:
    return f"""
# Your role

You are a digital twin running on a website, chatting with visitors of the website.
You represent the person whose website you are on.
You answer questions related to their career, background, skills and experience.

Here are the details of the person you are representing:

{summary}

If asked, you explain clearly that you are an AI that is the digital twin of this person.

# Context

Here is a summary of the person's LinkedIn profile so that you can answer questions:

{linkedin}

# Rules

Engage with the user. Be professional and engaging, as if talking to a potential client or future employer who came across the website.
Avoid answering questions that are not related to the user's career, background, skills and experience;
steer the conversation back to professional topics.

Always stay in character as the digital twin of the person you are representing. Represent the person.

IMPORTANT: If you don't know the answer, say so. Never make up an answer.
If the user asks about something not in the context, say that you don't know.
""".strip()


# Load static career context once; every chat turn reuses the same prompt.
SYSTEM_PROMPT = build_system_prompt(
    summary=(AGENT_DIR / "summary.txt").read_text(encoding="utf-8"),
    linkedin=load_pdf_text(AGENT_DIR / "linkedin.pdf"),
)


def record_email_tool(email: str) -> str:
    email = email.strip()
    if not email:
        return "Email missing"

    print(f"Tool called to record an email: {email}")
    with EMAIL_LOG.open("a", encoding="utf-8") as file:
        file.write(email + "\n")
    return "Email received"


RECORD_EMAIL_TOOL_SCHEMA: FunctionDefinition = {
    "name": "record_email_tool",
    "description": "Use this tool to record that a user provided their email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"}
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

TOOLS: list[ChatCompletionToolParam] = [
    {"type": "function", "function": RECORD_EMAIL_TOOL_SCHEMA}
]


def ask_agent(messages: list[ChatCompletionMessageParam]) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
    )

    while response.choices[0].finish_reason == "tool_calls":
        assistant_message = response.choices[0].message
        messages.append(
            cast(
                ChatCompletionMessageParam,
                assistant_message.model_dump(exclude_none=True),
            )
        )

        # OpenAI requires one tool response message for each requested tool call.
        for tool_call in assistant_message.tool_calls or []:
            if tool_call.type != "function":
                continue

            arguments = json.loads(tool_call.function.arguments)
            content = record_email_tool(arguments.get("email", ""))
            messages.append(
                cast(
                    ChatCompletionMessageParam,
                    {
                        "role": "tool",
                        "content": content,
                        "tool_call_id": tool_call.id,
                    },
                )
            )

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )

    return response.choices[0].message.content or ""


def chat(message: str, history: list[ChatCompletionMessageParam]) -> str:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": message},
    ]
    return ask_agent(messages)


def main() -> None:
    gr.ChatInterface(chat).launch(inbrowser=True)


if __name__ == "__main__":
    main()
