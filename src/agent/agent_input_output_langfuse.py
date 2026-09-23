import asyncio
import os
from dataclasses import dataclass
from importlib import import_module

from agents import Agent, RunContextWrapper, Runner
from agents.decorators import tool
from dotenv import load_dotenv
from pydantic import BaseModel

MODEL = "gpt-5.4-mini"
TRACE_NAME = "user-age-openai-agent"

load_dotenv(override=True)
if os.getenv("OPENAI_API_KEY"):
    os.environ.setdefault("OPENAI_API_KEY", os.environ["OPENAI_API_KEY"])
if os.getenv("LANGFUSE_BASE_URL"):
    os.environ.setdefault("LANGFUSE_HOST", os.environ["LANGFUSE_BASE_URL"])
os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "development")

OpenAIAgentsInstrumentor = import_module(
    "openinference.instrumentation.openai_agents"
).OpenAIAgentsInstrumentor
OpenAIAgentsInstrumentor().instrument()
langfuse = import_module("langfuse").get_client()


@dataclass
class UserInfo:
    name: str
    uid: int


class UserAge(BaseModel):
    age: int


@tool
async def fetch_user_age(wrapper: RunContextWrapper[UserInfo]) -> str:
    """Fetch the age of the user. Call this function to get user's age information."""
    return f"The user {wrapper.context.name} is 47 years old"


async def main() -> None:
    user_info = UserInfo(name="John", uid=123)
    question = "What is the age of the user?"
    agent = Agent[UserInfo](
        name="Assistant", model=MODEL, tools=[fetch_user_age], output_type=UserAge
    )

    with (
        langfuse.start_as_current_observation(
            name=TRACE_NAME,
            as_type="agent",
            input={"message": question, "user_id": user_info.uid},
        ) as span,
        import_module("langfuse").propagate_attributes(
            trace_name=TRACE_NAME,
            user_id=str(user_info.uid),
            tags=["agent", "openai-agents"],
            metadata={"model": MODEL},
            environment=os.environ["LANGFUSE_TRACING_ENVIRONMENT"],
        ),
    ):
        result = await Runner.run(
            starting_agent=agent,
            input=question,
            context=user_info,
        )
        span.update(output=result.final_output.model_dump())

    print(result)
    langfuse.flush()


if __name__ == "__main__":
    asyncio.run(main())
