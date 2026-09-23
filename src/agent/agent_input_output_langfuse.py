import asyncio
import os
from dataclasses import dataclass

from agents import Agent, RunContextWrapper, Runner
from agents.decorators import tool
from langfuse import get_client, propagate_attributes
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from pydantic import BaseModel

from settings import settings

MODEL = "gpt-5.4-mini"
TRACE_NAME = "user-age-openai-agent"

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_BASE_URL
os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = settings.LANGFUSE_TRACING_ENVIRONMENT

OpenAIAgentsInstrumentor().instrument()
langfuse = get_client()


@dataclass
class UserInfo:
    name: str
    uid: int


class UserAge(BaseModel):
    age: int


@tool
async def fetch_user_age(wrapper: RunContextWrapper[UserInfo]) -> str:
    """Return the user's age."""
    return f"The user {wrapper.context.name} is 47 years old"


async def main() -> None:
    user_info = UserInfo(name="John", uid=123)
    question = "What is the age of the user?"
    agent = Agent[UserInfo](
        name="Assistant",
        model=MODEL,
        tools=[fetch_user_age],
        output_type=UserAge,
    )

    with (
        langfuse.start_as_current_observation(
            name=TRACE_NAME,
            as_type="agent",
            input={"message": question, "user_id": user_info.uid},
        ) as span,
        propagate_attributes(
            trace_name=TRACE_NAME,
            user_id=str(user_info.uid),
            tags=["agent", "openai-agents"],
            metadata={"model": MODEL},
            environment=settings.LANGFUSE_TRACING_ENVIRONMENT,
        ),
    ):
        result = await Runner.run(
            starting_agent=agent,
            input=question,
            context=user_info,
        )
        span.update(output=result.final_output.model_dump())

    print(result.final_output)
    langfuse.flush()


if __name__ == "__main__":
    asyncio.run(main())
