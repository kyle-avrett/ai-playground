import asyncio
import os
from dataclasses import dataclass

import mlflow
from agents import Agent, RunContextWrapper, Runner
from agents.decorators import tool
from mlflow.entities import SpanType
from pydantic import BaseModel

from settings import settings

MODEL = "gpt-5.4-mini"
EXPERIMENT_NAME = "my-experiment"
QUESTION = "What is the age of the user?"

if settings.OPENAI_API_KEY:
    os.environ.setdefault("OPENAI_API_KEY", settings.OPENAI_API_KEY)
if not settings.MLFLOW_TRACKING_URI:
    raise RuntimeError("Set MLFLOW_TRACKING_URI before running this script.")

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.openai.autolog()


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

    agent = Agent[UserInfo](
        name="Assistant",
        model=MODEL,
        tools=[fetch_user_age],
        output_type=UserAge,
    )

    with mlflow.start_span(name="user_age_agent", span_type=SpanType.AGENT) as span:
        span.set_inputs({"message": QUESTION, "user_id": user_info.uid})
        result = await Runner.run(
            starting_agent=agent,
            input=QUESTION,
            context=user_info,
        )
        span.set_outputs(result.final_output.model_dump())

    print(result.final_output)
    mlflow.flush_trace_async_logging()

if __name__ == "__main__":
    asyncio.run(main())
