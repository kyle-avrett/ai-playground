import asyncio
import os
from dataclasses import dataclass

from agents import Agent, RunContextWrapper, Runner
from agents.decorators import tool
from pydantic import BaseModel

from settings import settings

MODEL = "gpt-5.4-mini"
if settings.OPENAI_API_KEY:
    os.environ.setdefault("OPENAI_API_KEY", settings.OPENAI_API_KEY)

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

async def main():
    user_info = UserInfo(name="John", uid=123)

    agent = Agent[UserInfo](
        name="Assistant",
        model=MODEL,
        tools=[fetch_user_age],
        output_type=UserAge
    )

    result = await Runner.run(
        starting_agent=agent,
        input="What is the age of the user?",
        context=user_info
    )

    print(result)

if __name__ == "__main__":
    asyncio.run(main())
