import os

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI


# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)


async def main():
    # Run the agent
    response = await agent.run("What is 1234 * 4567?")
    print(str(response))

    from llama_index.core.workflow import Context
    ctx = Context(agent)
    response = await agent.run("My name is Logan", ctx=ctx)
    print('response 1:', str(response))
    response = await agent.run("What is my name?", ctx=ctx)
    print('response 2:', str(response))


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
