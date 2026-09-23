"""Plan web searches, summarize them, and turn the findings into a report."""

import asyncio
import os
import sys
from pathlib import Path

from agents import Agent, Runner, WebSearchTool, function_tool, trace
from agents.model_settings import ModelSettings
from pydantic import BaseModel, Field

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import settings

MODEL = "gpt-5.4-mini"
SEARCH_COUNT = 5
DEFAULT_QUERY = "Most popular AI Agent frameworks in 2026"

if settings.OPENAI_API_KEY:
    os.environ.setdefault("OPENAI_API_KEY", settings.OPENAI_API_KEY)


class WebSearchItem(BaseModel):
    query: str = Field(description="The search term to use.")
    reason: str = Field(description="Why this search helps answer the query.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(
        description="Web searches to perform before writing the report."
    )


class ReportData(BaseModel):
    short_summary: str = Field(description="A 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final markdown report.")
    follow_up_questions: list[str] = Field(description="Topics worth researching next.")


PLANNER_INSTRUCTIONS = f"""
You are a research assistant. Given a user query, propose {SEARCH_COUNT} web searches
that will produce enough evidence to answer it.
"""

SEARCH_INSTRUCTIONS = """
You are a research assistant. Given one search term, search the web for that term and
produce a concise 2-3 paragraph summary under 300 words.
Capture the main points and reply only with the summary.
"""

WRITER_INSTRUCTIONS = """
You are a senior researcher writing a cohesive report.
You will receive the original query and summarized web research.
Write a detailed markdown report, at least 1000 words.
"""

EMAIL_INSTRUCTIONS = """
You are provided with a detailed report. Use your tool to send an email.
Convert the report into a clean HTML email with an appropriate subject line.
"""


PLANNER_AGENT = Agent(
    name="Planner Agent",
    instructions=PLANNER_INSTRUCTIONS,
    model=MODEL,
    output_type=WebSearchPlan,
)

SEARCH_AGENT = Agent(
    name="Search Agent",
    instructions=SEARCH_INSTRUCTIONS,
    tools=[WebSearchTool()],
    model=MODEL,
    # Force live research instead of letting the model answer from memory.
    model_settings=ModelSettings(tool_choice="required"),
)

WRITER_AGENT = Agent(
    name="Writer Agent",
    instructions=WRITER_INSTRUCTIONS,
    model=MODEL,
    output_type=ReportData,
)


@function_tool
def send_email_tool(subject: str, text_body: str, html_body: str) -> str:
    """Print the generated email until a real email provider is connected."""
    print(subject)
    print(text_body)
    return "Email sent successfully"


EMAIL_AGENT = Agent(
    name="Email Agent",
    instructions=EMAIL_INSTRUCTIONS,
    tools=[send_email_tool],
    model=MODEL,
)


async def plan_searches(query: str) -> list[WebSearchItem]:
    print("Planning searches...")
    result = await Runner.run(PLANNER_AGENT, f"Query: {query}")
    print("Searches: ", *result.final_output.searches, sep="\n")
    return result.final_output.searches


async def summarize_search(item: WebSearchItem) -> str:
    input_message = f"Search term: {item.query}\nReason for searching: {item.reason}"
    print(input_message)
    result = await Runner.run(SEARCH_AGENT, input_message)
    print(result.final_output)
    return str(result.final_output)


async def run_searches(query: str) -> list[str]:
    print(f"Searching: {query}")
    searches = await plan_searches(query)
    print("\n")

    print(f"Running {len(searches)} searches...")
    results = await asyncio.gather(*(summarize_search(item) for item in searches))
    print("Finished searching", list(results))
    return list(results)


async def write_report(query: str, search_results: list[str]) -> ReportData:
    print("Writing report...")
    research = "\n\n".join(search_results)
    input_message = f"Original query: {query}\n\nSummarized search results:\n{research}"
    result = await Runner.run(WRITER_AGENT, input_message)
    print("Finished writing report")
    return result.final_output


async def send_report_email(report: ReportData) -> str:
    print("Writing email...")
    result = await Runner.run(EMAIL_AGENT, report.markdown_report)
    print("Email sent")
    return str(result.final_output)


async def main(query: str = DEFAULT_QUERY) -> ReportData:
    with trace("Research trace", metadata={"model": MODEL, "query": query}):
        print("Starting research...")
        search_results = await run_searches(query)
        report = await write_report(query, search_results)
        await send_report_email(report)
        return report


if __name__ == "__main__":
    asyncio.run(main())
