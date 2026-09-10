import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

from settings import settings

# globals

COMPANY_NAME = "Kyle Avrett"
URL = "https://kyleavrett.com"
MODEL = "gpt-5-nano"
BROCHURE_MODEL = "gpt-4.1-mini"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

link_system_prompt = """
You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include in a brochure about the company,
such as links to an About page, or a Company page, or Careers/Jobs pages.
You should respond in JSON as in this example:

{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""

brochure_system_prompt = """
You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
"""


def fetch_website_contents(url):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    title = soup.title.string if soup.title else "No title found"
    if soup.body:
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        text = soup.body.get_text(separator="\n", strip=True)
    else:
        text = ""
    return (title + "\n\n" + text)[:2_000]  # ty: ignore[unsupported-operator]


def fetch_website_links(url):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    links = [link.get("href") for link in soup.find_all("a")]
    return [link for link in links if link]


def get_links_user_prompt(url):
    user_prompt = f"""
Here is the list of links on the website {url} -
Please decide which of these are relevant web links for a brochure about the company,
respond with the full https URL in JSON format.
Do not include Terms of Service, Privacy, email links.

Links (some might be relative links):

"""
    user_prompt += "\n".join(fetch_website_links(url))
    return user_prompt


def select_relevant_links(url, openai):
    messages=[
        {"role": "system", "content": link_system_prompt},
        {"role": "user", "content": get_links_user_prompt(url)}
    ]

    print('select_relevant_links', '::', 'messages')
    print(f"Selecting relevant links for {url} by calling {MODEL}")
    print(json.dumps(messages, indent=4, sort_keys=True))

    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"}
    )
    response = response.choices[0].message.content
    relevant_links = json.loads(response)
    print(f"Found {len(relevant_links['links'])} relevant links")
    print(json.dumps(response, indent=4, sort_keys=True))
    return relevant_links


def fetch_page_and_all_relevant_links(url, openai):
    result = f"## Landing Page:\n\n{fetch_website_contents(url)}\n## Relevant Links:\n"
    relevant_links = select_relevant_links(url, openai)
    for link in relevant_links["links"]:
        result += f"\n\n### Link: {link['type']}\n"
        result += fetch_website_contents(link["url"])
    return result


def get_brochure_user_prompt(company_name, url, openai):
    user_prompt = f"""
You are looking at a company called: {company_name}
Here are the contents of its landing page and other relevant pages;
use this information to build a short brochure of the company in markdown without code blocks.\n\n
"""
    user_prompt += fetch_page_and_all_relevant_links(url, openai)
    return user_prompt[:5_000]


def create_brochure(company_name, url):
    openai = OpenAI(api_key=settings.API_KEY)

    messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url, openai)}
        ]
    print('create_brochure', '::', 'messages')
    print(json.dumps(messages, indent=4, sort_keys=True))

    response = openai.chat.completions.create(
        model=BROCHURE_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content

def next_brochure_path(company_name):
    directory = Path(__file__).parent
    prefix = f"{company_name}."
    used = {
        int(path.name[len(prefix):-3])
        for path in directory.iterdir()
        if path.name.startswith(prefix) and path.suffix == ".md" and path.name[len(prefix):-3].isdigit()
    }
    number = 1
    while number in used:
        number += 1
    return directory / f"{company_name}.{number}.md"


def main():
    result = create_brochure(COMPANY_NAME, URL)
    next_brochure_path(COMPANY_NAME).write_text(str(result), encoding="utf-8")

if __name__ == '__main__':
   main()
