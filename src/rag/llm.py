import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)

client = OpenAI(
    api_key=os.environ["VIRTUAL_KEY"],
    base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4000"),
)


def completion(model, messages, response_format=None):
    kwargs = {}
    if response_format is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "schema": response_format.model_json_schema(),
            },
        }
    return client.chat.completions.create(model=model, messages=messages, **kwargs)


def embeddings(model, input):
    return client.embeddings.create(model=model, input=input)
