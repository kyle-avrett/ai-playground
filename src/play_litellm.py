import os

from litellm import completion
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.API_KEY

response = completion(
    model="openai/gpt-5.6-terra",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response.choices[0].message.content)
