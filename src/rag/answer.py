from pathlib import Path

from chromadb import PersistentClient
from llm import completion, embeddings
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential

BASE_DIR = Path(__file__).parent
DB_PATH = str(BASE_DIR / "database")
COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "text-embedding-3-large"
MODEL = "openai/gpt-4.1-nano"
RETRY_WAIT = wait_exponential(multiplier=1, min=10, max=240)
RETRIEVAL_K = 20
FINAL_K = 10

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""

chroma = PersistentClient(path=DB_PATH)
collection = chroma.get_or_create_collection(COLLECTION_NAME)


class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The chunk ids ordered from most relevant to least relevant."
    )


@retry(wait=RETRY_WAIT)
def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    chunk_text = "\n\n".join(
        f"# CHUNK ID: {index}:\n\n{chunk.page_content}"
        for index, chunk in enumerate(chunks, start=1)
    )
    user_prompt = f"""
The user has asked the following question:

{question}

Order all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.

Here are the chunks:

{chunk_text}

Reply only with the list of ranked chunk ids, nothing else.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    order = RankOrder.model_validate_json(response.choices[0].message.content).order
    return [chunks[index - 1] for index in order]


def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}"
        for chunk in chunks
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        *history,
        {"role": "user", "content": question},
    ]


@retry(wait=RETRY_WAIT)
def rewrite_query(question, history=None):
    """Use chat history to turn a contextual follow-up into a searchable query."""
    history = history or []

    message = f"""
You are in a conversation with a user.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Since the conversation is contextual, understand the meaning of the user question and add details based on the history.
Condense everything in a single contextually-rich VERY short and specific question, most likely to surface content.

EXAMPLE:
user: Who is the founder? -> Query: who is the founder?
assistant: The founder is FooBar
user: What role covers? -> Query: What role FooBar covers?
...

IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    response = completion(
        model=MODEL, messages=[{"role": "system", "content": message}]
    )
    return response.choices[0].message.content


def merge_chunks(*chunk_groups):
    merged = []
    seen = set()
    for chunks in chunk_groups:
        for chunk in chunks:
            if chunk.page_content in seen:
                continue
            seen.add(chunk.page_content)
            merged.append(chunk)
    return merged


def fetch_context_unranked(question):
    query = embeddings(model=EMBEDDING_MODEL, input=[question]).data[0].embedding
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    return [
        Result(page_content=document, metadata=metadata)
        for document, metadata in zip(results["documents"][0], results["metadatas"][0])
    ]


def fetch_context(original_question, history=None):
    rewritten_question = rewrite_query(original_question, history)
    chunks = merge_chunks(
        fetch_context_unranked(original_question),
        fetch_context_unranked(rewritten_question),
    )
    return rerank(original_question, chunks)[:FINAL_K]


@retry(wait=RETRY_WAIT)
def answer_question(
    question: str, history: list[dict] | None = None
) -> tuple[str, list[Result]]:
    history = history or []
    chunks = fetch_context(question, history)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks
