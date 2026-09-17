from multiprocessing import Pool
from pathlib import Path

from chromadb import PersistentClient
from llm import completion, embeddings
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
DB_PATH = str(BASE_DIR / "database")
KNOWLEDGE_BASE_PATH = BASE_DIR / "knowledge"
COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "text-embedding-3-large"
MODEL = "openai/gpt-4.1-nano"
AVERAGE_CHUNK_SIZE = 100
RETRY_WAIT = wait_exponential(multiplier=1, min=10, max=240)

# One chunking request per document; lower this when OpenAI rate limits.
WORKERS = 3


class Result(BaseModel):
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    headline: str = Field(
        description="A short heading likely to match a user query.",
    )
    summary: str = Field(
        description="A few sentences summarizing the chunk for question answering."
    )
    original_text: str = Field(
        description="The exact source text in this chunk, unchanged."
    )

    def as_result(self, document):
        return Result(
            page_content=f"{self.headline}\n\n{self.summary}\n\n{self.original_text}",
            metadata={"source": document["source"], "type": document["type"]},
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


def fetch_documents():
    documents = []
    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        if not folder.is_dir():
            continue
        for file in folder.rglob("*.md"):
            documents.append(
                {
                    "type": folder.name,
                    "source": file.as_posix(),
                    "text": file.read_text(encoding="utf-8"),
                }
            )

    print(f"Loaded {len(documents)} documents")
    return documents


def make_prompt(document):
    chunk_count = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {chunk_count} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


def make_messages(document):
    return [{"role": "user", "content": make_prompt(document)}]


@retry(wait=RETRY_WAIT)
def process_document(document):
    response = completion(
        model=MODEL, messages=make_messages(document), response_format=Chunks
    )
    chunks = Chunks.model_validate_json(response.choices[0].message.content).chunks
    return [chunk.as_result(document) for chunk in chunks]


def create_chunks(documents):
    chunks = []
    with Pool(processes=WORKERS) as pool:
        results = pool.imap_unordered(process_document, documents)
        for result in tqdm(results, total=len(documents)):
            chunks.extend(result)
    return chunks


def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_PATH)
    if COLLECTION_NAME in [collection.name for collection in chroma.list_collections()]:
        chroma.delete_collection(COLLECTION_NAME)

    texts = [chunk.page_content for chunk in chunks]
    vectors = [
        item.embedding for item in embeddings(model=EMBEDDING_MODEL, input=texts).data
    ]

    collection = chroma.get_or_create_collection(COLLECTION_NAME)
    collection.add(
        ids=[str(index) for index in range(len(chunks))],
        embeddings=vectors,
        documents=texts,
        metadatas=[chunk.metadata for chunk in chunks],
    )
    print(f"Vectorstore created with {collection.count()} documents")


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
