# RAG

## Quickstart

``` shell
cp .env.example .env
docker compose up

uv run ingest.py

uv run frontend_eval.py
uv run frontend_chat.py
```

## Layout

```text
src/rag/
├── docker-compose.yml   # runs LiteLLM proxy on :4000 plus Postgres for LiteLLM state
├── litellm_config.yml   # LiteLLM model aliases, master key, and database settings
├── llm.py               # shared OpenAI-compatible client for completions and embeddings
├── ingest.py            # read knowledge/*.md, LLM-chunk docs, embed chunks, write ChromaDB
├── answer.py            # RAG runtime: rewrite query, retrieve twice, merge, rerank, answer
├── eval.py              # shared evaluation core and CLI for one tests.jsonl row
├── frontend_chat.py     # Gradio chat UI with retrieved-context pane
├── frontend_eval.py     # Gradio evaluation dashboard with metrics and category charts
├── tests.jsonl          # eval questions, keywords, categories, reference answers
├── database/            # generated ChromaDB index, collection "docs"
└── knowledge/           # source markdown docs
    ├── company/
    ├── contracts/
    ├── employees/
    └── products/
```

## Diagram

```mermaid
flowchart LR
  subgraph Service["Local LLM service"]
    Compose["docker-compose.yml"]
    Config["litellm_config.yml"]
    Env["environment variables"]
    Proxy["LiteLLM proxy\n:4000"]
    Pg[("Postgres\nLiteLLM state")]
  end

  subgraph Client["Shared LLM client: llm.py"]
    Completion["completion()"]
    Embeddings["embeddings()"]
  end

  subgraph Build["Index build: ingest.py"]
    Docs["knowledge/**/*.md"]
    Load["fetch_documents"]
    Chunk["process_document\nLLM chunking"]
    Embed["create_embeddings"]
  end

  Index[("ChromaDB\n./database\ncollection: docs")]

  subgraph Runtime["Answer path: answer.py"]
    Question["question + history"]
    Rewrite["rewrite_query"]
    SearchOriginal["fetch_context_unranked\noriginal question"]
    SearchRewrite["fetch_context_unranked\nrewritten question"]
    Merge["merge_chunks"]
    Rerank["rerank"]
    Final["answer_question"]
  end

  subgraph ChatUI["Chat UI: frontend_chat.py"]
    Chat["Gradio Chatbot"]
    Context["Retrieved context pane"]
  end

  subgraph Eval["Evaluation"]
    Tests["tests.jsonl"]
    EvalCore["eval.py\nevaluate_all_retrieval\nevaluate_all_answers\nMRR, nDCG, coverage,\nLLM judge scores"]
    EvalUI["frontend_eval.py\nmetrics + category charts"]
  end

  Compose --> Proxy
  Config --> Proxy
  Env --> Proxy
  Proxy --> Pg
  Completion --> Proxy
  Embeddings --> Proxy

  Docs --> Load --> Chunk --> Embed --> Index
  Chunk --> Completion
  Embed --> Embeddings

  Chat --> Question
  Question --> Rewrite
  Rewrite --> Completion
  Question --> SearchOriginal
  Rewrite --> SearchRewrite
  SearchOriginal --> Embeddings
  SearchRewrite --> Embeddings
  SearchOriginal --> Index
  SearchRewrite --> Index
  Index --> Merge --> Rerank --> Final
  Rerank --> Completion
  Final --> Completion
  Final --> Chat
  Rerank --> Context

  Tests --> EvalCore
  EvalCore --> Question
  EvalCore --> EvalUI
```
