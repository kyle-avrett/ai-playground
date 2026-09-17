# Streaming Demo

FastAPI SSE endpoint and Quasar test page for fake token-by-token chat streaming.

## Quickstart

```sh
cd src/streaming
docker compose up --build
curl -N -X POST http://localhost:8000/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"hello"}'
```

Open `index.html` to use the browser client.
