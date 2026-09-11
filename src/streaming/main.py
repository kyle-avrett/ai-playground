import asyncio
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/stream")
async def stream_chat(payload: dict):
    """
    Simulates a streaming LLM response using Server-Sent Events (SSE).
    Accepts a prompt payload from the Quasar frontend.
    """
    prompt = payload.get("prompt", "")
    text_to_stream = f"Received your prompt: '{prompt}'. Here is your simulated streaming AI response token by token."

    async def token_generator():
        # Split into simulated 'tokens' (words and spaces)
        tokens = [word + " " for word in text_to_stream.split(" ")]
        
        for token in tokens:
            # Format the payload exactly to match the OpenAI SSE convention expected by the frontend
            data = {
                "choices": [
                    {
                        "delta": {
                            "content": token
                        }
                    }
                ]
            }
            # SSE protocol format: 'data: <JSON_STRING>\n\n'
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.08)  # Mimic natural LLM output pacing
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")
