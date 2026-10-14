import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()                      # loads .env into environment variables
MODEL = os.getenv("GEMINI_MODEL")
client = genai.Client()            # picks up GEMINI_API_KEY automatically


import time
from google.genai import errors


def stream_answer(prompt: str, temperature=0.7, max_tokens=800, top_p=0.95):
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        top_p=top_p,
    )
    for attempt in range(4):
        try:
            for chunk in client.models.generate_content_stream(
                model=MODEL, contents=prompt, config=config
            ):
                if chunk.text:
                    yield chunk.text
            return
        except errors.ServerError:
            if attempt == 3:
                raise
            wait = 2 ** attempt * 2      # waits 2s, 4s, 8s
            print(f"\n[server busy, retrying in {wait}s]")
            time.sleep(wait)

if __name__ == "__main__":
    for piece in stream_answer("Explain RAG in 3 simple sentences."):
        print(piece, end="", flush=True)