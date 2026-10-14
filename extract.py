import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from slots import BookingSlots
from retry import with_retry

load_dotenv()
client = genai.Client()
MODEL = os.getenv("GEMINI_MODEL")

INSTRUCTIONS = """Extract ambulance booking details from the user's message.
Return only what the user actually said. Use null for anything not mentioned. Never guess.
Keep names and places exactly as spoken.
Set is_emergency to true only if the user describes an urgent situation
(collapse, accident, breathing trouble, unconscious, heavy bleeding)."""


def extract_slots(message: str, current: BookingSlots) -> BookingSlots:
    prompt = (
        f"{INSTRUCTIONS}\n\n"
        f"Details collected so far: {current.model_dump_json()}\n\n"
        f"New user message: {message}\n\n"
        "Return the updated details."
    )
    res = with_retry(lambda: client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=BookingSlots,
        ),
    ))
    return res.parsed