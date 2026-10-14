import time
from google.genai import errors


def with_retry(fn, attempts=4):
    for attempt in range(attempts):
        try:
            return fn()
        except errors.APIError as e:
            if e.code not in (429, 500, 503) or attempt == attempts - 1:
                raise
            wait = 2 ** attempt * 3
            print(f"[gemini {e.code}] retrying in {wait}s")
            time.sleep(wait)