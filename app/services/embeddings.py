from typing import List
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import OpenAI, BadRequestError, APITimeoutError, APIConnectionError, RateLimitError
import os
import tiktoken

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Default max token length for embedding requests (text-embedding-3-small supports 8192 tokens)
EMBED_MAX_TOKENS = int(os.getenv("EMBED_MAX_TOKENS", "8000"))


def _truncate_by_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Fallback: naive char cut if tokenizer unavailable
        return text[: max(0, int(max_tokens * 4))]
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    tokens = tokens[:max_tokens]
    try:
        return enc.decode(tokens)
    except Exception:
        # As a last resort, fallback to char cut roughly
        return text[: max(0, int(max_tokens * 4))]


# Retry only on transient OpenAI errors, not on BadRequest (which is usually input-too-long)
@retry(
    wait=wait_exponential(min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError)),
)
def get_embedding(text: str) -> List[float]:
    model = os.getenv("OPENAI_MODEL_EMBED", "text-embedding-3-small")
    safe = _truncate_by_tokens(text or "", EMBED_MAX_TOKENS)
    res = client.embeddings.create(model=model, input=safe)
    return res.data[0].embedding
