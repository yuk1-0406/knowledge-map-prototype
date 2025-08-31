from typing import List
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
def get_embedding(text: str) -> List[float]:
    model = os.getenv("OPENAI_MODEL_EMBED", "text-embedding-3-small")
    text = text[:8000]
    res = client.embeddings.create(model=model, input=text)
    return res.data[0].embedding
