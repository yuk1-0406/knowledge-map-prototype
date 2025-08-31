import os, json
import re
from openai import OpenAI
from typing import List, Dict

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "あなたは学習支援の専門家です。与えられた学習ノートから"
    "(1) 不足ポイント（3〜5） (2) 選択式クイズ3問 をJSONで返す。"
    'スキーマ: {"gaps":[...], "quiz":[{"question":"","choices":["A","B","C","D"],"answer":"A","explanation":""},...]}'
)


def _extract_json(text: str) -> str:
    """Extract a JSON object string from text (handles ```json fences and extra prose)."""
    if not text:
        return ""
    # Prefer fenced code block content
    m = re.search(r"```(?:json|JSON)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: take substring between first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    # Last resort: remove stray backticks and return
    return text.replace("```", "").strip()


def generate_gaps_and_quiz(summary: str, snippets: List[str]) -> Dict:
    user = f"【要約】:\n{summary}\n\n【参考メモ（抜粋）】：\n" + "\n---\n".join(snippets[:5])
    res = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini"),
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        temperature=0.4,
        response_format={"type": "json_object"},
    )
    content = res.choices[0].message.content
    # Try direct parse, then sanitized fallback
    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON is not an object")
    except Exception:
        try:
            cleaned = _extract_json(content)
            data = json.loads(cleaned) if cleaned else {"gaps": [], "quiz": []}
            if not isinstance(data, dict):
                data = {"gaps": [], "quiz": []}
        except Exception:
            data = {"gaps": [], "quiz": []}
    # Attach raw response for debugging in UI
    data["_raw"] = content
    return data
