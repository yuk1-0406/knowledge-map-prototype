import os, json
from openai import OpenAI
from typing import List, Dict

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "あなたは学習支援の専門家です。与えられた学習ノートから"
    "(1) 不足ポイント（3〜5） (2) 選択式クイズ3問 をJSONで返す。"
    'スキーマ: {"gaps":[...], "quiz":[{"question":"","choices":["A","B","C","D"],"answer":"A","explanation":""},...]}'
)

def generate_gaps_and_quiz(summary: str, snippets: List[str]) -> Dict:
    user = f"【要約】:\n{summary}\n\n【参考メモ（抜粋）】：\n" + "\n---\n".join(snippets[:5])
    res = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini"),
        messages=[{"role":"system","content":SYSTEM_PROMPT}, {"role":"user","content":user}],
        temperature=0.4
    )
    content = res.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        data = {"gaps": [], "quiz": []}
    return data
