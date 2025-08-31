# AGENT_INSTRUCTIONS.md
# プロジェクト: 知識マップ × 学習支援（ハッカソン最小プロトタイプ）

## 🎯 目的（必達）
1) ノート/テキスト投入 → Embedding → ベクトルDB格納 → 類似検索  
2) 簡易「知識マップ」を描画（ノード=文書/トピック、エッジ=近接）  
3) LLMで不足指摘と選択式クイズ（3問以上）を生成

## スタック
- Python 3.11 / Streamlit
- OpenAI API（chat + embeddings）
- ChromaDB（ローカル）
- pyvis / networkx

## 実行
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app/streamlit_app.py
```
