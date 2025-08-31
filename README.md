# 知識マップ × 学習支援（ハッカソン最小プロトタイプ）

## セットアップ
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # OPENAI_API_KEY を設定
streamlit run app/streamlit_app.py
```

## 機能
- ノートアップロード → Embedding → VectorDB(Chroma)格納
- 類似検索 → 知識マップ（pyvis）描画
- LLM による不足ポイント指摘＆クイズ生成

## VS Code Quick Start
1. フォルダを VS Code で開く  
2. `Terminal → Run Task... → Run Streamlit`（自動で venv 作成→依存導入→起動）  
3. もしくは `Run and Debug` → **Run Streamlit App**

## ディレクトリ
```
knowledge-map-prototype/
  ├─ app/
  │  ├─ services/  # API/DB/グラフ/洞察
  │  ├─ ui/
  │  ├─ utils/
  │  └─ data/{
  │       raw/, chroma/
  │     }
  ├─ .vscode/
  ├─ .env.example
  ├─ requirements.txt
  ├─ AGENT_INSTRUCTIONS.md
  └─ README.md
```
