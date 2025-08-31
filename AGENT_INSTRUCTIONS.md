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


---

## 🧪 受け入れ基準（Definition of Done）
- [ ] ノートを複数ファイル（txt/md/pdfのテキスト抽出済み）で投入し、**重複なく**Chromaに格納できる  
- [ ] 「マップ作成」押下で、**10ノード以上**が可視化・ドラッグ可・ズーム可  
- [ ] 「不足指摘/クイズ生成」押下で、**不足点3つ以上・クイズ3問以上**を表示  
- [ ] 再読み込みしてもデータが残る（Chroma 永続化）  
- [ ] 例外発生時はユーザに**原因サマリ**を表示（API Key 未設定・トークン超過・ネットワークなど）

---

## 🧩 実装タスク（順序通りに実装）
1. **セットアップ**
   - `requirements.txt` インストール、`.env` 読み込み
   - Streamlit 起動ベースの雛形（サイドバー＋メイン）

2. **Embedding サービス**
   - `services/embeddings.py`
   - 関数: `get_embedding(text: str) -> list[float]`
   - OpenAI Embedding API（`text-embedding-3-small`）を利用、リトライつき

3. **Vector Store（Chroma）**
   - `services/vector_store.py`
   - `init_store(persist_dir: str)`
   - `upsert_texts(items: list[dict])`  # {id, text, meta:{title, source}}  
   - `search(query: str, top_k: int) -> list[dict]`  # {id, text, score, meta}

4. **グラフ生成**
   - `services/graph.py`
   - 入力: 検索結果 or 全ドキュメントの一部
   - 出力: pyvis Network を HTML iframe で描画できるオブジェクト
   - エッジ重み: 余弦類似度/スコアに基づき閾値で結線（例: score ≥ 0.7）

5. **不足指摘 & クイズ生成**
   - `services/insights.py`
   - 入力: 上位検索結果（テキスト）とユーザ要約
   - 出力:
     - 不足ポイント: 箇条書き（3〜5点）
     - クイズ: 選択式3問（解答・解説つき）
   - OpenAI Chat Completions（`gpt-4o-mini`）を使用。プロンプトに**出力フォーマット**を厳密指定（JSON）

6. **UI 統合（streamlit_app.py）**
   - アップロード → 保存（`/app/data/raw`）
   - 「インデックス作成」ボタンでテキスト読み → upsert
   - 検索クエリ入力 → 結果一覧表示 → 「マップ作成」/「不足指摘・クイズ」ボタン
   - 生成物をカード/タブで見やすく表示

---

## 🧱 データモデル（Chroma メタ）
- `id`: 一意なドキュメントID（`<filename>#<chunk_index>` を推奨）
- `text`: チャンク化された本文（800～1200トークン目安）
- `meta`:
  - `title`: 元ファイル名
  - `source`: ファイルパス or ユーザ入力
  - `created_at`: ISO日時文字列
  - `tags`: 任意（ユーザ入力）

---

## 🧠 LLM プロンプト（雛形）
### 不足指摘/クイズ（System）
あなたは学習支援の専門家です。与えられた学習ノートから
(1) 理解が浅い/不足しているポイント（3〜5項目）
(2) 選択式クイズ（3問、各問に選択肢4つ、正解と1行解説）
を日本語でJSONとして返してください。冗長説明は不要。
JSON スキーマ:
{
"gaps": ["...","..."],
"quiz": [
{"question":"...", "choices":["A","B","C","D"], "answer":"B", "explanation":"..."},
...
]
}

shell
コードをコピーする

### 不足指摘/クイズ（User）
【要約】:
{{user_summary}}

【参考メモ（抜粋）】:
{{top_k_snippets}}

yaml
コードをコピーする

---

## 🧷 コーディング規約（簡易）
- 例外は `try/except` で捕捉し、Streamlit `st.warning` or `st.error` で要因表示
- API呼び出しは**リトライ**（指数バックオフ）をつける
- 関数は **入出力型ヒント** を必ず付与
- UIは**縦一列**の操作フロー（上から順に: アップロード → インデックス → 検索 → マップ/分析）

---

## ▶️ 実行手順（開発者向け）
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # API キーを設定
streamlit run app/streamlit_app.py