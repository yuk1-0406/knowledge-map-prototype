import streamlit as st, os, uuid
from services.vector_store import init_store, upsert_texts, search
from services.graph import build_graph
from services.insights import generate_gaps_and_quiz

from dotenv import load_dotenv
load_dotenv()  # .env ファイルを読み込む

st.set_page_config(page_title="Knowledge Map Prototype", layout="wide")
st.title("知識マップ × 学習支援（ハッカソン試作）")

DATA_RAW = "app/data/raw"
os.makedirs(DATA_RAW, exist_ok=True)
init_store()

with st.sidebar:
    st.header("1) ノートをアップロード")
    files = st.file_uploader("txt/md形式推奨（pdfはテキスト抽出後のtxt）", type=["txt","md"], accept_multiple_files=True)
    if st.button("インデックス作成", type="primary") and files:
        items = []
        for f in files:
            text = f.read().decode("utf-8", errors="ignore")
            fid = f"{f.name}#{uuid.uuid4().hex[:8]}"
            path = os.path.join(DATA_RAW, f"{fid}.txt")
            with open(path, "w", encoding="utf-8") as out:
                out.write(text)
            items.append({"id": fid, "text": text, "meta": {"title": f.name, "source": path}})
        with st.status("Embedding & 登録中...", expanded=True):
            upsert_texts(items)
            st.write(f"{len(items)} 件を登録しました。")
        st.success("インデックス作成 完了")

st.header("2) 検索 & マップ")
q = st.text_input("検索クエリ（空でもOK: 全体から一部を可視化）", "")
topk = st.slider("取得件数", 5, 50, 15)

if st.button("マップ作成"):
    results = search(q or "overview", topk)
    html_path = build_graph(results)
    with open(html_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=620, scrolling=True)
    st.session_state["last_results"] = results
    st.success("マップ生成 完了")

st.header("3) 不足の指摘 & クイズ生成")
summary = st.text_area("要約（任意）", "これまでの学習の要点...")
if st.button("不足/クイズ 生成", type="secondary"):
    results = st.session_state.get("last_results", search(q or "overview", 10))
    snippets = [r["text"][:800] for r in results]
    data = generate_gaps_and_quiz(summary, snippets)
    st.subheader("不足ポイント")
    for g in data.get("gaps", []):
        st.markdown(f"- {g}")
    st.subheader("クイズ")
    for i, qz in enumerate(data.get("quiz", []), 1):
        st.markdown(f"**Q{i}. {qz.get('question','')}**")
        for ch in qz.get("choices", []):
            st.markdown(f"- {ch}")
        st.markdown(f"> 正解: **{qz.get('answer','?')}** / 解説: {qz.get('explanation','')}")
