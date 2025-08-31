from dotenv import load_dotenv
load_dotenv()  # .env ファイルを読み込む

import streamlit as st, os, uuid, json
# Import Streamlit's rerun exception to avoid catching it as an error
try:
    from streamlit.runtime.scriptrunner import RerunException  # type: ignore
except Exception:  # Fallback for older/newer versions
    RerunException = None  # type: ignore

from services.vector_store import init_store, upsert_texts, search, get_count, list_items, delete_by_ids, delete_all
from services.graph import build_graph
from services.insights import generate_gaps_and_quiz
from datetime import datetime

st.set_page_config(page_title="Knowledge Map Prototype", layout="wide")
st.title("知識マップ × 学習支援（ハッカソン試作）")

DATA_RAW = "app/data/raw"
EXPORT_DIR = "app/data/export"
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
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

    # --- DB確認 ---
    st.divider()
    with st.expander("DB確認", expanded=False):
        try:
            cnt = get_count()
            st.metric("ドキュメント件数", cnt)
            default_limit = 20 if cnt >= 20 else max(1, cnt) if cnt > 0 else 10
            limit = st.slider("表示件数 (最大50)", 1, 50, default_limit, key="db_limit")
            items = list_items(limit)
            rows = []
            for it in items:
                meta = it.get("meta", {}) or {}
                rows.append({
                    "id": it.get("id", ""),
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "length": len((it.get("text") or ""))
                })
            st.dataframe(rows, hide_index=True, use_container_width=True)

            # Deletion controls
            id_to_label = {}
            for r in rows:
                label = f"{r['title'] or '(no title)'} ({r['id']})"
                id_to_label[r["id"]] = label
            selected_ids = st.multiselect(
                "削除対象（複数選択可）",
                options=list(id_to_label.keys()),
                format_func=lambda x: id_to_label.get(x, x),
                key="db_del_ids",
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("選択を削除", use_container_width=True, key="db_del_btn"):
                    if selected_ids:
                        n = delete_by_ids(selected_ids)
                        st.success(f"{n} 件を削除しました。")
                        st.rerun()
                    else:
                        st.warning("削除対象を選択してください。")
            with c2:
                confirm = st.checkbox("全削除を許可する（危険）", key="db_del_all_ck")
                if st.button("全削除", type="secondary", use_container_width=True, disabled=not confirm, key="db_del_all_btn"):
                    if confirm:
                        n = delete_all()
                        st.success(f"全削除を実行しました（{n} 件）。")
                        st.rerun()

            payload = {"count": cnt, "items": rows}
            st.download_button(
                label="JSONをダウンロード",
                data=json.dumps(payload, ensure_ascii=False, indent=2),
                file_name="chroma_snapshot.json",
                mime="application/json",
                key="db_json_dl",
            )
        except Exception as e:
            # Allow Streamlit's rerun exception to bubble up, otherwise it shows as an error
            if (getattr(e, "rerun_data", None) is not None) or ("RerunData" in str(e)) or (RerunException and isinstance(e, RerunException)):
                raise
            st.error(f"DB確認でエラー: {e}")

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
    # デバッグ表示（件数と生出力/JSON）
    st.caption(f"不足: {len(data.get('gaps', []))}件 / クイズ: {len(data.get('quiz', []))}問")
    with st.expander("LLM生出力（デバッグ）", expanded=False):
        st.text(data.get("_raw", ""))
    with st.expander("生成JSON（デバッグ）", expanded=False):
        st.json(data)
    # 通常の表示
    st.subheader("不足ポイント")
    gaps = data.get("gaps", [])
    if isinstance(gaps, list) and gaps:
        for i, g in enumerate(gaps, 1):
            if isinstance(g, dict):
                text = g.get("text") or g.get("title") or str(g)
            else:
                text = str(g)
            st.markdown(f"- {i}. {text}")
    else:
        st.info("不足ポイントはありません。")

    st.subheader("クイズ")
    for i, qz in enumerate(data.get("quiz", []), 1):
        question = qz.get("question", "")
        st.markdown(f"**Q{i}. {question}**")
        choices = qz.get("choices", []) or []
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for idx, ch in enumerate(choices):
            st.markdown(f"- {labels[idx]}: {ch}")
        ans = qz.get("answer", "?")
        if isinstance(ans, int) and 0 <= ans < len(choices):
            ans_label = labels[ans]
        else:
            ans_label = str(ans)
        st.markdown(f"> 正解: **{ans_label}** / 解説: {qz.get('explanation','')}")

    # ここから: Markdown 生成と保存/ダウンロード
    md_lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    md_lines.append(f"# 学習支援レポート ({ts})")
    md_lines.append("")
    md_lines.append(f"- 検索クエリ: {q or 'overview'}")
    md_lines.append(f"- 取得件数: {len(results)}")
    md_lines.append("")
    md_lines.append("## 要約")
    md_lines.append(summary or "")
    md_lines.append("")

    md_lines.append("## 不足ポイント")
    if isinstance(gaps, list) and gaps:
        for i, g in enumerate(gaps, 1):
            if isinstance(g, dict):
                text = g.get("text") or g.get("title") or str(g)
            else:
                text = str(g)
            md_lines.append(f"{i}. {text}")
    else:
        md_lines.append("なし")
    md_lines.append("")

    md_lines.append("## クイズ")
    for i, qz in enumerate(data.get("quiz", []), 1):
        question = qz.get("question", "")
        choices = qz.get("choices", []) or []
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        md_lines.append(f"### Q{i}. {question}")
        for idx, ch in enumerate(choices):
            md_lines.append(f"- {labels[idx]}: {ch}")
        ans = qz.get("answer", "?")
        if isinstance(ans, int) and 0 <= ans < len(choices):
            ans_label = labels[ans]
        else:
            ans_label = str(ans)
        md_lines.append(f"> 正解: {ans_label}")
        expl = qz.get("explanation", "")
        if expl:
            md_lines.append(f"> 解説: {expl}")
        md_lines.append("")

    md_lines.append("## 参考ノート")
    for r in results:
        title = r.get("meta", {}).get("title", r.get("id", ""))
        source = r.get("meta", {}).get("source", "")
        suffix = f" ({source})" if source else ""
        md_lines.append(f"- {title}{suffix}")

    md_content = "\n".join(md_lines)
    default_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    st.download_button(
        label="Markdownをダウンロード",
        data=md_content,
        file_name=default_filename,
        mime="text/markdown",
    )

    if st.button("Markdownを保存（サーバー）"):
        save_path = os.path.join(EXPORT_DIR, default_filename)
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            st.success(f"保存しました: {save_path}")
        except Exception as e:
            st.error(f"保存に失敗しました: {e}")
