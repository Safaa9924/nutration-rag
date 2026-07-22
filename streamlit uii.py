"""
app.py
======
Streamlit UI for the refactored Stage-6 RAG retrieval pipeline.

Two modes:

  * Demo mode  — no ML dependencies required (no sentence-transformers,
    no chroma store, no cross-encoder download). Uses a small built-in
    sample corpus and a word-overlap scorer as a stand-in for the real
    retriever/reranker, purely so you can click around the UI and see
    how query building -> retrieval -> rerank -> context assembly ->
    diagnostics fit together.

  * Live mode  — imports retriever.py / reranker.py and calls your real
    hybrid retrieval + cross-encoder reranker against your actual
    semantic_chunks_final.csv + indexes from 05_create_chroma_store.py.
    Requires those files/deps to be present in the working directory.

Run with:
    streamlit run app.py
"""

import sys
import os
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intent_detector import detect_intent
from query_builder import build_query_plan, CONDITION_KEYWORDS, INTENT_KEYWORDS
from context_builder import build_context_package
from retrieval_report import build_retrieval_report


# ==================================================
# Page setup
# ==================================================
st.set_page_config(page_title="RAG Retrieval Explorer", page_icon="🔎", layout="wide")

st.title("🔎 RAG Retrieval Pipeline Explorer")
st.caption("Query Builder → Retriever → Reranker → Context Builder → Retrieval Report")


# ==================================================
# Demo corpus (used only in Demo mode)
# ==================================================
DEMO_CORPUS = [
    {"chunk_id": "c001", "page": 12, "chunk_text": "Patients with type 1 diabetes should avoid sugary drinks, white bread, and refined carbohydrates because these foods spike blood glucose quickly and make control harder."},
    {"chunk_id": "c002", "page": 12, "chunk_text": "Foods to avoid include sweetened beverages, pastries, and highly processed snacks that are high in added sugar and low in fibre."},
    {"chunk_id": "c003", "page": 18, "chunk_text": "Recommended foods for glycaemic control include non-starchy vegetables, legumes, whole grains, and lean protein sources."},
    {"chunk_id": "c004", "page": 20, "chunk_text": "A balanced breakfast might include oats with nuts and berries, which provide fibre and a lower glycaemic impact than sugary cereals."},
    {"chunk_id": "c005", "page": 21, "chunk_text": "Meal planning should space carbohydrate intake evenly across breakfast, lunch, dinner, and snacks to avoid large glucose spikes."},
    {"chunk_id": "c006", "page": 27, "chunk_text": "Sugary sodas, fruit juices with added sugar, and alcoholic beverages should be consumed sparingly as they can affect blood glucose and interact with insulin."},
    {"chunk_id": "c007", "page": 33, "chunk_text": "Regular physical activity, such as brisk walking for 30 minutes most days, improves insulin sensitivity and helps with glycaemic control."},
    {"chunk_id": "c008", "page": 40, "chunk_text": "Hypoglycaemia should be treated promptly with a fast-acting carbohydrate such as glucose tablets or juice, followed by a longer-acting snack."},
    {"chunk_id": "c009", "page": 45, "chunk_text": "Long-term complications of poorly controlled diabetes include kidney disease (nephropathy), cardiovascular disease, and nerve damage (neuropathy)."},
    {"chunk_id": "c010", "page": 46, "chunk_text": "Patients with diabetic nephropathy are often advised to moderate protein intake and monitor blood pressure closely to slow kidney disease progression."},
    {"chunk_id": "c011", "page": 52, "chunk_text": "Gestational diabetes nutrition therapy focuses on consistent carbohydrate distribution across meals to support a healthy pregnancy and normal foetal growth."},
    {"chunk_id": "c012", "page": 53, "chunk_text": "During pregnancy, women with gestational diabetes should avoid sugary snacks and prioritise high-fibre, low glycaemic index foods."},
    {"chunk_id": "c013", "page": 12, "chunk_text": "Patients with type 1 diabetes should avoid sugary drinks, white bread, and refined carbs, since these spike blood glucose quickly and complicate control."},
    {"chunk_id": "c014", "page": 60, "chunk_text": "Weight management through portion control and regular activity is a cornerstone of pre-diabetes lifestyle modification programmes."},
    {"chunk_id": "c015", "page": 61, "chunk_text": "Fibre-rich foods slow gastric emptying and blunt post-meal glucose spikes, making them a useful tool for glycaemic control."},
]


def _word_overlap_score(query: str, text: str) -> float:
    """Cheap stand-in for a real retriever/reranker score, demo-mode only."""
    q = set(query.lower().split())
    t = set(text.lower().split())
    if not q or not t:
        return 0.0
    return len(q & t) / len(q | t)


def demo_retrieve_and_rerank(query_text, k, top_n):
    df = pd.DataFrame(DEMO_CORPUS)
    df["score"] = df["chunk_text"].apply(lambda t: _word_overlap_score(query_text, t))
    df["retriever"] = "Demo (word-overlap)"
    retrieved = df.sort_values("score", ascending=False).head(k).reset_index(drop=True)

    reranked = retrieved.head(top_n).copy()
    reranked["rerank_score"] = reranked["score"] * 10  # scale to look like cross-encoder logits
    return retrieved, reranked


# ==================================================
# Sidebar — configuration
# ==================================================
with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "Pipeline mode",
        ["Demo (sample data, no ML deps)", "Live (your real indexes)"],
        index=0,
    )
    is_demo = mode.startswith("Demo")

    st.divider()
    st.header("Query")

    condition = st.selectbox(
        "Predicted condition",
        ["None"] + list(CONDITION_KEYWORDS.keys()),
        index=1,
    )
    condition = None if condition == "None" else condition

    intent_mode = st.radio("Intent", ["Auto-detect", "Manual"], index=0, horizontal=True)
    manual_intent = None
    if intent_mode == "Manual":
        manual_intent = st.selectbox("Select intent", list(INTENT_KEYWORDS.keys()))

    st.divider()
    st.header("Retrieval")
    k = st.slider("Candidates to retrieve (k)", 5, 100, 40, step=5)
    top_n = st.slider("Candidates kept after rerank (top_n)", 1, 20, 10)

    st.divider()
    st.header("Context building")
    max_context_chunks = st.slider("Max context chunks", 1, 20, 10)
    word_budget = st.slider("Word budget", 100, 3000, 1200, step=100)

    threshold_mode = st.radio("Score threshold mode", ["auto", "fixed"], index=0, horizontal=True)
    min_score_ratio = "auto"
    if threshold_mode == "fixed":
        min_score_ratio = st.slider("Fixed ratio (of max score)", 0.05, 0.95, 0.4, step=0.05)

    near_dup_threshold = st.slider("Near-duplicate threshold (Jaccard)", 0.0, 1.0, 0.7, step=0.05)

    if not is_demo:
        st.divider()
        st.header("Live mode data")
        csv_path = st.text_input("chunks CSV path", value="semantic_chunks_final.csv")
        st.caption(
            "Live mode also expects `04_vector_representation.py` and "
            "`05_create_chroma_store.py` in the working directory, and "
            "`sentence-transformers` installed."
        )


# ==================================================
# Live-mode index loading (cached)
# ==================================================
@st.cache_resource(show_spinner="Loading indexes (this can take a while the first time)...")
def _load_live_indexes(csv_path: str):
    from importlib.machinery import SourceFileLoader
    chunks_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    store = SourceFileLoader("stage5_create_chroma_store", "05_create_chroma_store.py").load_module()
    tfidf_vectorizer, tfidf_matrix, bm25, embedding_model, embedding_matrix = store.load_indexes()
    return chunks_df, tfidf_vectorizer, tfidf_matrix, bm25, embedding_model, embedding_matrix


# ==================================================
# Main input
# ==================================================
user_question = st.text_area(
    "User question",
    value="What foods should a Type 1 diabetic avoid?",
    height=80,
)

run = st.button("▶ Run retrieval pipeline", type="primary")

if run:
    with st.spinner("Building query plan..."):
        query_plan = build_query_plan(
            user_question,
            condition=condition,
            intent=manual_intent,  # None -> auto-detected inside build_query_plan
        )

    if is_demo:
        with st.spinner("Retrieving + reranking (demo scorer)..."):
            retrieved, reranked = demo_retrieve_and_rerank(query_plan.query_text, k, top_n)
    else:
        try:
            with st.spinner("Loading indexes..."):
                (chunks_df, tfidf_vectorizer, tfidf_matrix,
                 bm25, embedding_model, embedding_matrix) = _load_live_indexes(csv_path)

            from retriever import retrieve_top_k_hybrid
            from reranker import rerank_candidates

            with st.spinner("Retrieving (hybrid)..."):
                retrieved = retrieve_top_k_hybrid(
                    query=query_plan.query_text,
                    tfidf_vectorizer=tfidf_vectorizer, tfidf_matrix=tfidf_matrix,
                    bm25=bm25,
                    embedding_model=embedding_model, embedding_matrix=embedding_matrix,
                    chunks_df=chunks_df,
                    k=k,
                )
            with st.spinner("Reranking (cross-encoder)..."):
                reranked = rerank_candidates(query_plan.query_text, retrieved, top_n=top_n)

        except Exception as e:
            st.error(f"Live mode failed: {e}")
            st.info("Falling back to Demo mode data so you can still explore the UI.")
            retrieved, reranked = demo_retrieve_and_rerank(query_plan.query_text, k, top_n)

    with st.spinner("Building context package..."):
        context_result = build_context_package(
            query=query_plan.query_text,
            reranked_df=reranked,
            max_context_chunks=max_context_chunks,
            word_budget=word_budget,
            min_score_ratio=min_score_ratio,
            near_duplicate_threshold=near_dup_threshold,
        )

    report = build_retrieval_report(
        query_plan=query_plan,
        num_retrieved=len(retrieved),
        num_reranked=len(reranked),
        context_result=context_result,
    )

    st.session_state["query_plan"] = query_plan
    st.session_state["retrieved"] = retrieved
    st.session_state["reranked"] = reranked
    st.session_state["context_result"] = context_result
    st.session_state["report"] = report


# ==================================================
# Results
# ==================================================
if "context_result" in st.session_state:
    query_plan = st.session_state["query_plan"]
    retrieved = st.session_state["retrieved"]
    reranked = st.session_state["reranked"]
    context_result = st.session_state["context_result"]
    report = st.session_state["report"]

    tab_plan, tab_retrieved, tab_reranked, tab_context, tab_diag = st.tabs(
        ["🧩 Query Plan", "📥 Retrieved", "🎯 Reranked", "📄 Final Context", "🧪 Diagnostics"]
    )

    with tab_plan:
        c1, c2, c3 = st.columns(3)
        c1.metric("Condition", query_plan.condition or "—")
        c2.metric("Detected intent", query_plan.intent)
        c3.metric("Final query length (words)", len(query_plan.query_text.split()))

        st.subheader("Final retrieval query")
        st.code(query_plan.query_text, language=None)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Condition keywords")
            st.write(query_plan.condition_keywords or "—")
        with col2:
            st.subheader("Intent keywords")
            st.write(query_plan.intent_keywords or "—")

    with tab_retrieved:
        st.subheader(f"Retrieved candidates ({len(retrieved)})")
        st.dataframe(retrieved, width='stretch')

    with tab_reranked:
        st.subheader(f"Reranked candidates ({len(reranked)})")
        st.dataframe(reranked, width='stretch')

    with tab_context:
        m1, m2 = st.columns(2)
        m1.metric("Sources used", context_result["num_sources"])
        m2.metric("Words used", context_result["used_words"])

        st.subheader("Context passed to the LLM")
        st.text_area("context_text", context_result["context_text"], height=400, label_visibility="collapsed")

        st.download_button(
            "⬇ Download context as .txt",
            data=context_result["context_text"],
            file_name="context_package.txt",
            mime="text/plain",
        )

    with tab_diag:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Candidates in", report["num_retrieved_candidates"])
        m2.metric("Reranked", report["num_reranked_candidates"])
        m3.metric("Removed (low score)", report["num_removed_low_score"])
        m4.metric("Removed (duplicates)", report["num_removed_duplicates"])

        st.write(f"**Threshold mode:** {report['threshold_mode']}  |  **Threshold used:** {report['threshold_used']:.4f}"
                 if report["threshold_used"] is not None else "**No threshold computed (empty candidate set)**")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Removed — low score")
            st.dataframe(pd.DataFrame(report["removed_low_score_detail"]), width='stretch')
        with col2:
            st.subheader("Removed — duplicates")
            st.dataframe(pd.DataFrame(report["removed_duplicates_detail"]), width='stretch')
else:
    st.info("Enter a question and click **Run retrieval pipeline** to see results.")
