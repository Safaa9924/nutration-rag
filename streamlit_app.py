"""
Diabetes Nutrition RAG Assistant
==================================
Streamlit app adapted from the "Adapted RAG Pipeline — Nutritional Management
of Diabetes Mellitus" notebook.

Pipeline: Hybrid lexical+semantic retrieval (TF-IDF + BM25 + SentenceTransformer)
-> Cross-Encoder reranking -> Context packaging -> Prompt construction
-> LLM generation (via OpenRouter-compatible chat completions API)
-> Grounding / confidence diagnostics.

Data source: a precomputed chunk CSV (default name: semantic_chunks_final.csv),
bundled alongside this script or uploaded via the sidebar.
"""

import re
import time
import hashlib
from collections import Counter

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================================
# PAGE CONFIG
# ==================================================================

st.set_page_config(
    page_title="Diabetes Nutrition RAG",
    page_icon="🩺",
    layout="wide",
)

DEFAULT_CSV_NAME = "semantic_chunks_final.csv"

QUERY_MAP = {
    "Type 2 Diabetes": [
        "dietary management", "nutrition therapy", "meal planning",
        "carbohydrate intake", "glycaemic control", "glycaemic index",
        "fibre intake", "dietary fat", "weight management",
    ],
    "Type 1 Diabetes": [
        "nutrition therapy", "carbohydrate counting", "meal planning",
        "glycaemic control", "hypoglycaemia", "insulin and diet",
    ],
    "Pre-Diabetes": [
        "healthy eating", "weight management", "lifestyle modification",
        "physical activity", "glycaemic control",
    ],
    "Gestational Diabetes": [
        "nutrition therapy", "meal planning", "glycaemic control",
        "weight management", "healthy pregnancy diet",
    ],
}

DEFAULT_MODELS = [
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.2-3b-instruct",
]


# ==================================================================
# TEXT / SCORING HELPERS
# ==================================================================

def simple_tokenize(text):
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def min_max_normalize(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class MiniBM25:
    """Lightweight BM25 index (no external dependency)."""

    def __init__(self, tokenized_docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = tokenized_docs
        self.N = len(tokenized_docs)
        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avgdl = float(np.mean(self.doc_lens)) if self.doc_lens else 0.0

        self.term_freqs = [Counter(doc) for doc in tokenized_docs]

        self.df = Counter()
        for doc in tokenized_docs:
            self.df.update(set(doc))

        self.idf = {
            term: np.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for term, df in self.df.items()
        }

    def get_scores(self, query_tokens):
        scores = np.zeros(self.N, dtype=np.float32)
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, tf_dict in enumerate(self.term_freqs):
                tf = tf_dict.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_lens[i] / self.avgdl
                )
                scores[i] += (idf * tf * (self.k1 + 1)) / denom
        return scores


# ==================================================================
# CACHED RESOURCE BUILDERS
# ==================================================================

@st.cache_data(show_spinner=False)
def load_chunks_csv(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    df = pd.read_csv(BytesIO(file_bytes), encoding="utf-8-sig")
    if "chunk_text" not in df.columns:
        raise ValueError("CSV must contain a 'chunk_text' column.")
    if "chunk_id" not in df.columns:
        df.insert(0, "chunk_id", [f"chunk_{i:04d}" for i in range(1, len(df) + 1)])
    if "word_count" not in df.columns:
        df["word_count"] = df["chunk_text"].apply(lambda t: len(str(t).split()))
    df["chunk_text"] = df["chunk_text"].astype(str)
    return df.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def build_tfidf_index(texts: tuple):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.90,
        max_features=30000,
        sublinear_tf=True,
        norm="l2",
        dtype="float32",
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


@st.cache_resource(show_spinner=False)
def build_bm25_index(texts: tuple):
    tokenized_docs = [simple_tokenize(t) for t in texts]
    return MiniBM25(tokenized_docs)


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L12-v2"):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


@st.cache_data(show_spinner=False)
def build_embedding_matrix(_model, texts: tuple, cache_key: str):
    return _model.encode(
        list(texts),
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


# ==================================================================
# RETRIEVAL FUNCTIONS
# ==================================================================

def retrieve_top_k_hybrid(
    query, tfidf_vectorizer, tfidf_matrix, bm25, embedding_model, embedding_matrix,
    chunks_df, tfidf_weight=0.34, bm25_weight=0.33, semantic_weight=0.33, k=40,
):
    q_vec = tfidf_vectorizer.transform([query])
    tfidf_scores = min_max_normalize(cosine_similarity(q_vec, tfidf_matrix).flatten())

    bm25_scores = min_max_normalize(bm25.get_scores(simple_tokenize(query)))

    query_embedding = embedding_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    semantic_scores = min_max_normalize(
        cosine_similarity(query_embedding, embedding_matrix).flatten()
    )

    combined = (
        tfidf_weight * tfidf_scores
        + bm25_weight * bm25_scores
        + semantic_weight * semantic_scores
    )

    k = min(k, len(chunks_df))
    ranking = np.argsort(combined)[::-1][:k]

    results = chunks_df.iloc[ranking].copy()
    results["tfidf_score"] = tfidf_scores[ranking]
    results["bm25_score"] = bm25_scores[ranking]
    results["semantic_score"] = semantic_scores[ranking]
    results["score"] = combined[ranking]
    results["retriever"] = "Hybrid"

    return results[
        ["retriever", "chunk_id", "tfidf_score", "bm25_score",
         "semantic_score", "score", "chunk_text"]
    ].reset_index(drop=True)


def rerank_candidates(reranker, query, candidates_df, top_n=10):
    pairs = [(query, text) for text in candidates_df["chunk_text"]]
    scores = reranker.predict(pairs)
    reranked = candidates_df.copy()
    reranked["rerank_score"] = scores
    return (
        reranked.sort_values("rerank_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def build_context_package(query, reranked_df, max_context_chunks=10,
                           word_budget=1200, min_score_ratio=0.4):
    candidates = reranked_df.sort_values("rerank_score", ascending=False).reset_index(drop=True)

    if candidates.empty:
        return {"query": query, "selected_df": pd.DataFrame(),
                "context_text": "", "num_sources": 0, "used_words": 0}

    max_score = candidates["rerank_score"].max()
    selected_rows, seen_texts, used_words = [], set(), 0

    for _, row in candidates.iterrows():
        if row["rerank_score"] < max_score * min_score_ratio:
            continue

        text = row["chunk_text"]
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        if normalized in seen_texts:
            continue

        chunk_words = len(text.split())

        if used_words + chunk_words > word_budget:
            remaining = word_budget - used_words
            if remaining > 50:
                row = row.copy()
                row["chunk_text"] = " ".join(text.split()[:remaining])
                selected_rows.append(row)
                used_words += remaining
            break

        selected_rows.append(row)
        seen_texts.add(normalized)
        used_words += chunk_words

        if len(selected_rows) >= max_context_chunks:
            break

    selected_df = pd.DataFrame(selected_rows)

    context_blocks = [
        f"[Source {i+1}] rerank_score={row['rerank_score']:.3f}\n{row['chunk_text']}"
        for i, row in selected_df.iterrows()
    ]

    return {
        "query": query,
        "selected_df": selected_df,
        "context_text": "\n\n".join(context_blocks),
        "num_sources": len(selected_df),
        "used_words": used_words,
    }


def build_query(prediction):
    keywords = QUERY_MAP.get(prediction, [])
    query = prediction
    if keywords:
        query += " " + " ".join(keywords)
    return query


def confidence_label(score):
    if score >= 2.5:
        return "High"
    elif score >= 1.0:
        return "Medium"
    return "Low"


# ==================================================================
# PROMPT BUILDER  (V5 - realistic meals)
# ==================================================================

def build_prompt(prediction: str, context: str) -> str:
    if not context or not context.strip():
        raise ValueError("Retrieved context is empty or invalid.")

    clean_prediction = prediction.strip() if prediction else "Not Specified"
    clean_context = context.strip()

    prompt = f"""You are a Senior Clinical Dietitian and Evidence-Based Communication Agent.

Your primary objective is to generate an accurate, realistic, and patient-friendly nutrition report based **STRICTLY AND EXCLUSIVELY** on the retrieved scientific context provided inside <context> tags.

============================================================
CORE OPERATIONAL GUARDRAILS (STRICT COMPLIANCE REQUIRED)
============================================================

### 1. ABSOLUTE CONTEXT GROUNDING (ZERO HALLUCINATION)
- Extract facts **ONLY** from the provided `<context>`.
- **STRICTLY FORBIDDEN**: Do NOT use internal medical knowledge, general diabetes guidelines, or unmentioned nutrition facts.
- **OMISSION RULE**: If a nutrient, food, or lifestyle factor is missing in the `<context>`, **completely omit it**. Do NOT assume or estimate.

### 2. REALISTIC MEAL PLAN & NEUTRAL BASE RULE (CRITICAL)
- **Primary Ingredients**: Meals MUST prominently incorporate the "Recommended Foods" mentioned in `<context>`.
- **Neutral Food Bases (Allowed)**: To make meals realistic and edible, you MAY combine recommended foods with standard neutral staples (e.g., leafy greens, water, eggs, olive oil) ONLY IF they do NOT violate the "Foods to Limit" section.
- **Strict Avoidance**: You MUST ABSOLUTELY AVOID any items listed under "Foods to Limit".
- **Nutritional Truthfulness**: Assign health benefits in the meal plan ONLY to the exact food items mentioned in the context.

### 3. DYNAMIC GOLDEN RULES (NO REDUNDANCY)
- Generate **up to 5 distinct Golden Rules** based ONLY on unique facts in `<context>`.
- **NO REPETITION**: If the context only contains 2 or 3 distinct pieces of advice, output ONLY 2 or 3 rules. **NEVER repeat the same rule in different words** just to reach 5.

### 4. SOURCE EXTRACTION & METADATA CLEANING
- Extract **ONLY** human-readable author citations (e.g., "Reaven et al. (21)", "American Diabetes Association (20)").
- **CRITICAL**: Never output technical metadata, search scores, vectors, rerank scores, or document IDs (e.g., `rerank_score`, `score: 0.89`).
- **FALLBACK**: If no human-readable authors exist in the context, write EXACTLY:
  *"Source: Retrieved clinical document (specific authors not detailed in the provided text)."*

============================================================
INPUT DATA
============================================================
<patient_condition>
{clean_prediction}
</patient_condition>

<context>
{clean_context}
</context>

============================================================
REQUIRED OUTPUT FORMAT (MARKDOWN)
============================================================
Generate the report using this EXACT structure:

# 🩺 Diabetes Personalized Nutrition Report

---

## 1. 🧬 Patient Condition Overview
[Summarize only the condition mentioned in patient_condition/context. If absent, write: "Not described in the retrieved document."]

---

## 2. 📚 Scientific Evidence Summary
[Bullet points derived ONLY from <context>. Skip unmentioned topics.]

---

## 3. 🔗 Evidence-to-Action Translation Bridge
| Scientific Basis (from Context) | Actionable Recommendation |
| :--- | :--- |
[Populate table rows ONLY supported by <context>]

---

## 4. ✅ Your Daily Action Plan
[Numbered actionable steps derived directly from the table above]

---

## 5. 🍽️ Food Guide (Recommended vs. Limit)
| Recommended Foods | Foods to Limit |
| :--- | :--- |
[ONLY include foods explicitly named in <context>]

---

## 6. 🏃 Practical Lifestyle & Daily Habits
- **⏰ Meal Timing**: [Specific advice from context. **Fallback:** "Not specifically detailed in the retrieved document. Please consult your dietitian for personalized advice on this topic."]
- **🍳 Cooking Methods**: [Advice from context or Fallback]
- **🛒 Grocery Shopping**: [Advice from context or Fallback]
- **🍽️ Eating Outside**: [Advice from context or Fallback]
- **💧 Hydration**: [Advice from context or Fallback]

---

## 7. 🥗 Sample One-Day Meal Plan
*(Combine Recommended Foods with neutral staples to create realistic meals. Attribute health benefits ONLY to the correct source).*

- **Breakfast**: [Meal Item]
  *Why?* [Direct context justification]
- **Morning Snack**: [Meal Item]
  *Why?* [Direct context justification]
- **Lunch**: [Meal Item]
  *Why?* [Direct context justification]
- **Afternoon Snack**: [Meal Item]
  *Why?* [Direct context justification]
- **Dinner**: [Meal Item]
  *Why?* [Direct context justification]

---

## 8. 🧠 Quick Reference Card (Top Golden Rules)
*(Provide only unique rules derived from context. Do not duplicate rules).*
- 🥇 [Rule 1]
- 🥈 [Rule 2]
- 🥉 [Rule 3 (if applicable)]
- 4️⃣ [Rule 4 (if applicable)]
- 5️⃣ [Rule 5 (if applicable)]

---

## 9. 📖 Evidence Sources
[Human-readable author citations or the exact Fallback statement]

---

## 10. ⚠️ Important Disclaimer
This report is generated based on retrieved clinical evidence for informational purposes only and does not constitute medical advice. Please consult your physician or registered dietitian before altering your diet or medication plan.
"""
    return prompt


# ==================================================================
# LLM GENERATION (OpenRouter-compatible chat completions API)
# ==================================================================

def generate_answer(prompt, api_key, model, temperature=0.1, max_tokens=1200):
    """Call an OpenRouter-compatible /chat/completions endpoint.

    The API key is only ever read from the in-session sidebar field
    (or an OPENROUTER_API_KEY environment variable) and is never
    written to disk or logged.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    start = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        elapsed = time.time() - start
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()
        usage = result.get("usage", {})
        return {
            "answer": answer or "The model returned an empty response.",
            "elapsed": elapsed,
            "usage": usage,
            "error": None,
        }
    except Exception as e:
        return {"answer": None, "elapsed": time.time() - start, "usage": {}, "error": str(e)}


def evaluate_answer(embedding_model, answer, context_text):
    if not answer or not context_text:
        return None
    embeddings = embedding_model.encode(
        [answer, context_text], convert_to_numpy=True, normalize_embeddings=True
    )
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    if similarity >= 0.80:
        quality = "Excellent"
    elif similarity >= 0.60:
        quality = "Good"
    elif similarity >= 0.40:
        quality = "Moderate"
    else:
        quality = "Poor"
    return {"similarity": float(similarity), "quality": quality}


# ==================================================================
# SIDEBAR - CONFIGURATION
# ==================================================================

st.sidebar.title("🩺 Configuration")

st.sidebar.subheader("1. Data source")
uploaded_csv = st.sidebar.file_uploader(
    f"Upload chunk CSV (defaults to '{DEFAULT_CSV_NAME}' if bundled)",
    type=["csv"],
)

csv_bytes = None
if uploaded_csv is not None:
    csv_bytes = uploaded_csv.read()
else:
    import os
    if os.path.exists(DEFAULT_CSV_NAME):
        with open(DEFAULT_CSV_NAME, "rb") as f:
            csv_bytes = f.read()

st.sidebar.subheader("2. LLM backend (OpenRouter)")
api_key = st.sidebar.text_input(
    "OpenRouter API key",
    type="password",
    help="Read only for this session. Never stored or logged. "
         "Get a key at https://openrouter.ai/keys",
)
model_choice = st.sidebar.selectbox("Model", DEFAULT_MODELS, index=0)
custom_model = st.sidebar.text_input("...or custom model id (optional)")
active_model = custom_model.strip() or model_choice

st.sidebar.subheader("3. Retrieval settings")
tfidf_weight = st.sidebar.slider("TF-IDF weight", 0.0, 1.0, 0.34, 0.01)
bm25_weight = st.sidebar.slider("BM25 weight", 0.0, 1.0, 0.33, 0.01)
semantic_weight = st.sidebar.slider("Semantic weight", 0.0, 1.0, 0.33, 0.01)
retrieval_k = st.sidebar.slider("Candidates retrieved (k)", 10, 80, 40, 5)
rerank_top_n = st.sidebar.slider("Chunks kept after reranking", 3, 20, 10, 1)
word_budget = st.sidebar.slider("Context word budget", 300, 3000, 1200, 100)
confidence_threshold = st.sidebar.slider(
    "Minimum rerank score to answer (else abstain)", 0.0, 5.0, 3.0, 0.1
)

# ==================================================================
# MAIN
# ==================================================================

st.title("🩺 Diabetes Nutrition RAG Assistant")
st.caption(
    "Hybrid retrieval (TF-IDF + BM25 + embeddings) → Cross-Encoder reranking "
    "→ grounded nutrition report generation."
)

if csv_bytes is None:
    st.warning(
        f"No chunk data found. Upload a chunk CSV in the sidebar "
        f"(a file with a `chunk_text` column), or bundle `{DEFAULT_CSV_NAME}` "
        f"next to this script."
    )
    st.stop()

with st.spinner("Loading chunk data and building retrieval indices..."):
    chunks_df = load_chunks_csv(csv_bytes)
    texts = tuple(chunks_df["chunk_text"].tolist())
    csv_hash = hashlib.md5(csv_bytes).hexdigest()

    tfidf_vectorizer, tfidf_matrix = build_tfidf_index(texts)
    bm25 = build_bm25_index(texts)
    embedding_model = load_embedding_model()
    embedding_matrix = build_embedding_matrix(embedding_model, texts, csv_hash)
    cross_encoder = load_cross_encoder()

st.success(f"Loaded {len(chunks_df)} chunks and built all retrieval indices.")

col1, col2 = st.columns([2, 1])

with col1:
    condition = st.selectbox("Patient condition", list(QUERY_MAP.keys()))
    custom_question = st.text_area(
        "Optional: specific question (overrides the default condition query)",
        placeholder="e.g. What are the current dietary recommendations for "
                    "Type 2 diabetes, including carbohydrate intake, dietary "
                    "fat, fibre and weight management?",
        height=100,
    )

with col2:
    st.markdown("**Retrieval weights**")
    st.write(f"TF-IDF: {tfidf_weight:.2f} | BM25: {bm25_weight:.2f} | "
             f"Semantic: {semantic_weight:.2f}")

run = st.button("Generate Nutrition Report", type="primary")

if run:
    if not api_key:
        st.error("Please enter an OpenRouter API key in the sidebar.")
        st.stop()

    question = custom_question.strip() or build_query(condition)

    with st.spinner("Retrieving relevant evidence (hybrid search)..."):
        results = retrieve_top_k_hybrid(
            query=question,
            tfidf_vectorizer=tfidf_vectorizer,
            tfidf_matrix=tfidf_matrix,
            bm25=bm25,
            embedding_model=embedding_model,
            embedding_matrix=embedding_matrix,
            chunks_df=chunks_df,
            tfidf_weight=tfidf_weight,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            k=retrieval_k,
        )

    with st.spinner("Reranking candidates with cross-encoder..."):
        reranked = rerank_candidates(cross_encoder, question, results, top_n=rerank_top_n)

    context = build_context_package(
        query=question, reranked_df=reranked,
        max_context_chunks=rerank_top_n, word_budget=word_budget,
    )

    best_score = float(reranked.iloc[0]["rerank_score"]) if len(reranked) else 0.0

    if best_score < confidence_threshold:
        st.warning(
            f"Low-confidence retrieval (top rerank score {best_score:.2f} < "
            f"threshold {confidence_threshold:.2f}). No reliable answer could "
            f"be generated from the retrieved document."
        )
    else:
        with st.spinner(f"Generating report with {active_model}..."):
            prompt = build_prompt(prediction=condition, context=context["context_text"])
            gen = generate_answer(
                prompt=prompt, api_key=api_key, model=active_model,
                temperature=0.1, max_tokens=1200,
            )

        if gen["error"]:
            st.error(f"LLM generation failed: {gen['error']}")
        else:
            answer = gen["answer"]
            st.markdown(answer)

            st.divider()
            st.subheader("📊 Diagnostics")

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Sources used", context["num_sources"])
            d2.metric("Context words", context["used_words"])
            d3.metric("Top rerank score", f"{best_score:.3f}")
            d4.metric("Confidence", confidence_label(best_score))

            grounding = evaluate_answer(embedding_model, answer, context["context_text"])
            if grounding:
                g1, g2, g3 = st.columns(3)
                g1.metric("Answer-context similarity", f"{grounding['similarity']:.3f}")
                g2.metric("Grounding quality", grounding["quality"])
                g3.metric("Generation time", f"{gen['elapsed']:.1f}s")

            with st.expander("Retrieved & reranked source chunks"):
                st.dataframe(
                    reranked[["chunk_id", "rerank_score", "chunk_text"]],
                    use_container_width=True,
                )

            with st.expander("Raw context passed to the LLM"):
                st.text(context["context_text"])

st.divider()
st.caption(
    "⚠️ This tool generates informational content from retrieved clinical "
    "literature only. It is not medical advice — consult a physician or "
    "registered dietitian before changing any diet or medication plan."
)
