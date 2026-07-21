"""
04_vector_representation.py
============================
Stage 4 — Build the TF-IDF, BM25 and Sentence-Embedding indexes

Source: notebook cells 33, 34, 36, 38, 41

Reads semantic_chunks_final.csv (from 03_chunking.py) and builds the three
retrieval indexes used later by the hybrid retriever. This file only
*builds* the indexes in memory; 05_create_chroma_store.py persists them
to disk so 06_retrieve_context.py doesn't have to rebuild them every run.
"""

import re
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


# ==================================================
# Helper Functions
# ==================================================

def get_metadata_value(row, key):
    meta = row.get("metadata")
    if isinstance(meta, dict):
        return meta.get(key)
    return None


def simple_tokenize(text):
    """Simple tokenizer shared by BM25 and hybrid scoring."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def min_max_normalize(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    lo = scores.min()
    hi = scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


# ==================================================
# TF-IDF Index
# ==================================================

def build_tfidf_index(texts):

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        max_features=30000,
        sublinear_tf=True,
        norm="l2",
        dtype="float32"
    )

    matrix = vectorizer.fit_transform(texts)

    print("=" * 60)
    print("TF-IDF INDEX SUMMARY")
    print("=" * 60)
    print(f"Documents       : {len(texts)}")
    print(f"Vocabulary Size : {len(vectorizer.vocabulary_):,}")
    print(f"Matrix Shape    : {matrix.shape}")
    print(f"Non-zero Terms  : {matrix.nnz:,}")

    sparsity = (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100
    print(f"Sparsity        : {sparsity:.2f}%")

    return vectorizer, matrix


# ==================================================
# BM25 Index
# ==================================================

class MiniBM25:

    def __init__(self, tokenized_docs, k1=1.5, b=0.75):

        self.k1 = k1
        self.b = b

        self.docs = tokenized_docs
        self.N = len(tokenized_docs)

        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avgdl = np.mean(self.doc_lens)

        self.term_freqs = [Counter(doc) for doc in tokenized_docs]

        self.df = Counter()
        for doc in tokenized_docs:
            self.df.update(set(doc))

        self.idf = {
            term: np.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for term, df in self.df.items()
        }

        print("=" * 60)
        print("BM25 INDEX SUMMARY")
        print("=" * 60)
        print(f"Documents      : {self.N}")
        print(f"Vocabulary     : {len(self.df):,}")
        print(f"Average Length : {self.avgdl:.1f} words")

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

                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / self.avgdl)

                scores[i] += (idf * tf * (self.k1 + 1)) / denom

        return scores


# ==================================================
# Semantic Embedding Index
# ==================================================

def build_embedding_index(chunks_df, model_name="all-MiniLM-L6-v2"):

    print("=" * 60)
    print("BUILDING EMBEDDING INDEX")
    print("=" * 60)

    model = SentenceTransformer(model_name)

    texts = chunks_df["chunk_text"].tolist()

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print(f"Embedding Model : {model_name}")
    print(f"Documents       : {len(texts)}")
    print(f"Embedding Shape : {embeddings.shape}")

    return model, embeddings


# ==================================================
# Run
# ==================================================

if __name__ == "__main__":

    chunks_df = pd.read_csv("semantic_chunks_final.csv", encoding="utf-8-sig")

    texts = chunks_df["chunk_text"].tolist()

    tfidf_vectorizer, tfidf_matrix = build_tfidf_index(texts)

    tokenized_docs = [simple_tokenize(text) for text in texts]
    bm25 = MiniBM25(tokenized_docs)

    embedding_model, embedding_matrix = build_embedding_index(chunks_df)

    print("\nAll three indexes built. Run 05_create_chroma_store.py to persist them to disk.")
