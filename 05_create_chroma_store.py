"""
05_create_chroma_store.py
==========================
Stage 5 — Persist the retrieval indexes to disk

IMPORTANT NAMING NOTE
----------------------
The original notebook (adapted_rag_pipeline_diabetes_nutrition.ipynb) does
NOT use ChromaDB anywhere. All three indexes (TF-IDF, BM25, embeddings) are
built and used purely in-memory inside the notebook session.

Per your decision to stick to the notebook exactly (no Chroma), this file
keeps its original filename/position in the pipeline but its job is
adapted to what the notebook actually needs at this stage: saving the
indexes built in 04_vector_representation.py to disk, so
06_retrieve_context.py can load them without re-computing everything
(re-embedding the whole book, refitting TF-IDF, etc.) every time.

If you ever want a *real* Chroma vector store instead, that would require
adding the `chromadb` package and rewriting this file to call
`chromadb.Client().create_collection(...)` — that's new code, not
something from the notebook, so it wasn't added here.
"""

import pickle
import numpy as np
import pandas as pd

from importlib.machinery import SourceFileLoader

_vec = SourceFileLoader("stage4_vector_representation", "04_vector_representation.py").load_module()

INDEX_DIR = "indexes"


def save_indexes(tfidf_vectorizer, tfidf_matrix, bm25, embedding_model_name, embedding_matrix):
    import os
    os.makedirs(INDEX_DIR, exist_ok=True)

    with open(f"{INDEX_DIR}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)

    with open(f"{INDEX_DIR}/tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)

    with open(f"{INDEX_DIR}/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    with open(f"{INDEX_DIR}/embedding_model_name.txt", "w") as f:
        f.write(embedding_model_name)

    np.save(f"{INDEX_DIR}/embedding_matrix.npy", embedding_matrix)

    print("=" * 60)
    print("INDEXES SAVED")
    print("=" * 60)
    print(f"Location: {INDEX_DIR}/")
    print("- tfidf_vectorizer.pkl")
    print("- tfidf_matrix.pkl")
    print("- bm25.pkl")
    print("- embedding_model_name.txt")
    print("- embedding_matrix.npy")


def load_indexes():
    from sentence_transformers import SentenceTransformer

    with open(f"{INDEX_DIR}/tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    with open(f"{INDEX_DIR}/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(f"{INDEX_DIR}/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    with open(f"{INDEX_DIR}/embedding_model_name.txt", "r") as f:
        embedding_model_name = f.read().strip()

    embedding_model = SentenceTransformer(embedding_model_name)
    embedding_matrix = np.load(f"{INDEX_DIR}/embedding_matrix.npy")

    return tfidf_vectorizer, tfidf_matrix, bm25, embedding_model, embedding_matrix


if __name__ == "__main__":

    chunks_df = pd.read_csv("semantic_chunks_final.csv", encoding="utf-8-sig")
    texts = chunks_df["chunk_text"].tolist()

    tfidf_vectorizer, tfidf_matrix = _vec.build_tfidf_index(texts)

    tokenized_docs = [_vec.simple_tokenize(text) for text in texts]
    bm25 = _vec.MiniBM25(tokenized_docs)

    embedding_model_name = "all-MiniLM-L6-v2"
    embedding_model, embedding_matrix = _vec.build_embedding_index(
        chunks_df, model_name=embedding_model_name
    )

    save_indexes(tfidf_vectorizer, tfidf_matrix, bm25, embedding_model_name, embedding_matrix)
