"""
03_chunking.py
===============
Stage 3 — Adaptive/Semantic Chunking, Metadata, Quality Check

Source: notebook cells 21, 22, 24, 29, 32

Takes cleaned_text (from 02_preprocessing.py) and produces chunks_df, saved
as semantic_chunks_final.csv for the next stage (04_vector_representation.py).
"""

import os
import re
import pandas as pd

from importlib.machinery import SourceFileLoader

_docs = SourceFileLoader("stage1_documents", "01_documents.py").load_module()


# ==================================================
# Semantic Topic-aware Chunking
# ==================================================

def semantic_chunk_markdown(
    text,
    target_words=180,
    overlap_words=30,
    min_chunk_words=80
):
    """
    Topic-aware semantic chunking.

    Improvements:
    1. Preserve section boundaries.
    2. Never merge different headings.
    3. Merge paragraphs only inside the same section.
    4. Small overlap.
    5. Split only oversized paragraphs.
    """

    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Build sections
    sections = []
    current_heading = "General"
    current_paragraphs = []

    for part in text.split("\n\n"):

        part = part.strip()

        if not part:
            continue

        if part.startswith("#"):

            if current_paragraphs:
                sections.append((current_heading, current_paragraphs))

            current_heading = part
            current_paragraphs = []
            continue

        current_paragraphs.append(part)

    if current_paragraphs:
        sections.append((current_heading, current_paragraphs))

    # Chunk each section independently
    chunks = []

    for heading, paragraphs in sections:

        current_chunk = []
        current_words = 0

        for para in paragraphs:

            block = heading + "\n\n" + para
            words = block.split()

            # Oversized paragraph
            if len(words) > target_words:

                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_words = 0

                stride = target_words - overlap_words

                for start in range(0, len(words), stride):
                    piece = words[start:start + target_words]
                    if len(piece) >= min_chunk_words:
                        chunks.append(" ".join(piece))

                continue

            # Normal merge
            if current_words + len(words) <= target_words:
                current_chunk.append(block)
                current_words += len(words)
            else:
                chunks.append("\n\n".join(current_chunk))

                overlap = " ".join(
                    " ".join(current_chunk).split()[-overlap_words:]
                )

                current_chunk = [overlap, block]
                current_words = len(overlap.split()) + len(words)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

    # DataFrame
    records = []

    for i, chunk in enumerate(chunks, start=1):
        records.append({
            "chunk_id": f"chunk_{i:04d}",
            "chunk_text": chunk,
            "word_count": len(chunk.split())
        })

    return pd.DataFrame(records)


# ==================================================
# Chunk Quality Validation — detect leftover PDF artifacts
# ==================================================

ARTIFACT_PATTERN = re.compile(r"(?:\s/[CF]\d+\s*){8,}")


def find_pdf_artifacts(chunks_df):
    """
    Scan chunks for repeated PDF extraction artifacts (e.g. font/char codes).
    """
    hits = []

    for idx, row in chunks_df.iterrows():
        text = row["chunk_text"]
        if ARTIFACT_PATTERN.search(text):
            hits.append(idx)

    return hits


# ==================================================
# Add Chunk Metadata
# ==================================================

def add_chunk_metadata(chunks_df, pdf_path, publication_year=2003):

    chunks_df = chunks_df.copy()

    chunks_df["metadata"] = chunks_df.apply(
        lambda row: {
            "publication_year": publication_year,
            "source_file": os.path.basename(pdf_path),
            "chunk_id": row["chunk_id"],
            "word_count": row["word_count"],
            "char_count": len(row["chunk_text"]),
        },
        axis=1
    )

    return chunks_df


# ==================================================
# Run
# ==================================================

if __name__ == "__main__":

    with open("cleaned_text.txt", "r", encoding="utf-8") as f:
        cleaned_text = f.read()

    chunks_df = semantic_chunk_markdown(
        cleaned_text,
        target_words=400,
        overlap_words=80,
        min_chunk_words=80
    )

    chunks_df["source_file"] = os.path.basename(_docs.PDF_PATH)
    chunks_df["publication_year"] = 2003

    if "char_count" not in chunks_df.columns:
        chunks_df["char_count"] = chunks_df["chunk_text"].apply(len)

    print("=" * 60)
    print("SEMANTIC CHUNKING SUMMARY")
    print("=" * 60)

    word_counts = chunks_df["word_count"]
    print(f"Generated Chunks    : {len(chunks_df)}")
    print(f"Average Words       : {word_counts.mean():.1f}")
    print(f"Median Words        : {word_counts.median():.1f}")
    print(f"Min Words           : {word_counts.min()}")
    print(f"Max Words           : {word_counts.max()}")

    # Quality check
    artifact_hits = find_pdf_artifacts(chunks_df)
    print(f"Chunks with PDF artifacts: {len(artifact_hits)}")

    # Metadata
    chunks_df = add_chunk_metadata(chunks_df, _docs.PDF_PATH, publication_year=2003)

    OUTPUT_PATH = "semantic_chunks_final.csv"
    chunks_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\nSaved {len(chunks_df)} chunks to: {OUTPUT_PATH}")
