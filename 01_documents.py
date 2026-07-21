"""
01_documents.py
================
Stage 1 — PDF Loading

Source: notebook cells 4, 5, 7 (adapted_rag_pipeline_diabetes_nutrition.ipynb)

Loads the diabetes nutrition PDF with Docling and returns the raw extracted
text plus basic size stats. This is the first stage of the pipeline:

    PDF -> raw_text (this file)
        -> cleaned_text (02_preprocessing.py)
        -> chunks (03_chunking.py)
        -> indexes (04_vector_representation.py)
        -> persisted indexes (05_create_chroma_store.py)
        -> retrieval (06_retrieve_context.py)
        -> prompting / answer (07_prompting.py)
"""

from docling.document_converter import DocumentConverter


# ==================================================
# Config
# ==================================================

# Path to the source PDF. Update this to point at your local copy.
PDF_PATH = r"C:\Users\Admin\Desktop\safaa samy second term download\download4\New folder (3) - Copy\Nutritional Management of Diabetes Mellitus - 2003 - Frost.pdf"


# ==================================================
# Initialize Docling Converter
# ==================================================

converter = DocumentConverter()


# ==================================================
# PDF Loader
# ==================================================

def load_pdf_document(pdf_path):
    """
    Convert a PDF into raw text using Docling and return a summary dict.
    """
    result = converter.convert(pdf_path)

    doc = result.document

    text_parts = []

    for item, _ in doc.iterate_items():

        if hasattr(item, "text") and item.text:
            text_parts.append(item.text.strip())

    raw_text = "\n".join(text_parts)

    return {
        "source_file": pdf_path,
        "raw_text": raw_text,
        "char_count": len(raw_text),
        "word_count": len(raw_text.split())
    }


# ==================================================
# Run Loader
# ==================================================

if __name__ == "__main__":

    pdf_document = load_pdf_document(PDF_PATH)

    print("=" * 60)
    print("PDF DOCUMENT SUMMARY")
    print("=" * 60)

    print("Characters:", pdf_document["char_count"])
    print("Words:", pdf_document["word_count"])

    print("\nFIRST 1000 CHARACTERS")
    print("=" * 60)
    print(pdf_document["raw_text"][:1000])
