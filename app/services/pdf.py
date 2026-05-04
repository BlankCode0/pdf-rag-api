import fitz  # PyMuPDF
from pathlib import Path


def extract_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.
    Returns the full text as a single string.
    Raises ValueError if the PDF has no extractable text (scanned image PDF).
    """
    doc = fitz.open(pdf_path)
    full_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            full_text.append(text)

    doc.close()

    if not full_text:
        raise ValueError(
            "No extractable text found. "
            "This PDF may be a scanned image — OCR not supported."
        )

    return "\n".join(full_text)