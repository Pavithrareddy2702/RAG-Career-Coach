from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pypdf import PdfReader
import docx2txt


def read_uploaded_file(uploaded_file) -> str:
    """Read txt, pdf, or docx files uploaded from Streamlit and return their text content."""
    if uploaded_file is None:
        return ""

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if suffix not in (".pdf", ".docx"):
        raise ValueError("Unsupported file type. Please upload a .txt, .pdf, or .docx file.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        if suffix == ".pdf":
            reader = PdfReader(tmp_path)
            text = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(text)

        # suffix == ".docx"
        return docx2txt.process(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)