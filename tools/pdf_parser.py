import fitz

def parse_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            full_text.append(f"--- Page {page_num} ---\n{text.strip()}")

    doc.close()

    return "\n\n".join(full_text)

