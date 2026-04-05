import pdfplumber

def clean_text(text):
    if not text:
        return ""

    text = text.replace("\x00", "")
    text = text.replace("\r", "\n")

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def extract_digital_pdf(pdf_path):
    results = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()

                if text:
                    cleaned = clean_text(text)

                    if cleaned:
                        results.append({
                            "page": i + 1,
                            "content": cleaned,
                            "source": "digital"
                        })

    except Exception as e:
        print(f"[ERROR] Failed to extract text from {pdf_path}: {e}")

    return results


if __name__ == "__main__":
    pdf_path = "data/epstein files/sample.pdf"

    pages = extract_digital_pdf(pdf_path)

    for p in pages:
        print(f"\n--- Page {p['page']} ---")
        print(p["content"][:500])
