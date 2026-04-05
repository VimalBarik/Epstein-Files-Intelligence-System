from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import re


def clean_text(text):
    if not text:
        return ""

    text = text.replace("\x0c", " ")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)

    return text.strip()


def ocr_image(image: Image.Image):
    try:
        text = pytesseract.image_to_string(image)
        return clean_text(text)
    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        return ""


def extract_scanned_pdf(pdf_path, dpi=300):
    results = []

    try:
        images = convert_from_path(pdf_path, dpi=dpi)

        for i, img in enumerate(images):
            text = ocr_image(img)

            if text:
                results.append({
                    "page": i + 1,
                    "content": text,
                    "source": "ocr"
                })

    except Exception as e:
        print(f"[ERROR] Failed OCR for {pdf_path}: {e}")

    return results


def ocr_single_page(page):
    try:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        text = pytesseract.image_to_string(img)
        return clean_text(text)

    except Exception as e:
        print(f"[ERROR] Page OCR failed: {e}")
        return ""


if __name__ == "__main__":
    pdf_path = "data/epstein files/sample.pdf"

    pages = extract_scanned_pdf(pdf_path)

    for p in pages:
        print(f"\n--- Page {p['page']} ---")
        print(p["content"][:500])
