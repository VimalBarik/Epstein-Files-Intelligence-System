import os
import json
import fitz
from tqdm import tqdm

from extract_text import extract_digital_pdf
from ocr import extract_scanned_pdf, ocr_single_page
from image_handler import process_images
from project_paths import FINAL_DATA_JSON, PDF_DIR, RESULTS_JSON


def clean_text(text):
    if not text:
        return ""
    return " ".join(text.split()).strip()


def extract_mixed_pdf(pdf_path):
    results = []
    doc = fitz.open(pdf_path)

    for i, page in enumerate(doc):
        text = page.get_text("text")

        if text and len(text.strip()) > 50:
            results.append({
                "page": i + 1,
                "content": clean_text(text),
                "source": "digital"
            })
        else:
            ocr_text = ocr_single_page(page)

            if ocr_text:
                results.append({
                    "page": i + 1,
                    "content": clean_text(ocr_text),
                    "source": "ocr"
                })

    return results


def run_pipeline(
    results_json_path=os.fspath(RESULTS_JSON),
    pdf_folder=os.fspath(PDF_DIR),
    output_path=os.fspath(FINAL_DATA_JSON),
):
    with open(results_json_path, "r") as f:
        results = json.load(f)

    all_data = []
    progress = tqdm(results, desc="Running extraction pipeline", unit="pdf")

    for doc in progress:
        file_name = doc["file"]
        pdf_type = doc["type"]

        pdf_path = os.path.join(pdf_folder, file_name)
        progress.set_postfix({"file": file_name[:24], "type": pdf_type})

        try:
            if pdf_type == "digital":
                pages = extract_digital_pdf(pdf_path)

            elif pdf_type == "scanned":
                pages = extract_scanned_pdf(pdf_path)

            else:
                pages = extract_mixed_pdf(pdf_path)

            for p in pages:
                p["file"] = file_name
                p["content"] = clean_text(p["content"])
                p["type"] = "text"

                all_data.append(p)

            image_results = process_images(pdf_path)

            for img in image_results:
                img["file"] = file_name
                img["content"] = clean_text(img["content"])

                all_data.append(img)

        except Exception as e:
            tqdm.write(f"[ERROR] Failed processing {file_name}: {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"\n Pipeline complete!")
    print(f"Saved to: {output_path}")
    print(f"Total chunks: {len(all_data)}")


if __name__ == "__main__":
    run_pipeline()
