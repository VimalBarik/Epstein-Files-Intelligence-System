import os
import json
import fitz
from pdf2image import convert_from_path
import pytesseract
from tqdm import tqdm

class PDFTypeDetector:
    def __init__(self, text_threshold=50, density_threshold=0.01, use_ocr_fallback=True):
        self.text_threshold = text_threshold
        self.density_threshold = density_threshold
        self.use_ocr_fallback = use_ocr_fallback

    def analyze_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)

        total_text_length = 0
        total_area = 0
        image_count = 0
        page_stats = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            text_length = len(text.strip())
            total_text_length += text_length

            rect = page.rect
            page_area = rect.width * rect.height
            total_area += page_area

            images = page.get_images(full=True)
            num_images = len(images)
            image_count += num_images

            density = text_length / page_area if page_area > 0 else 0

            page_stats.append({
                "page": page_num + 1,
                "text_length": text_length,
                "density": density,
                "images": num_images
            })

        avg_density = total_text_length / total_area if total_area > 0 else 0

        return {
            "total_text_length": total_text_length,
            "avg_density": avg_density,
            "image_count": image_count,
            "pages": page_stats,
            "num_pages": len(doc)
        }

    def ocr_sample_check(self, pdf_path, max_pages=2):
        try:
            images = convert_from_path(pdf_path, first_page=1, last_page=max_pages)
            ocr_text = ""

            for img in images:
                ocr_text += pytesseract.image_to_string(img)

            return len(ocr_text.strip())
        except Exception as e:
            print(f"OCR fallback failed for {pdf_path}: {e}")
            return 0

    def classify(self, pdf_path, stats):
        text_length = stats["total_text_length"]
        density = stats["avg_density"]
        image_count = stats["image_count"]
        num_pages = stats["num_pages"]

        text_pages = sum(1 for p in stats["pages"] if p["text_length"] > self.text_threshold)
        image_pages = sum(1 for p in stats["pages"] if p["images"] > 0)

        text_ratio = text_pages / num_pages if num_pages else 0
        image_ratio = image_pages / num_pages if num_pages else 0

        if text_length < 100 and image_count > 0:
            if self.use_ocr_fallback:
                ocr_len = self.ocr_sample_check(pdf_path)
                if ocr_len > 200:
                    return "scanned"
            return "scanned"

        if text_ratio > 0.8 and density > self.density_threshold:
            return "digital"

        if text_ratio < 0.3 and image_ratio > 0.5:
            return "scanned"

        return "mixed"

    def detect(self, pdf_path):
        stats = self.analyze_pdf(pdf_path)
        pdf_type = self.classify(pdf_path, stats)

        return {
            "file": os.path.basename(pdf_path),
            "type": pdf_type,
            "stats": stats
        }


def process_folder(folder_path, output_json="results.json"):
    detector = PDFTypeDetector()
    results = []
    pdf_files = sorted(
        file for file in os.listdir(folder_path)
        if file.lower().endswith(".pdf")
    )

    progress = tqdm(pdf_files, desc="Classifying PDFs", unit="pdf")

    for file in progress:
        pdf_path = os.path.join(folder_path, file)

        try:
            result = detector.detect(pdf_path)
            results.append(result)
            progress.set_postfix({
                "file": file[:24],
                "type": result["type"],
                "pages": result["stats"]["num_pages"],
            })
        except Exception as e:
            tqdm.write(f"Error processing {file}: {e}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n Done! Results saved to {output_json}")


if __name__ == "__main__":
    folder_path = "epstein files"
    process_folder(folder_path)
