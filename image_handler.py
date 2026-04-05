import fitz
import os
from PIL import Image
import pytesseract

import torch
from project_paths import IMAGES_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = None
model = None


def load_caption_model():
    global processor, model
    if processor is None or model is None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        print("[INFO] Loading BLIP model... (first time may take a while)")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model.to(device)
        print(f"[INFO] BLIP loaded on {device}")
    return processor, model


def clean_text(text):
    if not text:
        return ""
    return " ".join(text.split()).strip()


def image_has_text(image, threshold=20):
    try:
        text = pytesseract.image_to_string(image)
        return len(text.strip()) > threshold
    except:
        return False


def caption_image(image):
    try:
        loaded_processor, loaded_model = load_caption_model()
        inputs = loaded_processor(image, return_tensors="pt").to(device)
        out = loaded_model.generate(**inputs, max_new_tokens=50)
        caption = loaded_processor.decode(out[0], skip_special_tokens=True)
        return clean_text(caption)
    except Exception as e:
        print(f"[ERROR] Captioning failed: {e}")
        return ""


def extract_images_from_pdf(pdf_path, output_folder=os.fspath(IMAGES_DIR)):
    os.makedirs(output_folder, exist_ok=True)

    doc = fitz.open(pdf_path)
    image_data = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                image_path = os.path.join(
                    output_folder,
                    f"{os.path.basename(pdf_path)}_p{page_index+1}_{img_index}.png"
                )

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                image_data.append({
                    "page": page_index + 1,
                    "image_path": image_path
                })

            except Exception as e:
                print(f"[ERROR] Image extraction failed: {e}")

    return image_data


def process_images(pdf_path):
    results = []

    images = extract_images_from_pdf(pdf_path)

    for img_data in images:
        page = img_data["page"]
        path = img_data["image_path"]

        try:
            image = Image.open(path).convert("RGB")

            if image_has_text(image):
                continue

            caption = caption_image(image)

            if caption:
                results.append({
                    "page": page,
                    "content": caption,
                    "type": "image_description",
                    "image_path": path
                })

        except Exception as e:
            print(f"[ERROR] Processing image failed: {e}")

    return results


if __name__ == "__main__":
    pdf_path = "data/epstein files/sample.pdf"

    results = process_images(pdf_path)

    for r in results:
        print(f"\n--- Page {r['page']} ---")
        print(r["content"])
