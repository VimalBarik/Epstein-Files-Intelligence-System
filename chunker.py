import json
import os
from tqdm import tqdm

from project_paths import CHUNKS_JSON, FINAL_DATA_JSON


def split_into_paragraphs(text):
    
    if not text:
        return []

    paragraphs = text.split("\n")

    
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


def chunk_text(text, max_chars=800, overlap=100):
    
    paragraphs = split_into_paragraphs(text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        
        if len(current_chunk) + len(para) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())

            
            if overlap > 0 and chunks:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + " " + para
            else:
                current_chunk = para
        else:
            current_chunk += " " + para

    
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks



def create_chunks(
    input_path=os.fspath(FINAL_DATA_JSON),
    output_path=os.fspath(CHUNKS_JSON),
    max_chars=1200,
    overlap=200
):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_chunks = []

    progress = tqdm(data, desc="Creating chunks", unit="item")

    for item in progress:
        text = item.get("content", "").strip()

        if not text or len(text) < 30:
            continue  

        chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)

        for c in chunks:
            if len(c) < 30:
                continue

            all_chunks.append({
                "file": item.get("file"),
                "page": item.get("page"),
                "type": item.get("type"),
                "content": c
            })

        progress.set_postfix({"chunks": len(all_chunks)})

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Created {len(all_chunks)} chunks")
    print(f"Saved to: {output_path}")



if __name__ == "__main__":
    create_chunks()
