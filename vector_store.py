import os
import faiss
import pickle
import numpy as np
from tqdm import tqdm

from project_paths import CHUNKS_JSON, FAISS_DIR


model = None


def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer

        print("[INFO] Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[INFO] Model loaded")
    return model


def load_chunks(path=os.fspath(CHUNKS_JSON)):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_embeddings(chunks):
    texts = [c["content"] for c in chunks]

    print(f"[INFO] Creating embeddings for {len(texts)} chunks...")
    embeddings = get_model().encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(embeddings)

    print(f"[INFO] FAISS index built with {index.ntotal} vectors")

    return index


def save_index(index, chunks, path=os.fspath(FAISS_DIR)):
    os.makedirs(path, exist_ok=True)

    faiss.write_index(index, os.path.join(path, "index.faiss"))

    with open(os.path.join(path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"[INFO] Index + metadata saved to {path}")


def build_vector_store(
    chunks_path=os.fspath(CHUNKS_JSON),
    output_path=os.fspath(FAISS_DIR),
):
    stage_progress = tqdm(total=4, desc="Building vector store", unit="stage")

    chunks = load_chunks(chunks_path)
    stage_progress.set_postfix({"step": "loaded_chunks", "count": len(chunks)})
    stage_progress.update(1)

    if not chunks:
        print("[ERROR] No chunks found!")
        stage_progress.close()
        return

    embeddings = create_embeddings(chunks)
    stage_progress.set_postfix({"step": "created_embeddings", "count": len(chunks)})
    stage_progress.update(1)

    index = build_faiss_index(embeddings)
    stage_progress.set_postfix({"step": "built_index", "vectors": index.ntotal})
    stage_progress.update(1)

    save_index(index, chunks, output_path)
    stage_progress.set_postfix({"step": "saved_index"})
    stage_progress.update(1)
    stage_progress.close()

    print("\n✅ Vector store ready!")


if __name__ == "__main__":
    build_vector_store()
