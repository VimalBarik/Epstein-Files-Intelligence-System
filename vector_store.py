import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

from project_paths import CHUNKS_JSON

# ---------------------------------------------------------------------------
# Pinecone setup
# ---------------------------------------------------------------------------

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX  = os.environ.get("PINECONE_INDEX", "epstein-index")

CHECKPOINT_DIR   = Path("data/checkpoints")
CHECKPOINT_EVERY = 10_000


def get_pinecone_index():
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set. Export a valid Pinecone API key before running.")
    from pinecone import Pinecone
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    return index


# ---------------------------------------------------------------------------
# Chunk loading
# ---------------------------------------------------------------------------

def load_chunks(path=os.fspath(CHUNKS_JSON)):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Checkpointing (so a crash mid-way doesn't lose progress)
# ---------------------------------------------------------------------------

def load_checkpoint():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = CHECKPOINT_DIR / "meta.json"
    if not meta_path.exists():
        return 0

    with open(meta_path) as f:
        meta = json.load(f)

    start = meta["next_index"]
    print(f"[INFO] Resuming from chunk {start}")
    return start


def save_checkpoint(next_index):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_DIR / "meta.json", "w") as f:
        json.dump({"next_index": next_index}, f)


def clear_checkpoints():
    import shutil
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)


# ---------------------------------------------------------------------------
# Build & upload
# ---------------------------------------------------------------------------

def build_vector_store(chunks_path=os.fspath(CHUNKS_JSON)):
    from embedder import embed_texts

    chunks = load_chunks(chunks_path)
    print(f"[INFO] Total chunks: {len(chunks)}")

    start_index = load_checkpoint()
    index       = get_pinecone_index()

    progress = tqdm(total=len(chunks), initial=start_index, desc="Uploading to Pinecone")

    batch_texts  = []
    batch_chunks = []

    for i in range(start_index, len(chunks)):
        chunk = chunks[i]
        batch_texts.append(f"passage: {chunk['content']}")
        batch_chunks.append((i, chunk))
        progress.update(1)

        if len(batch_texts) == CHECKPOINT_EVERY or i == len(chunks) - 1:
            # Embed
            embeddings = embed_texts(batch_texts, batch_size=16)

            # Build Pinecone vectors
            vectors = []
            for (idx, c), emb in zip(batch_chunks, embeddings):
                vectors.append({
                    "id": str(idx),
                    "values": emb.tolist(),
                    "metadata": {
                        "file":       c.get("file", ""),
                        "page":       str(c.get("page", "")),
                        "type":       c.get("type", "text"),
                        "content":    c.get("content", "")[:1000],
                        "source":     c.get("source", ""),
                        "image_path": c.get("image_path", ""),
                    },
                })

            # Upsert in sub-batches of 100 (Pinecone limit)
            for j in range(0, len(vectors), 100):
                index.upsert(vectors=vectors[j:j + 100])

            save_checkpoint(i + 1)
            print(f"[INFO] Upserted up to chunk {i + 1}")

            batch_texts  = []
            batch_chunks = []

    progress.close()
    clear_checkpoints()
    print("\n✅ Pinecone vector store ready!")


if __name__ == "__main__":
    build_vector_store()
