import numpy as np


def _get_device():
    """Pick the best available device: CUDA > MPS > CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


DEVICE = _get_device()

model = None


def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[INFO] Loading embedding model on device: {DEVICE}")
        model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEVICE)
        print("[INFO] Embedding model ready")
    return model


def embed_texts(texts, batch_size=16):
    if not texts:
        return np.array([])
    embeddings = get_model().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=DEVICE,
    )
    return embeddings


def embed_query(text):
    return get_model().encode(
        [f"query: {text}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=DEVICE,
    )


def embed_chunks(chunks):
    texts = [f"passage: {c['content']}" for c in chunks]
    return embed_texts(texts)
