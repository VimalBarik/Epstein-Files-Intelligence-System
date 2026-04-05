import numpy as np


model = None


def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer

        print("[INFO] Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[INFO] Embedding model ready")
    return model



def embed_texts(texts, batch_size=32):
    
    if not texts:
        return np.array([])

    embeddings = get_model().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True   
    )

    return embeddings



def embed_query(text):
   
    embedding = get_model().encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding



def embed_chunks(chunks):
    
    texts = [c["content"] for c in chunks]
    return embed_texts(texts)



if __name__ == "__main__":
    sample_texts = [
        "Jeffrey Epstein met with several individuals.",
        "This is a financial document.",
        "A court transcript of a witness."
    ]

    embeddings = embed_texts(sample_texts)

    print("Shape:", embeddings.shape)
    print("First vector:", embeddings[0][:10])
