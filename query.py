import json
import os
import pickle
from collections import Counter

import requests

from chunker import chunk_text
from project_paths import CHUNKS_JSON, FAISS_CHUNKS, FAISS_INDEX, FINAL_DATA_JSON


def _load_faiss():
    try:
        import faiss
        from embedder import embed_query
    except Exception:
        return None

    if not FAISS_INDEX.exists() or not FAISS_CHUNKS.exists():
        return None

    index = faiss.read_index(os.fspath(FAISS_INDEX))
    with open(FAISS_CHUNKS, "rb") as file_obj:
        chunks = pickle.load(file_obj)

    return index, chunks, embed_query


def load_chunks():
    if CHUNKS_JSON.exists():
        with open(CHUNKS_JSON, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    if FINAL_DATA_JSON.exists():
        with open(FINAL_DATA_JSON, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)

        chunks = []
        for item in data:
            for piece in chunk_text(item.get("content", "")):
                if len(piece.strip()) < 30:
                    continue
                chunks.append({
                    "file": item.get("file"),
                    "page": item.get("page"),
                    "type": item.get("type", "text"),
                    "content": piece.strip(),
                    "source": item.get("source"),
                    "image_path": item.get("image_path"),
                })
        return chunks

    return []


def load_index():
    loaded = _load_faiss()
    if loaded is not None:
        index, chunks, embed_query = loaded
        return {
            "mode": "faiss",
            "index": index,
            "chunks": chunks,
            "embed_query": embed_query,
        }

    return {
        "mode": "keyword",
        "index": None,
        "chunks": load_chunks(),
        "embed_query": None,
    }


def _keyword_score(query, text):
    query_terms = [term.lower() for term in query.split() if len(term) > 2]
    if not query_terms:
        return 0

    text_lower = (text or "").lower()
    counts = Counter(text_lower.split())
    score = 0

    for term in query_terms:
        score += counts.get(term, 0)
        if term in text_lower:
            score += 2

    return score


def search(query, k=5):
    store = load_index()
    chunks = store["chunks"]

    if not chunks:
        return []

    if store["mode"] == "faiss":
        query_vec = store["embed_query"](query)
        _, indices = store["index"].search(query_vec, k)
        return [chunks[idx] for idx in indices[0] if 0 <= idx < len(chunks)]

    scored = []
    for chunk in chunks:
        score = _keyword_score(query, chunk.get("content", ""))
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:k]]


def build_context(results, max_chars=4000):
    context = ""

    for item in results:
        chunk = f"[{item.get('file', 'Unknown')} | Page {item.get('page')}]\n{item.get('content', '')}\n\n"
        if len(context) + len(chunk) > max_chars:
            break
        context += chunk

    return context.strip()


def ask_ollama(prompt, model="llama3", timeout=60):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("response", "").strip()


def _fallback_answer(query, results):
    if not results:
        return "I could not find any matching indexed text. Build the dataset first, then try again."

    snippets = []
    for item in results[:3]:
        snippets.append(
            f"- {item.get('file', 'Unknown')} page {item.get('page', '?')}: {item.get('content', '')[:280].strip()}"
        )

    joined = "\n".join(snippets)
    return (
        "Ollama is unavailable, so here are the closest matching passages for your question:\n\n"
        f"{joined}"
    )


def answer_question(query):
    results = search(query, k=8)
    context = build_context(results)

    if not context:
        return "I could not find that in the indexed documents.", results

    prompt = f"""
You are a factual assistant. Answer ONLY using the provided context.

If the answer is not in the context, say:
"I could not find that in the documents."

Context:
{context}

Question: {query}

Answer:
"""

    try:
        answer = ask_ollama(prompt)
        if not answer:
            answer = _fallback_answer(query, results)
    except Exception:
        answer = _fallback_answer(query, results)

    return answer, results


def chat():
    print("\nEpstein Files AI (type 'exit' to quit)\n")

    while True:
        query = input("Ask: ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        answer, sources = answer_question(query)
        print("\nAnswer:\n")
        print(answer)
        print("\nSources:\n")
        for source in sources[:3]:
            print(f"- {source.get('file')} | Page {source.get('page')}")


if __name__ == "__main__":
    chat()
