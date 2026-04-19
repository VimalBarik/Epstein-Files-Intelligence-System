import os
import re
from dotenv import load_dotenv
load_dotenv()
from reranker import rerank

# ---------------------------------------------------------------------------
# API keys — set these in your .env file or Streamlit secrets
# ---------------------------------------------------------------------------

GROQ_API_KEY     = os.environ.get("GROQ_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX   = os.environ.get("PINECONE_INDEX",  "epstein-index")


# ---------------------------------------------------------------------------
# Follow-up detection
# ---------------------------------------------------------------------------

_FOLLOWUP_PATTERNS = re.compile(
    r"^\s*("
    r"show (me )?(the )?(related |relevant )?(documents?|sources?|files?|pages?|evidence|results?)"
    r"|what (documents?|sources?|files?|pages?) (support|show|mention|relate)"
    r"|(give|list|find|get) (me )?(the )?(documents?|sources?|files?|pages?|evidence)"
    r"|where (is|are|can I find) (that|this|those|it)"
    r"|which (documents?|sources?|files?|pages?)"
    r"|more (details?|info|information|context)"
    r"|can you (show|find|give|list|tell me more)"
    r"|tell me more"
    r"|elaborate"
    r"|expand (on )?(that|this)"
    r"|what else"
    r"|and (what|where|who|how|why)"
    r"|also"
    r"|related (documents?|sources?|files?)"
    r")",
    re.IGNORECASE,
)

_SHORT_PROMPT_WORD_LIMIT = 8


def is_followup(current_query: str, history: list) -> bool:
    if not history:
        return False
    if _FOLLOWUP_PATTERNS.search(current_query):
        return True
    if len(current_query.split()) <= _SHORT_PROMPT_WORD_LIMIT:
        return True
    return False


def build_enriched_query(current_query: str, history: list) -> str:
    if not history or len(history) < 2:
        return current_query

    last_user      = ""
    last_assistant = ""
    for msg in reversed(history):
        if msg["role"] == "assistant" and not last_assistant:
            last_assistant = msg["content"]
        elif msg["role"] == "user" and not last_user and last_assistant:
            last_user = msg["content"]
            break

    topic_terms = " ".join(w for w in last_user.split() if len(w) > 3)
    return f"{topic_terms} {current_query}".strip()


# ---------------------------------------------------------------------------
# Pinecone search
# ---------------------------------------------------------------------------

_pinecone_index = None


def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(PINECONE_INDEX)
    return _pinecone_index


def search(query, k=5, file_filter=None, type_filter=None):
    from embedder import embed_query

    query_vec = embed_query(query)[0].tolist()
    index     = get_pinecone_index()

    # Fetch more than k so we can filter
    fetch_k  = max(k * 4, 30)
    response = index.query(
        vector=query_vec,
        top_k=fetch_k,
        include_metadata=True,
    )

    results = []
    for match in response["matches"]:
        meta = match["metadata"]

        if file_filter and file_filter.lower() not in meta.get("file", "").lower():
            continue
        if type_filter and meta.get("type") != type_filter:
            continue

        results.append({
            "file":       meta.get("file", ""),
            "page":       meta.get("page", ""),
            "type":       meta.get("type", "text"),
            "content":    meta.get("content", ""),
            "source":     meta.get("source", ""),
            "image_path": meta.get("image_path", ""),
            "score":      match["score"],
        })

        if len(results) >= k:
            break

    return results


def build_context(results, max_chars=4000):
    context = ""
    for item in results:
        chunk = f"[{item.get('file', 'Unknown')} | Page {item.get('page')}]\n{item.get('content', '')}\n\n"
        if len(context) + len(chunk) > max_chars:
            break
        context += chunk
    return context.strip()


# ---------------------------------------------------------------------------
# Groq LLM
# ---------------------------------------------------------------------------

def ask_groq(prompt):
    from groq import Groq
    client   = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    return response.choices[0].message.content.strip()


def _fallback_answer(results):
    if not results:
        return "I could not find any matching indexed text."

    snippets = []
    for item in results[:3]:
        snippets.append(
            f"- {item.get('file', 'Unknown')} page {item.get('page', '?')}: "
            f"{item.get('content', '')[:280].strip()}"
        )
    return (
        "The LLM is unavailable. Here are the closest matching passages:\n\n"
        + "\n".join(snippets)
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def answer_question(query, history=None):
    history = history or []

    search_query = (
        build_enriched_query(query, history)
        if is_followup(query, history)
        else query
    )

    initial_results = search(search_query, k=20)   # high recall
    reranked = rerank(initial_results, search_query)
    results = reranked[:8]                         # high precision
    context = build_context(results)

    if not context:
        return "I could not find that in the indexed documents.", results

    conversation_snippet = ""
    if history and is_followup(query, history):
        recent = [m for m in history if m["role"] in ("user", "assistant")][-4:]
        if recent:
            conversation_snippet = "Recent conversation:\n"
            for m in recent:
                role_label = "User" if m["role"] == "user" else "Assistant"
                conversation_snippet += f"{role_label}: {m['content'][:400]}\n"
            conversation_snippet += "\n"

    prompt = f"""You are a factual assistant analyzing the Epstein files.

Use ONLY the provided document context to answer.

Rules:
- If the exact answer is stated → return it clearly
- If partial evidence exists → infer cautiously and explain
- Do NOT say "not found" if there is relevant evidence
- Only say "not found" if the context is clearly unrelated
- If the question is a follow-up, use the conversation history to understand the topic
- Always cite the file name and page number when referencing evidence

{conversation_snippet}Document Context:
{context}

Current Question: {query}

Answer:"""

    try:
        answer = ask_groq(prompt)
        if not answer:
            answer = _fallback_answer(results)
    except Exception as e:
        print(f"[ERROR] Groq failed: {e}")
        answer = _fallback_answer(results)

    return answer, results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def chat():
    print("\nEpstein Files AI (type 'exit' to quit)\n")
    history = []

    while True:
        query = input("Ask: ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        history.append({"role": "user", "content": query})
        answer, sources = answer_question(query, history=history)
        history.append({"role": "assistant", "content": answer})

        print("\nAnswer:\n")
        print(answer)
        print("\nSources:\n")
        for source in sources[:3]:
            print(f"- {source.get('file')} | Page {source.get('page')}")


if __name__ == "__main__":
    chat()
