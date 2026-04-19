import re
from collections import Counter

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def keyword_score(query_tokens, doc_tokens):
    q_count = Counter(query_tokens)
    d_count = Counter(doc_tokens)

    score = 0
    for word in q_count:
        if word in d_count:
            score += min(q_count[word], d_count[word])
    return score


def entity_boost(query, text):
    # simple heuristic: boost if proper nouns match (names, places)
    query_words = set(query.split())
    text_words  = set(text.split())

    matches = [w for w in query_words if w.istitle() and w in text_words]
    return len(matches) * 2


def rerank(results, query):
    query_tokens = tokenize(query)

    scored = []
    for r in results:
        text = r.get("content", "")

        doc_tokens = tokenize(text)

        # scores
        kw_score = keyword_score(query_tokens, doc_tokens)
        ent_score = entity_boost(query, text)
        vec_score = r.get("score", 0)

        # weighted combination
        final_score = (
            0.5 * vec_score +   # Pinecone similarity
            0.3 * kw_score +    # keyword overlap
            0.2 * ent_score     # entity match
        )

        r["rerank_score"] = final_score
        scored.append(r)

    return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)