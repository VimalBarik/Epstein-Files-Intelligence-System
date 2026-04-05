import re


YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def build_timeline(chunks):
    timeline = []

    for chunk in chunks:
        text = chunk.get("content", "")
        match = YEAR_RE.search(text)
        if not match:
            continue

        timeline.append({
            "year": int(match.group(1)),
            "file": chunk.get("file", "Unknown"),
            "page": chunk.get("page"),
            "text": text[:300].strip(),
        })

    timeline.sort(key=lambda item: (item["year"], item["file"], item.get("page") or 0))
    return timeline
