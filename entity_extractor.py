import re
from collections import defaultdict


ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")


def extract_entities(text):
    return {match.group(0).strip() for match in ENTITY_RE.finditer(text or "")}


def build_entity_graph(chunks):
    graph = defaultdict(set)

    for chunk in chunks:
        entities = sorted(extract_entities(chunk.get("content", "")))
        for entity in entities:
            for other in entities:
                if other != entity:
                    graph[entity].add(other)

    return dict(graph)

