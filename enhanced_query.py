from query import search


def filtered_search(query, k=8, file_filter=None, type_filter=None):
    results = search(query, k=max(k * 3, 20))
    filtered = []

    file_filter_lower = file_filter.lower() if file_filter else None

    for item in results:
        if file_filter_lower and file_filter_lower not in item.get("file", "").lower():
            continue
        if type_filter and item.get("type") != type_filter:
            continue
        filtered.append(item)
        if len(filtered) >= k:
            break

    return filtered

