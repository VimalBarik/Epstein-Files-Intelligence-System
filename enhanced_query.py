from query import search


def filtered_search(query, k=8, file_filter=None, type_filter=None):
    results = search(
        query,
        k=k,
        file_filter=file_filter,
        type_filter=type_filter,
    )
    return results
