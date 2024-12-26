from typing import List

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Splits a large text into smaller chunks with optional overlap.

    Args:
        text: The input text to be chunked.
        chunk_size: The maximum size of each chunk in characters or tokens.
        overlap: The number of characters/tokens to overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks