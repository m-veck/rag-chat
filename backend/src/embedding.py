import re
from collections import Counter
from pathlib import Path
from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
    'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
    'not', 'no', 'so', 'as', 'if', 'then', 'than', 'about', 'which', 'who',
    'what', 'when', 'where', 'how', 'why', 'all', 'any', 'both', 'each',
    'more', 'most', 'other', 'some', 'such', 'into', 'also', 'just', 'up',
    'out', 'only', 'own', 'same', 'very',
}


_MIN_KEYWORD_LENGTH = 3

_KEYWORD_PREFIX = "Keywords: "


def extract_keywords(text: str, top_n: int = 10) -> list:
    """Extract top keywords from text using word-frequency with stop-word filtering."""
    words = re.findall(rf'\b[a-zA-Z]{{{_MIN_KEYWORD_LENGTH},}}\b', text.lower())
    words = [w for w in words if w not in _STOP_WORDS]
    counter = Counter(words)
    return [word for word, _ in counter.most_common(top_n)]


# ---------------------------------------------------------------------------
# Per-file-type chunking configuration
# ---------------------------------------------------------------------------

# chunk_size  – maximum characters per chunk
# chunk_overlap – characters shared between consecutive chunks
# separators  – ordered list of split points (first match wins)
_CHUNKING_CONFIGS = {
    '.pdf': {
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'separators': ['\n\n', '\n', '. ', ' ', ''],
    },
    '.docx': {
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'separators': ['\n\n', '\n', '. ', ' ', ''],
    },
    '.txt': {
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'separators': ['\n\n', '\n', '. ', ' ', ''],
    },
    '.csv': {
        'chunk_size': 500,
        'chunk_overlap': 50,
        'separators': ['\n', ',', ' ', ''],
    },
    '.xlsx': {
        'chunk_size': 500,
        'chunk_overlap': 50,
        'separators': ['\n', ',', ' ', ''],
    },
    '.xls': {
        'chunk_size': 500,
        'chunk_overlap': 50,
        'separators': ['\n', ',', ' ', ''],
    },
    '.json': {
        'chunk_size': 500,
        'chunk_overlap': 50,
        'separators': ['\n', '}', '{', ',', ' ', ''],
    },
}

_DEFAULT_CHUNKING_CONFIG = {
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'separators': ['\n\n', '\n', '. ', ' ', ''],
}


class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Smart chunking: each document is split with parameters chosen for its
        file type (size, overlap, and separators).  Falls back to the defaults
        for unknown extensions.
        """
        chunks = []
        for doc in documents:
            ext = Path(doc.metadata.get('source', '')).suffix.lower()
            cfg = _CHUNKING_CONFIGS.get(ext, _DEFAULT_CHUNKING_CONFIG)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg['chunk_size'],
                chunk_overlap=cfg['chunk_overlap'],
                length_function=len,
                separators=cfg['separators'],
            )
            chunks.extend(splitter.split_documents([doc]))
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

    def embed_chunks_with_keywords(self, chunks: List[Any]) -> tuple:
        """
        Keyword-augmented embedding: extract top keywords from each chunk and
        prepend them to the text before encoding.  This enriches the embedding
        vector with key-term signal without changing the stored chunk text.

        Returns:
            embeddings       – np.ndarray of shape (n_chunks, embedding_dim)
            keywords_per_chunk – list[list[str]], one keyword list per chunk
        """
        keywords_per_chunk = []
        augmented_texts = []

        for chunk in chunks:
            keywords = extract_keywords(chunk.page_content)
            keywords_per_chunk.append(keywords)
            if keywords:
                augmented_text = f"{_KEYWORD_PREFIX}{', '.join(keywords)}\n\n{chunk.page_content}"
            else:
                augmented_text = chunk.page_content
            augmented_texts.append(augmented_text)

        print(f"[INFO] Generating keyword-augmented embeddings for {len(augmented_texts)} chunks...")
        embeddings = self.model.encode(augmented_texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings, keywords_per_chunk

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings, keywords = emb_pipe.embed_chunks_with_keywords(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
    print("[INFO] Example keywords:", keywords[0] if keywords else [])