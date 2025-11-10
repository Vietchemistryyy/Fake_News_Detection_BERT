"""FAISS-backed Knowledge Base for retrieval (optional faiss, fallback to sklearn)

This module provides KnowledgeBaseFAISS which computes dense embeddings via
Sentence-Transformers and stores them in a FAISS index for fast nearest-neighbour
retrieval. If FAISS is not available, it falls back to sklearn's NearestNeighbors.

Notes:
- Requires `sentence-transformers` and (ideally) `faiss-cpu` installed.
- Save/load persists index, metadata and tokenizer/embeddings model name.
"""
from __future__ import annotations

import os
import pickle
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBT = True
except Exception:
    SentenceTransformer = None
    _HAS_SBT = False

from sklearn.neighbors import NearestNeighbors


class KnowledgeBaseFAISS:
    """Dense KB using Sentence-Transformers + FAISS.

    Basic usage:
        kb = KnowledgeBaseFAISS(embedding_model_name='sentence-transformers/all-MiniLM-L6-v2')
        kb.build_from_texts(texts, ids=optional_ids)
        kb.save('models/kb/')
        kb.load('models/kb/')
        docs = kb.retrieve('some query', top_k=5)

    If FAISS isn't installed this will use sklearn.NearestNeighbors (slower).
    """

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", normalize: bool = True, device: str = "cpu"):
        if not _HAS_SBT:
            raise ImportError(
                "Sentence-Transformers not found. Install with: pip install sentence-transformers"
            )

        self.embedding_model_name = embedding_model_name
        # allow forcing device ('cpu' or 'cuda') to the sentence-transformers model
        self.device = device
        self.model = SentenceTransformer(embedding_model_name, device=device)
        self.normalize = normalize

        # in-memory storage
        self._embeddings: Optional[np.ndarray] = None
        self._ids: Optional[List[Any]] = None
        self._texts: Optional[List[str]] = None

        # index/container
        self._index = None
        self._use_faiss = _HAS_FAISS

    def _maybe_normalize(self, vectors: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def build_from_texts(self, texts: List[str], ids: Optional[List[Any]] = None, batch_size: int = 64):
        """Compute embeddings and build index.

        Args:
            texts: list of document texts
            ids: optional list of identifiers (same length as texts)
            batch_size: embedding batch size
        """
        if ids is not None and len(ids) != len(texts):
            raise ValueError("ids must be same length as texts")

        # compute embeddings in batches
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.normalize:
            embeddings = self._maybe_normalize(embeddings)

        # persist in-memory
        self._embeddings = embeddings
        self._texts = list(texts)
        self._ids = list(ids) if ids is not None else list(range(len(texts)))

        # build index
        self._build_index()

    def _build_index(self):
        if self._embeddings is None:
            raise RuntimeError("No embeddings available. Call build_from_texts first.")

        dim = self._embeddings.shape[1]

        if self._use_faiss:
            # use inner product on normalized vectors (cosine similarity)
            index = faiss.IndexFlatIP(dim)
            if not index.is_trained:
                # IndexFlat doesn't need training, but keep pattern
                pass
            # FAISS expects contiguous float32 array
            index.add(np.ascontiguousarray(self._embeddings))
            self._index = index
        else:
            # sklearn fallback
            n_neighbors = min(10, max(1, self._embeddings.shape[0]))
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
            nn.fit(self._embeddings)
            self._index = nn

    def add_texts(self, texts: List[str], ids: Optional[List[Any]] = None, batch_size: int = 64):
        """Add new documents to the KB (append). Rebuilds or updates index in-place.

        For FAISS this will add vectors to the existing Index. For sklearn fallback
        we will re-fit the NearestNeighbors object with the expanded embeddings.
        """
        if ids is not None and len(ids) != len(texts):
            raise ValueError("ids must be same length as texts")

        new_emb = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        new_emb = np.asarray(new_emb, dtype=np.float32)
        if self.normalize:
            new_emb = self._maybe_normalize(new_emb)

        # initialize if empty
        if self._embeddings is None:
            self._embeddings = new_emb
            self._texts = list(texts)
            self._ids = list(ids) if ids is not None else list(range(len(new_emb)))
            self._build_index()
            return

        # append embeddings & metadata
        self._embeddings = np.vstack([self._embeddings, new_emb])
        base = len(self._texts)
        self._texts.extend(list(texts))
        if ids is not None:
            self._ids.extend(list(ids))
        else:
            self._ids.extend(list(range(base, base + len(new_emb))))

        # update index
        if self._use_faiss:
            # ensure faiss index exists
            if self._index is None:
                self._build_index()
            else:
                self._index.add(np.ascontiguousarray(new_emb))
        else:
            # sklearn: refit
            self._build_index()

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Any, float, str]]:
        """Retrieve top_k documents for a query.

        Returns list of tuples: (id, score, text)
        Score is similarity (higher = more similar). When using sklearn fallback score is 1 - cosine_distance.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_from_texts or load first.")

        q_emb = self.model.encode([query])
        q_emb = np.asarray(q_emb, dtype=np.float32)
        if self.normalize:
            q_emb = self._maybe_normalize(q_emb)

        if self._use_faiss:
            # faiss returns scores (inner product) and indices
            k = min(top_k, self._embeddings.shape[0])
            D, I = self._index.search(q_emb, k)
            # D shape (1, k)
            scores = D[0].tolist()
            idxs = I[0].tolist()
            results = []
            for idx, score in zip(idxs, scores):
                if idx < 0:
                    continue
                results.append((self._ids[idx], float(score), self._texts[idx]))
            return results
        else:
            # sklearn NearestNeighbors: returns distances (cosine)
            k = min(top_k, self._embeddings.shape[0])
            dist, idxs = self._index.kneighbors(q_emb, n_neighbors=k)
            results = []
            for d, idx in zip(dist[0].tolist(), idxs[0].tolist()):
                score = 1.0 - d  # convert cosine distance to similarity
                results.append((self._ids[idx], float(score), self._texts[idx]))
            return results

    def save(self, path: str):
        """Save index and metadata to directory `path`.

        Files created:
            index.faiss (if faiss) or embeddings.npy + sklearn_nn.pkl
            metadata.pkl (ids, texts, embedding_model_name, normalize)
        """
        os.makedirs(path, exist_ok=True)

        meta = {
            "ids": self._ids,
            "texts": self._texts,
            "embedding_model_name": self.embedding_model_name,
            "normalize": self.normalize,
            "use_faiss": self._use_faiss,
        }

        meta_path = os.path.join(path, "metadata.pkl")
        with open(meta_path, "wb") as fh:
            pickle.dump(meta, fh)

        if self._use_faiss:
            if not _HAS_FAISS:
                raise RuntimeError("FAISS not available at save time")
            # save faiss index
            faiss.write_index(self._index, os.path.join(path, "index.faiss"))
            # also save embeddings (optional)
            if self._embeddings is not None:
                np.save(os.path.join(path, "embeddings.npy"), self._embeddings)
        else:
            # save embeddings and sklearn NN
            if self._embeddings is not None:
                np.save(os.path.join(path, "embeddings.npy"), self._embeddings)
            # sklearn object
            with open(os.path.join(path, "sklearn_index.pkl"), "wb") as fh:
                pickle.dump(self._index, fh)

    def load(self, path: str):
        """Load index and metadata from directory `path`."""
        meta_path = os.path.join(path, "metadata.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"metadata.pkl not found in {path}")

        with open(meta_path, "rb") as fh:
            meta = pickle.load(fh)

        self._ids = meta.get("ids")
        self._texts = meta.get("texts")
        self.embedding_model_name = meta.get("embedding_model_name", self.embedding_model_name)
        self.normalize = meta.get("normalize", self.normalize)
        self._use_faiss = meta.get("use_faiss", self._use_faiss)

        emb_path = os.path.join(path, "embeddings.npy")
        if os.path.exists(emb_path):
            self._embeddings = np.load(emb_path)

        if self._use_faiss and _HAS_FAISS:
            idx_file = os.path.join(path, "index.faiss")
            if not os.path.exists(idx_file):
                raise FileNotFoundError("FAISS index file not found but metadata expects FAISS")
            self._index = faiss.read_index(idx_file)
        else:
            sklearn_idx = os.path.join(path, "sklearn_index.pkl")
            if not os.path.exists(sklearn_idx):
                # if sklearn index missing but embeddings present, we can rebuild
                if self._embeddings is not None:
                    self._build_index()
                    return
                raise FileNotFoundError("sklearn_index.pkl not found and cannot rebuild index")
            with open(sklearn_idx, "rb") as fh:
                self._index = pickle.load(fh)
