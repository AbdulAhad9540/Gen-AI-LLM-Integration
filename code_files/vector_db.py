import os
import pickle
import threading
from typing import List, Dict, Any, Tuple

import numpy as np


class SimpleVectorStore:
    """A tiny vector store using numpy arrays and pickle persistence.

    Not designed for production, but simple and dependency-free for demos.
    """

    def __init__(self, path: str = "vector_store.pkl"):
        self.path = path
        self.lock = threading.Lock()
        # lists for metadata and vectors
        self.ids: List[str] = []
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.metadatas: List[Dict[str, Any]] = []
        # load if exists
        if os.path.exists(self.path):
            try:
                self._load()
            except Exception:
                # ignore load errors for simplicity
                pass

    def _save(self):
        with self.lock:
            with open(self.path, "wb") as f:
                pickle.dump({
                    "ids": self.ids,
                    "vectors": self.vectors,
                    "metadatas": self.metadatas,
                }, f)

    def _load(self):
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        self.ids = data.get("ids", [])
        self.vectors = data.get("vectors", np.zeros((0, 0), dtype=np.float32))
        self.metadatas = data.get("metadatas", [])

    def upsert(self, ids: List[str], vectors: List[np.ndarray], metadatas: List[Dict[str, Any]]):
        with self.lock:
            vectors = np.vstack([np.array(v, dtype=np.float32) for v in vectors])
            if self.vectors.size == 0:
                self.vectors = vectors
            else:
                # check dims
                if vectors.shape[1] != self.vectors.shape[1]:
                    raise ValueError("Vector dimensionality mismatch")
                self.vectors = np.vstack([self.vectors, vectors])
            self.ids.extend(ids)
            self.metadatas.extend(metadatas)
            print(f"[SimpleVectorStore] Upserted {len(ids)} vectors. Total now: {len(self.ids)}. Vector shape: {self.vectors.shape}")
        self._save()

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        # assumes 1D vectors
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return list of (id, score, metadata) sorted by score desc."""
        with self.lock:
            if self.vectors.size == 0:
                print("[SimpleVectorStore] No vectors in store.")
                return []
            q = np.array(query_vector, dtype=np.float32)
            # normalize for faster computation
            norms = np.linalg.norm(self.vectors, axis=1)
            qnorm = np.linalg.norm(q)
            if qnorm == 0:
                print("[SimpleVectorStore] Query vector norm is zero.")
                return []
            dots = np.dot(self.vectors, q)
            sims = dots / (norms * qnorm + 1e-12)
            top_idx = np.argsort(-sims)[:top_k]
            results = []
            print(f"[SimpleVectorStore] Search scores: {[float(sims[i]) for i in top_idx]}")
            for i in top_idx:
                print(f"[SimpleVectorStore] Top result: id={self.ids[i]}, score={float(sims[i])}, text={self.metadatas[i].get('text','')[:80]}")
                results.append((self.ids[i], float(sims[i]), self.metadatas[i]))
            return results
