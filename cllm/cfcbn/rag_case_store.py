"""
rag_case_store.py — RAG-based fault case vector store for adaptive alpha tuning.

Two vectorisation backends are available.  Switch by changing USE_API_EMBED
at the top of this file:

    USE_API_EMBED = False   # default — local BoW encoder, no API call needed
    USE_API_EMBED = True    # API-based embedding (APIYI proxy for OpenAI models)

When USE_API_EMBED = True the embedding request is forwarded to the APIYI
endpoint (OpenAI-compatible, no direct access to openai.com required).
Set EMBED_API_KEY to your APIYI key below.

Hard-coded intentionally: the key is NOT read from environment variables to
avoid accidental leakage through shared deployment environments.

Design
------
Each confirmed fault case is encoded as an evidence text chunk:
    "fault_type=<type> svc_A:m1,m2 svc_B:m3 pod_C:m4"
Only observable evidence (no root-cause label) is included, so a NEW case
at inference time produces the same format as stored historical cases.

Alpha is set as:
    alpha = clip(1 - K, MIN_ALPHA, MAX_ALPHA)
where K = cosine similarity of the new case to the best historical match.

K ~ 1  =>  similar history exists  =>  CBN prior reliable  =>  alpha low
K ~ 0  =>  novel fault             =>  cold prior dangerous =>  alpha = 1.0
"""

from __future__ import annotations

import os
import re
import json
import pickle
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Version switch — change this and delete the index files to rebuild
# ─────────────────────────────────────────────────────────────────────────────

# Set to True to use API-based embeddings (APIYI / OpenAI-compatible).
# Set to False (default) to use the local BoW encoder (no API needed).
USE_API_EMBED = False

# APIYI API key.  Used only when USE_API_EMBED = True.
# Replace with your actual key; do NOT put it in environment variables.
EMBED_API_KEY   = "sk-tgQpuuAdudLNhPMl2c374f28781740Bd930d5eD1Cb626461"
EMBED_BASE_URL  = "https://api.apiyi.com/v1"          # APIYI endpoint
EMBED_MODEL     = "text-embedding-3-small"             # 1536-dim
EMBED_DIM       = 1536

# ─────────────────────────────────────────────────────────────────────────────
# File names inside store_dir
# ─────────────────────────────────────────────────────────────────────────────
FAISS_INDEX_FILE = "rag_index.faiss"
VOCAB_FILE       = "rag_vocab.pkl"   # local BoW only
META_FILE        = "rag_meta.pkl"

MIN_ALPHA = 0.20
MAX_ALPHA = 1.00


# ─────────────────────────────────────────────────────────────────────────────
# Evidence text builder  (shared by both backends)
# ─────────────────────────────────────────────────────────────────────────────

def _case_to_evidence_text(
    anomaly_services: Dict[str, List[str]],
    anomaly_pods:     Dict[str, List[str]],
    fault_type:       str = "",
) -> str:
    """
    Convert a fault case's observable evidence to a text chunk.
    Root-cause label is intentionally excluded so the same format is used
    for both indexing (confirmed cases) and querying (new cases at inference).
    """
    parts = []
    if fault_type:
        parts.append(f"fault_type={fault_type.replace(' ', '_')}")
    for svc, metrics in sorted(anomaly_services.items()):
        if metrics:
            parts.append(f"svc_{svc}:{','.join(sorted(set(str(m) for m in metrics)))}")
    for pod, metrics in sorted(anomaly_pods.items()):
        if metrics:
            parts.append(f"pod_{pod}:{','.join(sorted(set(str(m) for m in metrics)))}")
    return " ".join(parts) if parts else "empty_case"


# ─────────────────────────────────────────────────────────────────────────────
# Backend A: Local BoW vectoriser  (default, no API)
# ─────────────────────────────────────────────────────────────────────────────

class _BagOfWordsVectorizer:
    """
    Minimal incremental BoW vectoriser.
    Vocabulary grows as new tokens are encountered.
    Vectors are L2-normalised (inner product == cosine similarity).
    """

    def __init__(self):
        self.vocab: Dict[str, int] = {}

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in re.split(r'[ :,=_]+', text.lower()) if t]

    def fit_transform(self, text: str) -> np.ndarray:
        """Add new tokens to vocab and return L2-normalised vector."""
        tokens = self._tokenize(text)
        for tok in tokens:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        return self._transform(tokens)

    def transform(self, text: str) -> np.ndarray:
        """Transform with existing vocab; unknown tokens are silently ignored."""
        return self._transform(self._tokenize(text))

    def _transform(self, tokens: List[str]) -> np.ndarray:
        dim = max(len(self.vocab), 1)
        vec = np.zeros(dim, dtype=np.float32)
        for tok in tokens:
            idx = self.vocab.get(tok)
            if idx is not None and idx < dim:
                vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec


# ─────────────────────────────────────────────────────────────────────────────
# Backend B: API-based vectoriser  (APIYI / OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

class _APIVectorizer:
    """
    Calls the APIYI embedding endpoint (OpenAI-compatible).
    Uses the standard requests library — no openai SDK required.
    Retries up to MAX_RETRIES times on transient errors.
    """

    MAX_RETRIES = 3
    RETRY_SLEEP = 5

    def __init__(
        self,
        api_key:  str = EMBED_API_KEY,
        base_url: str = EMBED_BASE_URL,
        model:    str = EMBED_MODEL,
        dim:      int = EMBED_DIM,
    ):
        import requests as _req
        self._req      = _req
        self._api_key  = api_key
        self._base_url = base_url.rstrip('/')
        self._model    = model
        self._dim      = dim
        self.vocab     = {}   # unused; kept for interface compatibility
        logger.info(
            f"[RAGCaseStore] API embedding backend  "
            f"url={self._base_url}  model={self._model}  dim={self._dim}"
        )

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Call /v1/embeddings and return L2-normalised float32 matrix."""
        url     = f"{self._base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }
        payload = {"model": self._model, "input": texts}

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self._req.post(url, headers=headers,
                                      data=json.dumps(payload), timeout=30)
                resp.raise_for_status()
                data = resp.json()
                vecs = np.array(
                    [item["embedding"] for item in data["data"]],
                    dtype=np.float32,
                )
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms < 1e-9, 1.0, norms)
                return vecs / norms
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(
                        f"[RAGCaseStore] Embedding attempt {attempt+1} failed: {e}; "
                        f"retrying in {self.RETRY_SLEEP}s"
                    )
                    time.sleep(self.RETRY_SLEEP)
                else:
                    logger.error(
                        f"[RAGCaseStore] Embedding failed after "
                        f"{self.MAX_RETRIES} attempts: {e}"
                    )
                    return None   # signal failure to caller

    def fit_transform(self, text: str) -> np.ndarray:
        """Embed a single text; falls back to zero-vector on API failure."""
        result = self._embed_batch([text])
        if result is None:
            # API unavailable — return zero vector of expected dim
            return np.zeros(self._dim, dtype=np.float32)
        return result[0]

    def transform(self, text: str) -> np.ndarray:
        result = self._embed_batch([text])
        if result is None:
            return np.zeros(self._dim, dtype=np.float32)
        return result[0]

    @property
    def dim(self) -> int:
        return self._dim


# ─────────────────────────────────────────────────────────────────────────────
# RAGCaseStore — unified FAISS-backed vector store
# ─────────────────────────────────────────────────────────────────────────────

class RAGCaseStore:
    """
    Vector store for fault case evidence chunks.

    Backend is selected by USE_API_EMBED at module load time.

    Switching backends
    ------------------
    1. Edit USE_API_EMBED and (if True) EMBED_API_KEY at the top of this file.
    2. Delete case_store/rag_index.faiss, rag_vocab.pkl, rag_meta.pkl.
    3. Re-run — the index is rebuilt automatically with the new backend.
    """

    def __init__(self, store_dir: str = "case_store"):
        self.store_dir  = store_dir
        self._idx_path  = os.path.join(store_dir, FAISS_INDEX_FILE)
        self._voc_path  = os.path.join(store_dir, VOCAB_FILE)
        self._meta_path = os.path.join(store_dir, META_FILE)

        os.makedirs(store_dir, exist_ok=True)

        if USE_API_EMBED:
            self._vectorizer = _APIVectorizer(
                api_key=EMBED_API_KEY,
                base_url=EMBED_BASE_URL,
                model=EMBED_MODEL,
                dim=EMBED_DIM,
            )
            self._backend = "api"
        else:
            self._vectorizer = _BagOfWordsVectorizer()
            self._backend = "bow"

        self._meta: List[dict] = []
        self._index = None

        self._load()
        print(
            f"[RAGCaseStore] backend={self._backend}  "
            f"store={store_dir}  vectors={len(self._meta)}"
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        try:
            import faiss
            if (os.path.exists(self._idx_path)
                    and os.path.exists(self._meta_path)):
                self._index = faiss.read_index(self._idx_path)
                with open(self._meta_path, 'rb') as f:
                    self._meta = pickle.load(f)
                if self._backend == "bow" and os.path.exists(self._voc_path):
                    with open(self._voc_path, 'rb') as f:
                        self._vectorizer = pickle.load(f)
                logger.info(
                    f"[RAGCaseStore] Loaded {self._index.ntotal} vectors"
                )
        except Exception as e:
            logger.warning(f"[RAGCaseStore] Load failed ({e}); starting fresh")
            self._index = None
            self._meta  = []

    def _save(self):
        try:
            import faiss
            if self._index is not None:
                faiss.write_index(self._index, self._idx_path)
            with open(self._meta_path, 'wb') as f:
                pickle.dump(self._meta, f)
            if self._backend == "bow":
                with open(self._voc_path, 'wb') as f:
                    pickle.dump(self._vectorizer, f)
        except Exception as e:
            logger.warning(f"[RAGCaseStore] Save failed: {e}")

    def _rebuild_bow_index(self):
        """Re-vectorise all stored texts after BoW vocabulary growth."""
        try:
            import faiss
            if not self._meta:
                return
            dim = len(self._vectorizer.vocab)
            if dim == 0:
                return
            new_vecs = []
            for m in self._meta:
                v = self._vectorizer.transform(m.get("evidence_text", ""))
                if len(v) < dim:
                    v = np.pad(v, (0, dim - len(v)))
                elif len(v) > dim:
                    v = v[:dim]
                norm = np.linalg.norm(v)
                new_vecs.append(
                    (v / norm if norm > 1e-9 else v).astype(np.float32)
                )
            mat = np.vstack(new_vecs)
            idx = faiss.IndexFlatIP(dim)
            idx.add(mat)
            self._index = idx
        except Exception as e:
            logger.warning(f"[RAGCaseStore] BoW index rebuild failed: {e}")
            self._index = None

    # ── Public API ───────────────────────────────────────────────────────────

    def add(
        self,
        anomaly_services: Dict[str, List[str]],
        anomaly_pods:     Dict[str, List[str]],
        fault_type:       str,
        root_cause:       str,
    ) -> None:
        """Add a confirmed fault case to the vector store."""
        import faiss
        text = _case_to_evidence_text(anomaly_services, anomaly_pods, fault_type)
        vec  = self._vectorizer.fit_transform(text).astype(np.float32)

        self._meta.append({
            "evidence_text": text,
            "fault_type":    fault_type,
            "root_cause":    root_cause,
        })

        if self._backend == "api":
            # API backend: fixed 1536-dim, append directly
            dim = EMBED_DIM
            if self._index is None:
                self._index = faiss.IndexFlatIP(dim)
            v = vec.reshape(1, -1)
            if v.shape[1] != dim:
                pad = np.zeros((1, dim), dtype=np.float32)
                l = min(v.shape[1], dim)
                pad[0, :l] = v[0, :l]
                v = pad
            self._index.add(v)
        else:
            # BoW backend: vocab may have grown, rebuild whole index
            self._rebuild_bow_index()

        self._save()

    def query_best(
        self,
        anomaly_services: Dict[str, List[str]],
        anomaly_pods:     Dict[str, List[str]],
        fault_type:       str = "",
        top_k:            int = 1,
    ) -> Tuple[float, Optional[dict]]:
        """
        Return (best_score, best_meta) for the closest historical case.
        best_score in [0, 1] (cosine similarity via L2-normalised inner product).
        Returns (0.0, None) if store is empty or index unavailable.
        """
        if self._index is None or self._index.ntotal == 0:
            return 0.0, None

        text = _case_to_evidence_text(anomaly_services, anomaly_pods, fault_type)
        try:
            q = self._vectorizer.transform(text).astype(np.float32)
        except Exception as e:
            logger.warning(f"[RAGCaseStore] Query embed failed: {e}")
            return 0.0, None

        dim = self._index.d
        if len(q) < dim:
            q = np.pad(q, (0, dim - len(q)))
        elif len(q) > dim:
            q = q[:dim]

        norm = np.linalg.norm(q)
        if norm < 1e-9:
            return 0.0, None
        q = (q / norm).reshape(1, -1)

        try:
            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(q, k)
            best_score = float(np.clip(scores[0][0], 0.0, 1.0))
            best_idx   = int(indices[0][0])
            if best_idx < 0 or best_idx >= len(self._meta):
                return 0.0, None
            return best_score, self._meta[best_idx]
        except Exception as e:
            logger.warning(f"[RAGCaseStore] FAISS search failed: {e}")
            return 0.0, None

    def suggest_alpha(
        self,
        anomaly_services: Dict[str, List[str]],
        anomaly_pods:     Dict[str, List[str]],
        fault_type:       str = "",
    ) -> Tuple[float, float, Optional[dict]]:
        """
        Return (alpha, best_score, best_meta).
        alpha = clip(1 - best_score, MIN_ALPHA, MAX_ALPHA)
        """
        best_score, best_meta = self.query_best(
            anomaly_services, anomaly_pods, fault_type
        )
        alpha = float(np.clip(1.0 - best_score, MIN_ALPHA, MAX_ALPHA))
        return alpha, best_score, best_meta

    @property
    def backend(self) -> str:
        return self._backend

    def __len__(self) -> int:
        return len(self._meta)
