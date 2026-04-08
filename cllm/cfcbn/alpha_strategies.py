"""
alpha_strategies.py

Strategy A   rag        RAGAlphaStrategy
    Local FAISS + BoW encoder. No API call.
    alpha = 1 - cosine_sim(new_case_evidence, best_historical_match)

Strategy A2  rag_api    RAGApiAlphaStrategy
    Same RAG mechanism, but vectors produced by the APIYI embedding API
    (text-embedding-3-small, 1536-dim). Semantically richer; ~0.1-0.3s/case.
    Comparison against rag shows how much embedding quality matters.

Strategy B   jaccard    JaccardAlphaStrategy
    Weighted-Jaccard over CBNAccumulator history features.
    alpha = 1 - max_jaccard_similarity  (no separate store)

Strategy C   entropy    EntropyAlphaStrategy
    Shannon entropy of CBN posterior.
    High entropy -> CBN uncertain -> alpha up.
    alpha = alpha_min + (alpha_max - alpha_min) * H_norm
"""

from __future__ import annotations

import math
import logging
import os
import pickle
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ALPHA_MIN = 0.20
ALPHA_MAX = 1.00


class BaseAlphaStrategy:
    name: str = "base"

    def suggest(
        self,
        anomaly_services: Dict[str, List[str]],
        anomaly_pods:     Dict[str, List[str]],
        fault_type:       str,
        accumulator,
        rag_store=None,
    ) -> Tuple[float, str]:
        raise NotImplementedError


# ── Strategy A: RAG local BoW ────────────────────────────────────────────────

class RAGAlphaStrategy(BaseAlphaStrategy):
    """
    alpha = clip(1 - K, ALPHA_MIN, ALPHA_MAX)
    K = cosine similarity via local BoW FAISS store (no API).
    K high -> history reliable -> low alpha (more CBN).
    K low  -> novel fault      -> alpha=1.0 (pure CF).
    """
    name = "rag"

    def suggest(self, anomaly_services, anomaly_pods, fault_type,
                accumulator, rag_store=None) -> Tuple[float, str]:
        if rag_store is None or len(rag_store) == 0:
            return ALPHA_MAX, "RAG(bow) store empty — alpha=1.0 (pure CF)"
        alpha, best_k, best_meta = rag_store.suggest_alpha(
            anomaly_services, anomaly_pods, fault_type)
        if best_meta:
            reason = (f"RAG(bow): sim={best_k:.3f} "
                      f"match={best_meta.get('root_cause','?')} "
                      f"-> alpha={alpha:.3f}")
        else:
            reason = f"RAG(bow): no match -> alpha={alpha:.3f}"
        return alpha, reason


# ── Strategy A2: RAG with APIYI embedding API ────────────────────────────────

class RAGApiAlphaStrategy(BaseAlphaStrategy):
    """
    Identical mechanism to RAGAlphaStrategy but vectors come from the
    APIYI embedding API (text-embedding-3-small, 1536-dim) instead of BoW.

    This strategy maintains its own dedicated RAGCaseStore instance that
    always uses the _APIVectorizer, independent of the global USE_API_EMBED
    flag in rag_case_store.py.  The store is saved in a sibling directory
    named <original_store_dir>_api/.

    Ablation value:
      rag vs rag_api shows the impact of embedding quality on alpha estimation.
    """
    name = "rag_api"

    def __init__(self):
        self._store = None
        self._store_dir = None

    def _get_store(self, base_store_dir: str):
        """Return (creating if needed) an API-backed RAGCaseStore."""
        api_dir = base_store_dir.rstrip("/\\") + "_api"
        if self._store is not None and self._store_dir == api_dir:
            return self._store

        from cfcbn.rag_case_store import (
            RAGCaseStore, _APIVectorizer, _BagOfWordsVectorizer,
            EMBED_API_KEY, EMBED_BASE_URL, EMBED_MODEL, EMBED_DIM,
        )
        import faiss

        store = RAGCaseStore.__new__(RAGCaseStore)
        store.store_dir  = api_dir
        store._idx_path  = os.path.join(api_dir, "rag_api_index.faiss")
        store._voc_path  = os.path.join(api_dir, "rag_api_vocab.pkl")
        store._meta_path = os.path.join(api_dir, "rag_api_meta.pkl")
        store._backend   = "api"
        store._meta      = []
        store._index     = None
        os.makedirs(api_dir, exist_ok=True)

        try:
            store._vectorizer = _APIVectorizer(
                api_key=EMBED_API_KEY,
                base_url=EMBED_BASE_URL,
                model=EMBED_MODEL,
                dim=EMBED_DIM,
            )
        except Exception as e:
            logger.warning(f"[RAGApi] API init failed ({e}); falling back to BoW")
            store._vectorizer = _BagOfWordsVectorizer()
            store._backend    = "bow_fallback"

        # Load persisted index if present
        try:
            if (os.path.exists(store._idx_path) and
                    os.path.exists(store._meta_path)):
                store._index = faiss.read_index(store._idx_path)
                with open(store._meta_path, "rb") as fh:
                    store._meta = pickle.load(fh)
                logger.info(f"[RAGApi] Loaded {store._index.ntotal} vectors "
                            f"from {api_dir}")
        except Exception as e:
            logger.warning(f"[RAGApi] Load failed ({e}); starting fresh")

        self._store     = store
        self._store_dir = api_dir
        print(f"[RAGApi] store backend={store._backend}  dir={api_dir}  "
              f"vectors={len(store._meta)}")
        return self._store

    def suggest(self, anomaly_services, anomaly_pods, fault_type,
                accumulator, rag_store=None) -> Tuple[float, str]:
        # Resolve the base directory from the BoW rag_store (or use a tmpdir)
        if rag_store is not None:
            base_dir = rag_store.store_dir
        else:
            base_dir = tempfile.mkdtemp(prefix="rag_api_base_")

        store = self._get_store(base_dir)

        if len(store) == 0:
            return ALPHA_MAX, "RAG(api) store empty — alpha=1.0 (pure CF)"

        alpha, best_k, best_meta = store.suggest_alpha(
            anomaly_services, anomaly_pods, fault_type)
        if best_meta:
            reason = (f"RAG(api): sim={best_k:.3f} "
                      f"match={best_meta.get('root_cause','?')} "
                      f"-> alpha={alpha:.3f}")
        else:
            reason = f"RAG(api): no match -> alpha={alpha:.3f}"
        return alpha, reason

    def add_confirmed(self, anomaly_services, anomaly_pods, fault_type,
                      root_cause, base_store_dir: str):
        """Add a confirmed case to the API store. Called after accumulation."""
        store = self._get_store(base_store_dir)
        store.add(anomaly_services, anomaly_pods, fault_type, root_cause)

    def reset(self):
        """Clear in-memory state between experiment runs."""
        self._store     = None
        self._store_dir = None


# ── Strategy B: Jaccard ──────────────────────────────────────────────────────

class JaccardAlphaStrategy(BaseAlphaStrategy):
    """
    alpha = clip(1 - max_weighted_jaccard(new, history), ALPHA_MIN, ALPHA_MAX)
    Reuses the same feature space as the CBN; no separate store needed.
    """
    name = "jaccard"

    def suggest(self, anomaly_services, anomaly_pods, fault_type,
                accumulator, rag_store=None) -> Tuple[float, str]:
        if accumulator.total == 0:
            return ALPHA_MAX, "No history — alpha=1.0 (pure CF)"
        from cfcbn.cbn_accumulator import case_to_feature_vector, weighted_jaccard
        fake = {"anomaly_services": anomaly_services, "anomaly_pods": anomaly_pods}
        q    = case_to_feature_vector(fake, accumulator.services, accumulator.metrics)
        sims = [weighted_jaccard(q, h["feat"]) for h in accumulator.history]
        best_k = float(max(sims)) if sims else 0.0
        alpha  = float(np.clip(1.0 - best_k, ALPHA_MIN, ALPHA_MAX))
        return alpha, (f"Jaccard: max_sim={best_k:.3f} over "
                       f"{len(sims)} entries -> alpha={alpha:.3f}")


# ── Strategy C: Entropy (creative) ───────────────────────────────────────────

class EntropyAlphaStrategy(BaseAlphaStrategy):
    """
    alpha = alpha_min + (alpha_max - alpha_min) * H_norm
    H_norm = H(CBN_posterior) / log(n_services)
    Measures CBN posterior confidence directly, not historical similarity.
    """
    name = "entropy"

    def suggest(self, anomaly_services, anomaly_pods, fault_type,
                accumulator, rag_store=None) -> Tuple[float, str]:
        if accumulator.total == 0:
            return ALPHA_MAX, "No history — alpha=1.0 (pure CF)"
        fake     = {"anomaly_services": anomaly_services, "anomaly_pods": anomaly_pods}
        posterior = accumulator.compute_posterior(fake)
        probs    = np.array(list(posterior.values()), dtype=np.float64)
        probs    = probs[probs > 1e-12]
        H        = -float(np.sum(probs * np.log(probs)))
        n        = len(accumulator.services)
        H_max    = math.log(n) if n > 1 else 1.0
        H_norm   = float(np.clip(H / H_max, 0.0, 1.0))
        alpha    = float(np.clip(
            ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * H_norm, ALPHA_MIN, ALPHA_MAX))
        return alpha, (f"Entropy: H_norm={H_norm:.3f} "
                       f"(H={H:.3f}) -> alpha={alpha:.3f}")


# ── Registry ─────────────────────────────────────────────────────────────────

STRATEGIES: Dict[str, BaseAlphaStrategy] = {
    "rag":     RAGAlphaStrategy(),
    "rag_api": RAGApiAlphaStrategy(),
    "jaccard": JaccardAlphaStrategy(),
    "entropy": EntropyAlphaStrategy(),
}


def get_strategy(name: str) -> BaseAlphaStrategy:
    if name not in STRATEGIES:
        raise ValueError(
            f"Unknown alpha strategy '{name}'. "
            f"Available: {list(STRATEGIES.keys())}"
        )
    return STRATEGIES[name]
