"""BGE-M3 Experimental Retrieval Backend.

Provides dense + sparse retrieval using BAAI/bge-m3 as an A/B-testable
alternative to the baseline BM25 + FAISS (nomic-embed) pipeline.

Architecture:
- BGE-M3 encodes queries and documents into dense (1024-d) and sparse
  (vocab-dim) representations in a single forward pass.
- Dense vectors are indexed via FAISS (cosine similarity via inner product).
- Sparse vectors are stored in a scipy CSR matrix; search is a dot product.
- Results are fused with configurable dense/sparse weights.

This module does NOT modify the existing BM25, FAISS, or EmbeddingGenerator.
It is a parallel path selectable via RetrievalMethod.BGE_M3_HYBRID.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import faiss
import numpy as np
from app.config import get_settings
from app.evidence.models import Chunk, EvidenceRef
from app.evidence.store import EvidenceStore, get_evidence_store
from app.logging_config import get_logger

logger = get_logger("argus.retrieval.bge_m3")

# Lazy-loaded model singleton (thread-safe)
_model_lock = threading.Lock()
_bge_m3_model: Any = None


class _TransformersBGE3:
    """Lightweight wrapper around raw transformers for BGE-M3 dense+sparse.

    Provides the same encode() interface as BGEM3FlagModel but uses
    AutoModel + ColBERT linear layers directly. Used when FlagEmbedding
    is unavailable or too heavy.
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        import torch
        self._torch = torch

    def encode(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        import torch
        with torch.no_grad():
            encoded = self.tokenizer(
                texts, padding=True, truncation=True,
                max_length=kwargs.get("max_length", 512),
                return_tensors="pt",
            )
            output = self.model(**encoded, output_hidden_states=True)
            # Dense: CLS pooling
            dense = output.last_hidden_state[:, 0, :].numpy()
            # Sparse: ColBERT-style max-pooling over linear
            hidden = output.last_hidden_state  # (batch, seq, hidden)
            sparse_linear = None
            for name, param in self.model.named_parameters():
                if "sparse" in name and "linear" in name:
                    sparse_linear = param
                    break
            if sparse_linear is not None:
                sparse = self._torch.einsum("bsd,dv->bsv", hidden, sparse_linear)
                sparse = sparse.max(dim=1).values.clamp(min=0).numpy()
            else:
                # Fallback: use token embeddings as sparse signal
                sparse = hidden.max(dim=1).values.clamp(min=0).numpy()
            return {"dense_vecs": dense, "lexical_weights": [self._row_to_dict(s) for s in sparse]}

    @staticmethod
    def _row_to_dict(row: Any) -> dict[int, float]:
        """Convert a sparse row to {token_id: weight} dict, keeping nonzero."""
        d: dict[int, float] = {}
        for i, v in enumerate(row):
            if v > 0:
                d[i] = float(v)
        return d


@dataclass
class BGEIndexStats:
    """Diagnostic stats for a BGE-M3 index build/search cycle."""

    chunk_count: int = 0
    dense_dim: int = 0
    sparse_nnz: int = 0
    build_time_s: float = 0.0
    search_time_s: float = 0.0
    dense_index_size_bytes: int = 0
    sparse_index_size_bytes: int = 0
    ram_usage_bytes: int = 0
    vram_usage_bytes: int = 0


class BGEM3Retriever:
    """BGE-M3 dense + sparse retriever.

    Maintains its own FAISS index (dense) and scipy CSR matrix (sparse),
    completely independent of the baseline HybridRetriever's indexes.
    """

    def __init__(
        self,
        store: EvidenceStore | None = None,
        model_name: str | None = None,
        dense_index_path: Path | None = None,
        sparse_index_path: Path | None = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ):
        self.store = store or get_evidence_store()
        self.settings = get_settings()
        self.model_name = model_name or getattr(
            self.settings, "bge_m3_model", "BAAI/bge-m3"
        )
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        default_dense = self.settings.data_dir / "indexes" / "bge_m3_dense.index"
        default_sparse = self.settings.data_dir / "indexes" / "bge_m3_sparse.pkl"
        self.dense_index_path = dense_index_path or getattr(
            self.settings, "bge_m3_dense_index_path", default_dense
        )
        self.sparse_index_path = sparse_index_path or getattr(
            self.settings, "bge_m3_sparse_index_path", default_sparse
        )
        self.dense_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.sparse_index_path.parent.mkdir(parents=True, exist_ok=True)

        # Index state
        self._dense_index: faiss.Index | None = None
        self._sparse_matrix: Any = None  # scipy.sparse.csr_matrix
        self._chunk_ids: list[str] = []  # aligned with index rows
        self._dense_dim: int = 1024  # BGE-M3 default
        self._stats = BGEIndexStats()

    # -- Model access --------------------------------------------------------

    @classmethod
    def _get_model(cls) -> Any:
        """Lazy-load the BGE-M3 model (singleton, thread-safe).

        Tries FlagEmbedding first (full dense+sparse+colbert support),
        falls back to raw transformers for lighter CPU-only usage.
        """
        global _bge_m3_model
        with _model_lock:
            if _bge_m3_model is None:
                logger.info("loading_bge_m3_model", model="BAAI/bge-m3")
                try:
                    from FlagEmbedding import BGEM3FlagModel

                    _bge_m3_model = BGEM3FlagModel(
                        "BAAI/bge-m3",
                        use_fp16=False,  # CPU-safe default; set True for GPU
                    )
                    _bge_m3_model._backend = "flag"
                    logger.info("bge_m3_model_loaded")
                except Exception as exc_flag:  # noqa: BLE001
                    logger.warning("flag_embedding_unavailable", error=str(exc_flag))
                    try:
                        import torch
                        from transformers import AutoModel, AutoTokenizer

                        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
                        model = AutoModel.from_pretrained("BAAI/bge-m3")
                        model.eval()
                        _bge_m3_model = _TransformersBGE3(model, tokenizer)
                        _bge_m3_model._backend = "transformers"
                        logger.info("bge_m3_model_loaded_via_transformers")
                    except Exception as exc_tf:  # noqa: BLE001
                        logger.error("bge_m3_model_load_failed", error=str(exc_tf))
                        raise
            return _bge_m3_model

    # -- Encode --------------------------------------------------------------

    def _encode(self, texts: list[str]) -> dict[str, Any]:
        """Encode texts into dense + sparse representations.

        Returns dict with:
          - 'dense': np.ndarray shape (n, dense_dim)
          - 'sparse': list of dicts {token_id: weight} (ColBERT-style)
        """
        model = self._get_model()
        t0 = time.monotonic()
        output = model.encode(
            texts,
            batch_size=32,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        elapsed = time.monotonic() - t0
        logger.debug("bge_m3_encode", count=len(texts), time_s=round(elapsed, 3))
        return {
            "dense": np.array(output["dense_vecs"], dtype=np.float32),
            "sparse": output["lexical_weights"],  # list[dict[int, float]]
        }

    def _encode_query(self, query: str) -> dict[str, Any]:
        """Encode a single query."""
        result = self._encode([query])
        return {
            "dense": result["dense"][0],
            "sparse": result["sparse"][0],
        }

    # -- Index build ---------------------------------------------------------

    def build_index(self, chunks: list[Chunk] | None = None) -> None:
        """Build both dense (FAISS) and sparse (CSR) indexes from chunks."""
        if chunks is None:
            chunks = list(self.store.iter_chunks())
        if not chunks:
            logger.warning("bge_m3_build_empty")
            return

        t0 = time.monotonic()
        texts = [c.text for c in chunks]
        encoded = self._encode(texts)

        dense_vecs = encoded["dense"]
        sparse_dicts = encoded["sparse"]
        self._chunk_ids = [str(c.id) for c in chunks]
        self._dense_dim = dense_vecs.shape[1]

        # Build dense FAISS index
        faiss.normalize_L2(dense_vecs)
        self._dense_index = faiss.IndexFlatIP(self._dense_dim)
        self._dense_index.add(dense_vecs.astype(np.float32))

        # Build sparse CSR matrix
        self._sparse_matrix = self._build_sparse_csr(sparse_dicts)

        elapsed = time.monotonic() - t0
        self._stats = BGEIndexStats(
            chunk_count=len(chunks),
            dense_dim=self._dense_dim,
            sparse_nnz=self._sparse_matrix.nnz if self._sparse_matrix is not None else 0,
            build_time_s=round(elapsed, 3),
        )

        self._save_index()
        logger.info(
            "bge_m3_index_built",
            chunks=len(chunks),
            dense_dim=self._dense_dim,
            sparse_nnz=self._stats.sparse_nnz,
            build_time_s=self._stats.build_time_s,
        )

    @staticmethod
    def _build_sparse_csr(sparse_dicts: list[dict[int, float]]) -> Any:
        """Convert a list of {token_id: weight} dicts to a scipy CSR matrix."""
        from scipy.sparse import csr_matrix

        rows, cols, data = [], [], []
        for row_idx, d in enumerate(sparse_dicts):
            for token_id, weight in d.items():
                rows.append(row_idx)
                cols.append(int(token_id))
                data.append(float(weight))
        if not data:
            return csr_matrix((0, 0), dtype=np.float32)
        return csr_matrix(
            (data, (rows, cols)),
            shape=(len(sparse_dicts), max(cols) + 1),
            dtype=np.float32,
        )

    # -- Save / Load ---------------------------------------------------------

    def _save_index(self) -> None:
        """Persist dense FAISS index and sparse CSR matrix to disk."""
        if self._dense_index is not None:
            faiss.write_index(self._dense_index, str(self.dense_index_path))
        if self._sparse_matrix is not None:
            import pickle

            data = {
                "matrix": self._sparse_matrix,
                "chunk_ids": self._chunk_ids,
                "dense_dim": self._dense_dim,
            }
            with self.sparse_index_path.open("wb") as f:
                pickle.dump(data, f)
        logger.debug("bge_m3_index_saved")

    def load_index(self) -> bool:
        """Load indexes from disk. Returns True if successful."""
        if not self.dense_index_path.exists() or not self.sparse_index_path.exists():
            return False
        try:
            self._dense_index = faiss.read_index(str(self.dense_index_path))
            import pickle

            with self.sparse_index_path.open("rb") as f:
                data = pickle.load(f)
            self._sparse_matrix = data["matrix"]
            self._chunk_ids = data["chunk_ids"]
            self._dense_dim = data["dense_dim"]
            logger.info(
                "bge_m3_index_loaded",
                chunks=len(self._chunk_ids),
                dense_dim=self._dense_dim,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("bge_m3_index_load_failed", error=str(exc))
            return False

    def ensure_index(self) -> None:
        """Ensure the index is built or loaded."""
        if self._dense_index is None and not self.load_index():
            self.build_index()

    # -- Search --------------------------------------------------------------

    def search_dense(
        self, query: str, top_k: int = 20
    ) -> list[tuple[UUID, float, dict[str, float]]]:
        """Search using dense vectors only.

        Returns list of (chunk_id, score, diagnostics) tuples.
        """
        self.ensure_index()
        if self._dense_index is None or self._dense_index.ntotal == 0:
            return []

        q = self._encode_query(query)
        dense_vec = q["dense"].reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(dense_vec)

        t0 = time.monotonic()
        scores, indices = self._dense_index.search(dense_vec, min(top_k, self._dense_index.ntotal))
        self._stats.search_time_s = round(time.monotonic() - t0, 4)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._chunk_ids) and score > 0:
                chunk_id = UUID(self._chunk_ids[idx])
                diag = {"dense_score": float(score), "sparse_score": 0.0, "fused_score": float(score)}
                results.append((chunk_id, float(score), diag))
        return results

    def search_sparse(
        self, query: str, top_k: int = 20
    ) -> list[tuple[UUID, float, dict[str, float]]]:
        """Search using sparse vectors only.

        Returns list of (chunk_id, score, diagnostics) tuples.
        """
        self.ensure_index()
        if self._sparse_matrix is None or self._sparse_matrix.shape[0] == 0:
            return []

        q = self._encode_query(query)
        sparse_vec = self._dict_to_sparse_row(q["sparse"], self._sparse_matrix.shape[1])

        t0 = time.monotonic()
        # Dot product: (1, vocab) @ (vocab, n_chunks) -> (1, n_chunks)
        scores_dense = (sparse_vec @ self._sparse_matrix.T).toarray().flatten()
        top_indices = np.argsort(scores_dense)[::-1][:top_k]
        self._stats.search_time_s = round(time.monotonic() - t0, 4)

        results = []
        for idx in top_indices:
            score = float(scores_dense[idx])
            if score > 0 and idx < len(self._chunk_ids):
                chunk_id = UUID(self._chunk_ids[idx])
                diag = {"dense_score": 0.0, "sparse_score": score, "fused_score": score}
                results.append((chunk_id, score, diag))
        return results

    def search_hybrid(
        self,
        query: str,
        top_k: int = 20,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> list[tuple[UUID, float, dict[str, float]]]:
        """Search using fused dense + sparse.

        Returns list of (chunk_id, fused_score, diagnostics) tuples.
        """
        self.ensure_index()
        if self._dense_index is None or self._dense_index.ntotal == 0:
            return []

        dw = dense_weight if dense_weight is not None else self.dense_weight
        sw = sparse_weight if sparse_weight is not None else self.sparse_weight
        total_w = dw + sw
        if total_w > 0:
            dw /= total_w
            sw /= total_w

        q = self._encode_query(query)

        # Dense search
        dense_vec = q["dense"].reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(dense_vec)
        dense_scores, dense_indices = self._dense_index.search(
            dense_vec, min(top_k * 2, self._dense_index.ntotal)
        )
        dense_map: dict[int, float] = {}
        for score, idx in zip(dense_scores[0], dense_indices[0]):
            if 0 <= idx and score > 0:
                dense_map[int(idx)] = float(score)

        # Sparse search
        sparse_vec = self._dict_to_sparse_row(q["sparse"], self._sparse_matrix.shape[1])
        sparse_scores_dense = (sparse_vec @ self._sparse_matrix.T).toarray().flatten()
        sparse_map: dict[int, float] = {}
        for idx in range(len(self._chunk_ids)):
            s = float(sparse_scores_dense[idx])
            if s > 0:
                sparse_map[idx] = s

        # Normalize and fuse
        max_dense = max(dense_map.values()) if dense_map else 1.0
        max_sparse = max(sparse_map.values()) if sparse_map else 1.0

        all_indices = set(dense_map.keys()) | set(sparse_map.keys())
        fused: list[tuple[int, float, float, float, float]] = []
        for idx in all_indices:
            d_norm = dense_map.get(idx, 0.0) / max_dense if max_dense > 0 else 0.0
            s_norm = sparse_map.get(idx, 0.0) / max_sparse if max_sparse > 0 else 0.0
            fused_score = dw * d_norm + sw * s_norm
            fused.append((idx, fused_score, dense_map.get(idx, 0.0), sparse_map.get(idx, 0.0), fused_score))

        fused.sort(key=lambda x: x[1], reverse=True)
        top = fused[:top_k]

        t0 = time.monotonic()
        self._stats.search_time_s = round(time.monotonic() - t0, 4)

        results = []
        for idx, fs, ds, ss, _ in top:
            chunk_id = UUID(self._chunk_ids[idx])
            diag = {"dense_score": ds, "sparse_score": ss, "fused_score": fs}
            results.append((chunk_id, fs, diag))
        return results

    # -- Convenience: search returning EvidenceRefs --------------------------

    def search_as_refs(
        self,
        query: str,
        top_k: int = 20,
        mode: str = "hybrid",
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> list[EvidenceRef]:
        """Search and return EvidenceRef objects with full diagnostics.

        Modes: 'dense', 'sparse', 'hybrid'.
        """
        if mode == "dense":
            raw = self.search_dense(query, top_k)
        elif mode == "sparse":
            raw = self.search_sparse(query, top_k)
        else:
            raw = self.search_hybrid(query, top_k, dense_weight, sparse_weight)

        if not raw:
            return []

        chunk_ids = [cid for cid, _, _ in raw]
        scores = [s for _, s, _ in raw]
        diagnostics = [d for _, _, d in raw]

        refs = self.store.get_evidence_refs(chunk_ids, scores)
        for ref, diag in zip(refs, diagnostics):
            ref.metadata["bge_m3_dense_score"] = diag["dense_score"]
            ref.metadata["bge_m3_sparse_score"] = diag["sparse_score"]
            ref.metadata["bge_m3_fused_score"] = diag["fused_score"]
            ref.metadata["retrieval_method"] = "bge_m3"
        return refs

    @staticmethod
    def _dict_to_sparse_row(d: dict[int, float], width: int) -> Any:
        """Convert a {token_id: weight} dict to a (1, width) scipy sparse row."""
        from scipy.sparse import csr_matrix

        if not d:
            return csr_matrix((1, width), dtype=np.float32)
        cols = [int(k) for k in d.keys()]
        vals = [float(v) for v in d.values()]
        max_col = max(cols)
        if max_col >= width:
            width = max_col + 1
        return csr_matrix((vals, ([0] * len(cols), cols)), shape=(1, width), dtype=np.float32)

    # -- Stats / diagnostics -------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        return {
            "model": self.model_name,
            "chunk_count": len(self._chunk_ids),
            "dense_dim": self._dense_dim,
            "dense_index_ntotal": self._dense_index.ntotal if self._dense_index else 0,
            "sparse_nnz": self._sparse_matrix.nnz if self._sparse_matrix is not None else 0,
            "build_time_s": self._stats.build_time_s,
            "search_time_s": self._stats.search_time_s,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
        }

    def get_vram_usage(self) -> dict[str, Any]:
        """Report GPU VRAM usage if available."""
        try:
            import torch

            if torch.cuda.is_available():
                return {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "vram_allocated_mb": round(
                        torch.cuda.memory_allocated(0) / 1024 / 1024, 1
                    ),
                    "vram_reserved_mb": round(
                        torch.cuda.memory_reserved(0) / 1024 / 1024, 1
                    ),
                }
        except ImportError:
            pass
        return {"gpu": None, "note": "CPU-only mode"}
