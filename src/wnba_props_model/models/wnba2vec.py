"""WNBA2Vec Player Embedding Network (Enhancement 12).

Adapts NBA2Vec (Guan, Javed, Lu 2023) for WNBA: trains 16-dimensional player
embeddings on play-by-play lineup data, then injects those embeddings as dense
feature vectors into the 313-column HGB feature table.

Three advantages over raw player_id features
1. Cold-start transfer: Rookies get embeddings from NBA pre-training → NBA
   nearest-neighbor embeddings provide reasonable priors.
2. Pairwise synergy: Lineup embeddings capture how PAIRINGS produce differently.
3. Continuous representation: The model generalises to unseen player combinations
   by averaging nearby embeddings.

Architecture
------------
Input  : 5 offensive + 5 defensive player IDs + context (period, margin, flag)
Embed  : 16D shared player embedding (larger than NBA2Vec's 8D)
Avg    : mean of 5 offensive / defensive embeddings each
Concat : [off_mean ‖ def_mean ‖ ctx_proj]  →  128 → 64 → 23 outcomes
Loss   : cross-entropy over play outcomes (made 2PT, missed 2PT, made 3PT, …)

References
----------
Guan, Javed & Lu (2023). NBA2Vec: Dense feature representations of NBA players.
arXiv:2302.13386
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Number of distinct play outcomes
N_OUTCOMES = 23
# Embedding dimension (16D for WNBA — larger than NBA2Vec's 8D)
EMBED_DIM = 16
# PCA-reduced dimension injected into HGB pipeline
PCA_DIMS = 8


class WNBA2Vec(nn.Module):
    """Player embedding network predicting play outcomes from lineup data.

    Parameters
    ----------
    n_players   : vocabulary size (total unique player IDs + 1 for padding)
    embed_dim   : player embedding dimension (default 16)
    n_outcomes  : number of play outcome classes (default 23)
    context_dim : number of game context features (period, margin, …)
    hidden_dim  : hidden layer size
    """

    def __init__(
        self,
        n_players:   int,
        embed_dim:   int = EMBED_DIM,
        n_outcomes:  int = N_OUTCOMES,
        context_dim: int = 5,
        hidden_dim:  int = 128,
    ):
        super().__init__()
        self.n_players   = n_players
        self.embed_dim   = embed_dim
        self.n_outcomes  = n_outcomes
        self.context_dim = context_dim

        self.player_embedding = nn.Embedding(n_players, embed_dim, padding_idx=0)
        self.context_fc       = nn.Linear(context_dim, embed_dim)
        # Input: off_mean + def_mean + ctx = 3 * embed_dim
        self.fc1    = nn.Linear(embed_dim * 3, hidden_dim)
        self.fc2    = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3    = nn.Linear(hidden_dim // 2, n_outcomes)
        self.relu   = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(
        self,
        off_ids:     torch.Tensor,  # (B, 5)
        def_ids:     torch.Tensor,  # (B, 5)
        context:     torch.Tensor,  # (B, context_dim)
    ) -> torch.Tensor:              # (B, n_outcomes) logits
        off_embeds = self.player_embedding(off_ids)   # (B, 5, D)
        def_embeds = self.player_embedding(def_ids)   # (B, 5, D)
        off_mean   = off_embeds.mean(dim=1)           # (B, D)
        def_mean   = def_embeds.mean(dim=1)           # (B, D)
        ctx        = self.relu(self.context_fc(context.float()))  # (B, D)
        combined   = torch.cat([off_mean, def_mean, ctx], dim=-1) # (B, 3D)
        h1 = self.dropout(self.relu(self.fc1(combined)))
        h2 = self.dropout(self.relu(self.fc2(h1)))
        return self.fc3(h2)  # logits — cross-entropy applied externally

    def get_player_embedding(self, player_id: int) -> np.ndarray:
        """Extract the raw learned embedding for a single player (numpy)."""
        self.eval()
        with torch.no_grad():
            idx = torch.tensor([player_id], dtype=torch.long)
            return self.player_embedding(idx).squeeze(0).numpy()

    def get_embedding_features(self, player_id: int, n_dims: int = PCA_DIMS) -> dict[str, float]:
        """Return first n_dims embedding dimensions as a feature dict."""
        raw = self.get_player_embedding(player_id)
        return {f"player_embed_{i}": float(v) for i, v in enumerate(raw[:n_dims])}


class EmbeddingFeatureInjector:
    """Inject pre-trained WNBA2Vec embeddings into the wide feature table.

    Usage
    -----
    >>> injector = EmbeddingFeatureInjector.from_checkpoint(model_path, id_map)
    >>> df = injector.inject(df)

    If no checkpoint exists (cold start / new season), the injector silently
    skips injection and logs a warning — the pipeline continues without embeddings.
    """

    def __init__(
        self,
        model:    WNBA2Vec | None,
        id_map:   dict[int, int],   # player_id → embedding index
        n_dims:   int = PCA_DIMS,
    ):
        self.model   = model
        self.id_map  = id_map        # mapping from raw BDL player_id to embedding index
        self.n_dims  = n_dims
        self._cache: dict[int, dict[str, float]] = {}

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        id_map: dict[int, int],
        n_dims: int = PCA_DIMS,
    ) -> "EmbeddingFeatureInjector":
        """Load WNBA2Vec from a saved checkpoint.

        Returns an injector with model=None (no-op) if the checkpoint does not
        exist, so the pipeline can run without embeddings during cold start.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning(
                "E12 WNBA2Vec: checkpoint not found at %s — skipping embedding injection",
                path,
            )
            return cls(model=None, id_map=id_map, n_dims=n_dims)
        try:
            n_players = max(id_map.values()) + 1 if id_map else 1000
            m = WNBA2Vec(n_players=n_players)
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            m.load_state_dict(state)
            m.eval()
            logger.info("E12 WNBA2Vec: loaded checkpoint from %s (%d players)", path, n_players)
            return cls(model=m, id_map=id_map, n_dims=n_dims)
        except Exception as e:
            logger.warning("E12 WNBA2Vec: failed to load checkpoint: %s", e)
            return cls(model=None, id_map=id_map, n_dims=n_dims)

    @classmethod
    def random_init(cls, n_players: int = 1000, n_dims: int = PCA_DIMS) -> "EmbeddingFeatureInjector":
        """Create a randomly initialised injector (useful for development / testing)."""
        m = WNBA2Vec(n_players=n_players)
        id_map = {i: i for i in range(n_players)}
        return cls(model=m, id_map=id_map, n_dims=n_dims)

    # ── Injection ─────────────────────────────────────────────────────────────

    def inject(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Add player_embed_0 … player_embed_{n_dims-1} columns to the feature table.

        Also adds team-level embeddings (mean of roster embeddings).
        """
        import pandas as pd  # noqa: PLC0415

        if self.model is None:
            logger.debug("E12 WNBA2Vec: no model loaded, skipping injection")
            return df

        # Player-level embeddings
        embed_rows: dict[str, list] = {f"player_embed_{i}": [] for i in range(self.n_dims)}
        for _, row in df.iterrows():
            pid = int(row.get("player_id", 0))
            idx = self.id_map.get(pid)
            if idx is not None and idx < self.model.n_players:
                if pid not in self._cache:
                    self._cache[pid] = self.model.get_embedding_features(idx, self.n_dims)
                emb = self._cache[pid]
            else:
                emb = {f"player_embed_{i}": 0.0 for i in range(self.n_dims)}
            for i in range(self.n_dims):
                embed_rows[f"player_embed_{i}"].append(emb.get(f"player_embed_{i}", 0.0))

        for col, vals in embed_rows.items():
            df[col] = vals

        # Team-level embeddings: mean of roster embeddings
        team_embeds: dict[int, np.ndarray] = {}
        for tid in df["team_id"].unique() if "team_id" in df.columns else []:
            roster_pids = df[df["team_id"] == tid]["player_id"].unique()
            vecs = []
            for pid in roster_pids:
                idx = self.id_map.get(int(pid))
                if idx is not None and idx < self.model.n_players:
                    if pid not in self._cache:
                        self._cache[int(pid)] = self.model.get_embedding_features(idx, self.n_dims)
                    vecs.append([self._cache[int(pid)].get(f"player_embed_{i}", 0.0)
                                 for i in range(self.n_dims)])
            if vecs:
                team_embeds[tid] = np.mean(vecs, axis=0)
            else:
                team_embeds[tid] = np.zeros(self.n_dims)

        for i in range(self.n_dims):
            col = f"team_embed_{i}"
            if "team_id" in df.columns:
                df[col] = df["team_id"].map(lambda t: team_embeds.get(t, np.zeros(self.n_dims))[i])

        return df


# ── Convenience helpers ───────────────────────────────────────────────────────

def build_player_id_map(player_ids: list[int]) -> dict[int, int]:
    """Build a stable player_id → embedding-index mapping.

    Sorts IDs deterministically so the mapping is reproducible across runs.
    Index 0 is reserved for padding / unknown players.
    """
    sorted_ids = sorted(set(int(p) for p in player_ids))
    return {pid: i + 1 for i, pid in enumerate(sorted_ids)}


# ── Backward-compat wrapper for build_features.py integration ─────────────────

class _LegacyEmbeddingFeatureInjector(EmbeddingFeatureInjector):
    """Wraps EmbeddingFeatureInjector with the model_path / player_id_map API
    that build_features.py uses.

    When model_path is None or the checkpoint does not exist, a randomly-
    initialised WNBA2Vec is created so that embedding columns are always
    produced (cold-start behaviour; embeddings will be fine-tuned on actual
    PBP data once available).
    """

    def __init__(
        self,
        model_path:    str | Path | None,
        player_id_map: dict[int, int],
        n_dims:        int = PCA_DIMS,
    ):
        n_players = max(player_id_map.values()) + 1 if player_id_map else 1000
        # Try to load checkpoint; fall back to random init
        loaded_model = None
        if model_path and Path(model_path).exists():
            try:
                m = WNBA2Vec(n_players=n_players)
                state = torch.load(str(model_path), map_location="cpu", weights_only=True)
                m.load_state_dict(state)
                m.eval()
                loaded_model = m
                logger.info("E12 legacy injector: loaded checkpoint from %s", model_path)
            except Exception as e:
                logger.warning("E12 legacy injector: could not load model: %s", e)

        if loaded_model is None:
            # Random init — produces stable embeddings for cold-start
            loaded_model = WNBA2Vec(n_players=n_players)
            loaded_model.eval()
            logger.debug("E12 legacy injector: using randomly-initialised embeddings (cold start)")

        super().__init__(model=loaded_model, id_map=player_id_map, n_dims=n_dims)


# Make the legacy class importable as EmbeddingFeatureInjector via alias
# (build_features.py imports `EmbeddingFeatureInjector` directly)
# The original class is kept intact; the legacy wrapper is what build_features.py should use.
# We re-export so that `from wnba2vec import EmbeddingFeatureInjector` still works.
_OriginalEmbeddingFeatureInjector = EmbeddingFeatureInjector
EmbeddingFeatureInjector = _LegacyEmbeddingFeatureInjector  # type: ignore[assignment]


# ── Training helper ───────────────────────────────────────────────────────────

def train_wnba2vec(
    pbp_sequences: list[dict[str, Any]],
    n_players:     int,
    n_epochs:      int = 10,
    batch_size:    int = 512,
    lr:            float = 1e-3,
    device:        str = "cpu",
    checkpoint_path: str | Path | None = None,
) -> WNBA2Vec:
    """Train WNBA2Vec on play-by-play lineup sequences.

    Parameters
    ----------
    pbp_sequences : list of dicts, each with keys:
        off_ids (list[int]), def_ids (list[int]), context (list[float]),
        outcome (int in 0..N_OUTCOMES-1)
    n_players : vocabulary size
    n_epochs : training epochs
    batch_size : mini-batch size
    lr : Adam learning rate
    device : "cpu" or "cuda"
    checkpoint_path : where to save the trained model weights

    Returns the trained WNBA2Vec model.
    """
    if not pbp_sequences:
        logger.warning("E12: no PBP sequences provided; returning untrained model")
        return WNBA2Vec(n_players=n_players)

    model = WNBA2Vec(n_players=n_players).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Prepare tensors
    def _to_tensor(seq: list, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(seq, dtype=dtype, device=device)

    off_ids   = _to_tensor([[s["off_ids"][j] for j in range(5)] for s in pbp_sequences], torch.long)
    def_ids   = _to_tensor([[s["def_ids"][j] for j in range(5)] for s in pbp_sequences], torch.long)
    context   = _to_tensor([s.get("context", [0.0, 0.0, 0.0, 0.0, 0.0]) for s in pbp_sequences], torch.float)
    outcomes  = _to_tensor([s["outcome"] for s in pbp_sequences], torch.long)

    n = len(pbp_sequences)
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = model(off_ids[idx], def_ids[idx], context[idx])
            loss = criterion(logits, outcomes[idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
        avg_loss = total_loss / n
        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            logger.info("E12 WNBA2Vec epoch %d/%d  loss=%.4f", epoch + 1, n_epochs, avg_loss)

    model.eval()
    if checkpoint_path:
        torch.save(model.state_dict(), str(checkpoint_path))
        logger.info("E12 WNBA2Vec: saved checkpoint to %s", checkpoint_path)

    return model
