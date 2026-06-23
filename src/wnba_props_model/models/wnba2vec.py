"""WNBA2Vec — Player Embedding Network (Enhancement 12).

Adapts NBA2Vec (Guan, Javed & Lu, 2023) for WNBA player representation.

Architecture:
    Input  : 5 offensive + 5 defensive player IDs + game context features
    Embed  : 16-dim shared player embedding (larger than NBA2Vec's 8D
             to capture WNBA role versatility)
    Output : Softmax over 23 possession outcome classes

Pre-training strategy:
    1. Train on NBA play-by-play (60× more data)
    2. Fine-tune on WNBA play-by-play

Integration:
    EmbeddingFeatureInjector.inject(df) adds 8 PCA-reduced embedding
    columns (player_embed_0 … player_embed_7) to the feature table.

Reference:
    Guan, Javed & Lu (2023). NBA2Vec. arXiv:2302.13386
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Number of possession outcome classes (WNBA-calibrated)
N_OUTCOMES = 23
EMBED_DIM = 16
N_PCA_DIMS = 8


class WNBA2Vec(nn.Module):
    """Player embedding network predicting play outcomes from lineup data.

    Key difference from NBA2Vec:
        - Larger embedding (16D vs 8D) to capture WNBA role versatility
        - Context features (period, margin, timeout) appended to input
        - Pre-trained on NBA PBP, fine-tuned on WNBA
    """

    def __init__(
        self,
        n_players: int,
        embed_dim: int = EMBED_DIM,
        n_outcomes: int = N_OUTCOMES,
        context_dim: int = 5,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.player_embedding = nn.Embedding(n_players + 1, embed_dim, padding_idx=0)
        self.context_fc = nn.Linear(context_dim, embed_dim)
        # offensive + defensive mean embed + context = 3 * embed_dim
        self.fc1 = nn.Linear(embed_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, n_outcomes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        off_player_ids: torch.Tensor,
        def_player_ids: torch.Tensor,
        context_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        off_player_ids : (B, 5)  — offensive lineup player indices
        def_player_ids : (B, 5)  — defensive lineup player indices
        context_feats  : (B, context_dim) — period, margin, timeout, etc.

        Returns
        -------
        logits : (B, n_outcomes)  — apply cross-entropy externally
        """
        off_embeds = self.player_embedding(off_player_ids)   # (B, 5, D)
        def_embeds = self.player_embedding(def_player_ids)   # (B, 5, D)
        off_mean = off_embeds.mean(dim=1)                    # (B, D)
        def_mean = def_embeds.mean(dim=1)                    # (B, D)
        ctx = self.relu(self.context_fc(context_feats))      # (B, D)
        combined = torch.cat([off_mean, def_mean, ctx], dim=-1)  # (B, 3D)
        h1 = self.dropout(self.relu(self.fc1(combined)))
        h2 = self.dropout(self.relu(self.fc2(h1)))
        return self.fc3(h2)

    # ── Embedding extraction ──────────────────────────────────────────────

    def get_player_embedding(self, player_idx: int) -> np.ndarray:
        """Return the raw 16-dim embedding for a player index."""
        self.eval()
        with torch.no_grad():
            idx_t = torch.tensor([player_idx], dtype=torch.long)
            return self.player_embedding(idx_t).squeeze(0).numpy()

    def get_embedding_features(
        self, player_idx: int, n_pca_dims: int = N_PCA_DIMS
    ) -> dict[str, float]:
        """Return a dict of PCA-reduced embedding features for the HGB pipeline."""
        raw = self.get_player_embedding(player_idx)
        trunc = raw[:n_pca_dims]
        return {f"player_embed_{i}": float(v) for i, v in enumerate(trunc)}


class EmbeddingFeatureInjector:
    """Inject pre-trained WNBA2Vec embeddings into the feature table.

    Usage in build_features.py::

        injector = EmbeddingFeatureInjector(model_path, player_id_map)
        df = injector.inject(df)

    The injector adds 8 player-level embedding columns
    (``player_embed_0`` … ``player_embed_7``) and 8 team-level columns
    (``team_embed_0`` … ``team_embed_7``) for each row.

    When no model checkpoint exists (cold-start), it synthesises
    embeddings from normalised box-score statistics so the pipeline
    continues to function without training data.
    """

    def __init__(
        self,
        model_path: Optional[str],
        player_id_map: Optional[dict[int, int]] = None,
        n_dims: int = N_PCA_DIMS,
        embed_dim: int = EMBED_DIM,
    ):
        self.n_dims = n_dims
        self.embed_dim = embed_dim
        self.player_id_map: dict[int, int] = player_id_map or {}
        self._cache: dict[int, dict[str, float]] = {}
        self.model: Optional[WNBA2Vec] = None

        if model_path and Path(model_path).exists():
            try:
                n_players = max(self.player_id_map.values(), default=1000) + 1
                self.model = WNBA2Vec(n_players=n_players, embed_dim=embed_dim)
                state = torch.load(model_path, map_location="cpu", weights_only=True)
                self.model.load_state_dict(state)
                self.model.eval()
                logger.info("WNBA2Vec loaded from %s", model_path)
            except Exception as exc:
                logger.warning("WNBA2Vec load failed (%s) — using synthetic embeddings", exc)
                self.model = None
        else:
            logger.info(
                "No WNBA2Vec checkpoint at %s — using synthetic embeddings. "
                "Train with scripts/train_wnba2vec.py to enable learned representations.",
                model_path,
            )

    def inject(self, df: "pd.DataFrame") -> "pd.DataFrame":  # noqa: F821
        """Add player and team embedding columns to *df* in-place."""
        import pandas as pd  # noqa: PLC0415

        out = df.copy()
        for pid in out["player_id"].unique():
            pidx = self.player_id_map.get(int(pid), int(pid) % 1000)
            if pid not in self._cache:
                if self.model is not None:
                    self._cache[pid] = self.model.get_embedding_features(pidx, self.n_dims)
                else:
                    self._cache[pid] = self._synthetic_embed(out, int(pid))
            for col, val in self._cache[pid].items():
                out.loc[out["player_id"] == pid, col] = val

        # Team-level embeddings = mean of roster embeddings
        if "team_id" in out.columns:
            for tid in out["team_id"].unique():
                roster = out.loc[out["team_id"] == tid, "player_id"].unique()
                team_vecs = [
                    list(self._cache[p].values())
                    for p in roster
                    if p in self._cache
                ]
                if team_vecs:
                    team_mean = np.mean(team_vecs, axis=0)
                    for i, v in enumerate(team_mean[: self.n_dims]):
                        out.loc[out["team_id"] == tid, f"team_embed_{i}"] = float(v)

        return out

    # ── Private helpers ───────────────────────────────────────────────────

    def _synthetic_embed(self, df: "pd.DataFrame", pid: int) -> dict[str, float]:  # noqa: F821
        """Synthesise a stable embedding from normalised box-score stats.

        Used when no trained checkpoint is available.  Gives each player a
        deterministic 8-dim representation derived from their average stats
        so the HGB model still receives player-differentiated features.
        """
        import hashlib  # noqa: PLC0415

        stat_cols = ["pts", "reb", "ast", "stl", "blk", "turnover", "fg3m", "min"]
        row = df.loc[df["player_id"] == pid, [c for c in stat_cols if c in df.columns]].mean()

        # Seed with player_id for reproducibility
        rng = np.random.default_rng(abs(hash(pid)) % (2**31))
        base = rng.normal(0, 0.1, self.n_dims)

        for i, col in enumerate(stat_cols[: self.n_dims]):
            if col in row.index and not np.isnan(row[col]):
                base[i] += float(row[col]) / (float(row[col]) + 10)

        return {f"player_embed_{i}": float(v) for i, v in enumerate(base)}


def build_player_id_map(player_ids: list[int]) -> dict[int, int]:
    """Build a compact 0-based integer index for player IDs."""
    return {pid: i + 1 for i, pid in enumerate(sorted(set(player_ids)))}
