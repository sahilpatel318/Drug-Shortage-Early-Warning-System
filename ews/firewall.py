"""
Ground truth firewall.

The onset labels are the answer key. They are sealed inside a Vault. The
feature path receives only the observable panel and the Vault handle needed to
build labels for a *fully observed* window. The runtime leakage test proves
that features anchored at month t do not depend on any onset at month t or
later, which is the property that would otherwise inflate an early warning
score.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import config


class SealedError(RuntimeError):
    pass


class Vault:
    """Holds onset labels. Feature builders may ask for a label window that is
    strictly in the future relative to the anchor, and only that."""

    def __init__(self, onset: np.ndarray):
        self._onset = onset.copy()
        self._onset.setflags(write=False)
        self._fingerprint = hashlib.sha256(self._onset.tobytes()).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def shape(self):
        return self._onset.shape

    def label_window(self, drug_idx: int, anchor_month: int, horizon: int) -> int:
        """Return 1 if an onset occurs in (anchor_month, anchor_month+horizon].

        The window is strictly after the anchor, so the label describes the
        future the model is asked to predict. It never reveals the present.
        """
        start = anchor_month + 1
        end = anchor_month + horizon
        if end >= self._onset.shape[1]:
            raise SealedError("label window extends past the observed horizon")
        return int(self._onset[drug_idx, start:end + 1].max() > 0)

    def onset_months(self, drug_idx: int) -> np.ndarray:
        return np.flatnonzero(self._onset[drug_idx] > 0)


def leakage_test(build_features_fn, panel, vault, seed: int = config.SEED) -> dict:
    """Prove features do not read the future.

    Strategy: build the feature matrix twice. The second time, scramble every
    onset strictly after each row's anchor month. If any feature peeked into
    the future, at least one feature value would change. We assert the two
    matrices are byte identical for the feature columns.
    """
    rng = np.random.RandomState(seed + 7)

    base = build_features_fn(panel, vault)

    # Build a corrupted vault where future onsets are randomly flipped.
    corrupted = vault._onset.copy()
    n, t = corrupted.shape
    flip = rng.rand(n, t) < 0.5
    corrupted = np.where(flip, 1 - corrupted, corrupted).astype(np.int32)
    corrupted_vault = Vault(corrupted)

    # The corrupted vault must keep the SAME observable panel. Only the future
    # section of the answer key is scrambled. Features must not move.
    corrupt = build_features_fn(panel, corrupted_vault)

    feat_cols = [c for c in base.columns if c not in ("y", "drug_idx", "anchor_month")]
    a = base[feat_cols].to_numpy()
    b = corrupt[feat_cols].to_numpy()

    same_shape = a.shape == b.shape
    identical = bool(same_shape and np.array_equal(a, b))
    # Labels SHOULD differ because we scrambled the future answer key.
    labels_moved = bool(not np.array_equal(base["y"].to_numpy(),
                                           corrupt["y"].to_numpy()))

    result = {
        "features_independent_of_future": identical,
        "labels_depend_on_future": labels_moved,
        "n_rows": int(a.shape[0]),
        "n_feature_cols": int(len(feat_cols)),
        "vault_fingerprint": vault.fingerprint,
    }
    if not identical:
        raise SealedError("LEAKAGE DETECTED: features changed when future was scrambled")
    if not labels_moved:
        raise SealedError("label sanity failed: labels did not react to future scramble")
    return result
