"""Shared setup for every section notebook.

Frozen on Nov 20: don't edit this file or clean_songs.csv. If something looks
wrong, tell the Producer instead of changing it, so everyone keeps the same data.
"""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit

ROOT = Path(__file__).resolve().parent

FEATURES = ["duration_ms", "explicit", "danceability", "energy", "key", "loudness", "mode",
            "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo",
            "time_signature"]


def load():
    """Load the frozen data.

    d.songs    one row per unique song, with `split` (train/test) and `fold` (0-4, train rows only)
    d.X_train, d.y_train, d.X_test, d.y_test
    d.cv       the shared 5-fold cross-validation splitter (folds grouped by artist)
    """
    songs = pd.read_csv(ROOT / "clean_songs.csv")
    X = pd.get_dummies(songs[FEATURES + ["broad_genre"]], columns=["broad_genre"])
    y = songs["failure"]
    train = songs["split"] == "train"
    return SimpleNamespace(
        songs=songs,
        X_train=X[train], y_train=y[train],
        X_test=X[~train], y_test=y[~train],
        cv=PredefinedSplit(songs.loc[train, "fold"]),
    )


def final_model():
    """The frozen final model (unfitted). Fit it on d.X_train, d.y_train."""
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)
