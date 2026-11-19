"""Shared setup for every section notebook.

Frozen on Nov 20: don't edit this file or clean_songs.csv. If something looks
off, post in the group chat instead of changing it, so everyone keeps the same data.
"""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent

GENRES = ["electronic", "hiphop/rnb", "latin", "pop", "rock"]

FEATURES = ["duration_ms", "explicit", "danceability", "energy", "key", "loudness", "mode",
            "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo",
            "time_signature"]


def load():
    """Load the frozen data. The target is the song's parent genre.

    d.songs    one row per unique song, with `split` (train/test) and `fold` (0-4, train rows only)
    d.X_train, d.y_train, d.X_test, d.y_test
    d.cv       the shared 5-fold cross-validation splitter (folds grouped by artist)
    """
    songs = pd.read_csv(ROOT / "clean_songs.csv")
    X, y = songs[FEATURES], songs["broad_genre"]
    train = songs["split"] == "train"
    return SimpleNamespace(
        songs=songs,
        X_train=X[train], y_train=y[train],
        X_test=X[~train], y_test=y[~train],
        cv=PredefinedSplit(songs.loc[train, "fold"]),
    )


# The two frozen models (unfitted). Fit them on d.X_train, d.y_train.

def logistic_regression():
    """Simple: logistic regression on scaled features."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def random_forest():
    """Complex: many decision trees that vote."""
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)


MODELS = {
    "logistic_regression": logistic_regression,
    "random_forest": random_forest,
}
