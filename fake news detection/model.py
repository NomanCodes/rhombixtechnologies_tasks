"""
Train a Fake News detector and save model.pkl

Usage:
    python model.py
    python model.py --csv news.csv --algo logreg
    python model.py --csv news.csv --algo nb
    python model.py --csv news.csv --text_col headline --label_col target
"""

import argparse
import json
import os
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


SAMPLE_ROWS = [
    ("Government announces new education policy", "REAL"),
    ("Scientists discover water on Mars", "REAL"),
    ("Aliens abducted the prime minister last night", "FAKE"),
    ("Man claims eating chocolate cures all diseases", "FAKE"),
    ("Stock market reaches record high today", "REAL"),
    ("World will end tomorrow, experts say", "FAKE"),
    ("Health ministry launches national vaccination drive", "REAL"),
    ("Secret potion guarantees instant millionaire status", "FAKE"),
    ("Local team wins the regional football championship", "REAL"),
    ("Time traveler warns of robot invasion next week", "FAKE"),
]


def ensure_dataset(csv_path: str, text_col: str, label_col: str) -> None:
    """Create a tiny sample dataset if csv_path does not exist."""
    if os.path.exists(csv_path):
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df = pd.DataFrame(SAMPLE_ROWS, columns=[text_col, label_col])
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[info] '{csv_path}' not found. Created a small sample dataset with {len(df)} rows.")


def build_pipeline(algorithm: str, small_corpus: bool) -> Pipeline:
    """
    Create TF-IDF + classifier pipeline.
    For small corpora, relax pruning to avoid 'After pruning, no terms remain' errors.
    """
    if small_corpus:
        # Very permissive for tiny datasets
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            stop_words=None,
            lowercase=True,
        )
    else:
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            stop_words="english",
            lowercase=True,
        )

    algo = algorithm.lower()
    if algo == "nb":
        clf = MultinomialNB()
    elif algo in ("logreg", "logistic", "logistic_regression"):
        clf = LogisticRegression(
            C=2.0,
            penalty="l2",
            class_weight="balanced",
            max_iter=2000,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm} (use 'logreg' or 'nb')")

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def compute_metrics(y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted")
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1w),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
    }


def train(
    csv_path: str = "news.csv",
    text_col: str = "text",
    label_col: str = "label",
    algorithm: str = "logreg",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, dict]:
    ensure_dataset(csv_path, text_col, label_col)

    df = pd.read_csv(csv_path, encoding="utf-8").dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].astype(str).str.strip()

    X = df[text_col].values
    y = df[label_col].values

    # Info
    print(f"[info] Loaded {len(df)} rows from {csv_path}")
    print("[info] Label distribution:")
    try:
        print(df[label_col].value_counts(dropna=False).to_string())
    except Exception:
        pass

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Decide if corpus is small
    small_corpus = len(X_train) < 200
    pipe = build_pipeline(algorithm, small_corpus=small_corpus)

    # Fit with auto-recovery if pruning happens
    try:
        pipe.fit(X_train, y_train)
    except ValueError as e:
        msg = str(e).lower()
        if "after pruning, no terms remain" in msg:
            print("[warn] Pruning removed all terms. Retrying with very permissive TF-IDF...")
            pipe = build_pipeline(algorithm, small_corpus=True)
            pipe.fit(X_train, y_train)
        else:
            raise

    y_pred = pipe.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    # Save artifacts
    joblib.dump(pipe, "model.pkl")
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[ok] Saved model.pkl and metrics.json")
    print(json.dumps(metrics, indent=2))
    return pipe, metrics


def parse_args():
    ap = argparse.ArgumentParser(description="Train Fake News model and save model.pkl")
    ap.add_argument("--csv", default="news.csv", help="Path to CSV file")
    ap.add_argument("--text_col", default="text", help="Text column name")
    ap.add_argument("--label_col", default="label", help="Label column name")
    ap.add_argument(
        "--algo",
        default="logreg",
        choices=["logreg", "nb"],
        help="Classifier: logreg (Logistic Regression) or nb (MultinomialNB)",
    )
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        csv_path=args.csv,
        text_col=args.text_col,
        label_col=args.label_col,
        algorithm=args.algo,
        test_size=args.test_size,
        random_state=args.random_state,
    )
