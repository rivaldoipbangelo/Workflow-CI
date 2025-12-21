import os

import mlflow
import mlflow.sklearn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


DATA_DIR = os.path.join(os.path.dirname(__file__), "spiderman_youtube_review_preprocessing")
DATA_FILE = "spiderman_youtube_review_preprocessed.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

EXPERIMENT_NAME = "spiderman_sentiment_ci"


def load_data():
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Dataset kosong! Cek kembali file hasil preprocessing.")
    X = df["clean_review"].astype(str)
    y = df["sentiment_label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def main():
    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data()

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ("logreg", LogisticRegression(max_iter=1000, solver="liblinear")),
        ]
    )

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="logreg_ci"):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[RESULT] Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()