# train_model.py
import argparse
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pathlib
from sklearn.metrics import accuracy_score, classification_report
import random

random.seed(42)


def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def df_is_no_header(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        first = next(r, None)
        if first is None:
            return False
        return len(first) == 2 and len(first[1]) <= 12 and "text" not in first[0].lower()


def ensure_min_class_counts(df, label_col="emotion", min_count=2):
    counts = df[label_col].value_counts().to_dict()
    new_rows = df.copy()
    for label, count in counts.items():
        if count < min_count:
            # duplicate rows randomly to reach at least min_count
            needed = min_count - count
            sample_rows = df[df[label_col] == label]
            for _ in range(needed):
                new_rows = pd.concat([new_rows, sample_rows.sample(1, replace=True)])
    return new_rows.reset_index(drop=True)


def main(data_path):
    if df_is_no_header(data_path):
        df = pd.read_csv(data_path, header=None, names=["text", "emotion"])
    else:
        df = pd.read_csv(data_path)

    if "text" not in df.columns or "emotion" not in df.columns:
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ["text", "emotion"]
        else:
            raise SystemExit("Data must have two columns: text,emotion")

    df = df.dropna(subset=["text", "emotion"]).reset_index(drop=True)
    df["text"] = df["text"].apply(clean_text)

    # Duplicate rare classes
    df = ensure_min_class_counts(df)
    print("⚠ Small-class augmentation performed (duplicated rows so stratify can run).")
    print("Dataset class counts:", df["emotion"].value_counts().to_dict())

    X = df["text"].values
    y = df["emotion"].values

    # Automatically pick test size based on dataset size
    n_classes = len(set(y))
    n_samples = len(y)
    min_test_size = max(0.3, n_classes / n_samples + 0.05)  # small dataset protection
    min_test_size = min(min_test_size, 0.5)  # never more than 50%
    print(f"Adjusted test_size = {min_test_size:.2f}")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=min_test_size, random_state=42, stratify=y
        )
    except ValueError:
        print("⚠ Stratify failed, running simple split (tiny dataset).")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=min_test_size, random_state=42
        )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ("clf", MultinomialNB())
    ])

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred) * 100
    test_acc = accuracy_score(y_test, y_test_pred) * 100

    print(f"\nTrain accuracy: {train_acc:.2f}%")
    print(f"Test  accuracy: {test_acc:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred, zero_division=0))

    p = pathlib.Path("models")
    p.mkdir(parents=True, exist_ok=True)
    out = p / "model.pkl"
    joblib.dump(pipeline, out)
    print("\n✅ Model saved successfully at:", out.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/emotion_dataset.csv")
    args = parser.parse_args()
    main(args.data_path)