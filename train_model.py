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

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())

def main(data_path):
    df = pd.read_csv(data_path, header=None, names=["text","emotion"]) if df_is_no_header(data_path) else pd.read_csv(data_path)
    # attempt to support both header/no-header
    if "text" not in df.columns or "emotion" not in df.columns:
        # try naive split if CSV is two columns
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ["text", "emotion"]
        else:
            raise SystemExit("Data must have two columns: text,emotion")

    df = df.dropna(subset=["text","emotion"])
    X = df["text"].apply(clean_text).values
    y = df["emotion"].values

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1,2))),
        ("clf", MultinomialNB())
    ])
    pipeline.fit(X, y)

    p = pathlib.Path("models"); p.mkdir(parents=True, exist_ok=True)
    out = p / "model.pkl"
    joblib.dump(pipeline, out)
    print("WROTE:", out.resolve())

# helper to detect headerless CSV (very naive)
def df_is_no_header(path):
    import csv
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        first = next(r, None)
        # if first row contains more than one word in first cell then probably headerless? (best-effort)
        if first is None:
            return False
        # Heuristic: if exactly 2 cells and second looks like emotion (short), assume no header
        return len(first) == 2 and len(first[1]) <= 12 and "text" not in first[0].lower()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/emotion_dataset.csv")
    args = parser.parse_args()
    main(args.data_path)
