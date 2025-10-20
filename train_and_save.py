import pandas as pd, re, pathlib, joblib, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Read data
try:
    df = pd.read_csv('data/emotion_dataset.csv')
except Exception as e:
    print('ERROR: Could not read data/emotion_dataset.csv:', e, file=sys.stderr)
    raise SystemExit(1)

df = df.dropna(subset=['text','emotion'])
if df.shape[0] == 0:
    print('ERROR: No rows after dropping NA in data/emotion_dataset.csv', file=sys.stderr)
    raise SystemExit(1)

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"http\\S+"," ", s)
    s = re.sub(r"[^a-z0-9\\s]"," ", s)
    return " ".join(s.split())

X = df['text'].apply(clean_text).values
y = df['emotion'].values

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1,2))),
    ('clf', MultinomialNB())
])
pipe.fit(X, y)

p = pathlib.Path('models'); p.mkdir(parents=True, exist_ok=True)
out = p / 'model.pkl'
joblib.dump(pipe, out)
print('WROTE:', out.resolve())
