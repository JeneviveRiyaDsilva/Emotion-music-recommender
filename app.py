# app.py  (full file - replace your existing app.py with this)
import streamlit as st
import joblib
import json
import random
import pathlib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from difflib import get_close_matches

# ---------------- NLTK setup ----------------
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
lemmatizer = WordNetLemmatizer()

# ---------------- Keywords / fallback detector ----------------
EMOTION_KEYWORDS = {
    "happy": ["happy", "joy", "joyful", "elated", "glad", "cheerful", "amazing", "great", "delighted", "excited"],
    "sad": ["sad", "lonely", "depressed", "unhappy", "down", "miserable", "melancholy", "grief", "blue"],
    "angry": ["angry", "mad", "furious", "irritat", "annoy", "hate", "rage"],
    "fear": ["scared", "fear", "afraid", "nervous", "worried", "anxious", "panic", "terrified"],
    "surprise": ["surprise", "surprised", "wow", "unexpected", "shocked", "astonish"],
    "neutral": ["ok", "fine", "normal", "neutral", "so-so", "alright"]
}
ALL_KEYWORDS = list({k for v in EMOTION_KEYWORDS.values() for k in v})

def normalize_text(s: str):
    s = str(s).lower()
    tokens = [lemmatizer.lemmatize(t) for t in word_tokenize(s)]
    return [t for t in tokens if any(c.isalnum() for c in t)]

def fallback_predict(text: str) -> str:
    tokens = normalize_text(text)
    votes = {}
    def add_vote(e, w=1):
        votes[e] = votes.get(e, 0) + w
    # exact and prefix matches
    for t in tokens:
        for emo, keys in EMOTION_KEYWORDS.items():
            if any(k == t or t.startswith(k) or k.startswith(t) for k in keys):
                add_vote(emo, 2)
    # fuzzy matches
    for t in tokens:
        matches = get_close_matches(t, ALL_KEYWORDS, n=2, cutoff=0.82)
        for m in matches:
            for emo, keys in EMOTION_KEYWORDS.items():
                if m in keys:
                    add_vote(emo, 1)
    # raw substring fallback
    text_lower = text.lower()
    for emo, keys in EMOTION_KEYWORDS.items():
        for k in keys:
            if k in text_lower:
                add_vote(emo, 1)
    return max(votes, key=votes.get) if votes else "neutral"

# ---------------- Emoji sets ----------------
EMOJI_SETS = {
    "happy": "😀😄😃😁😂😊🙂☺😇",
    "sad": "😔😪😕😟🙁☹🥺😢😣😫😥😭",
    "angry": "😒😤😡😠🤬👿💢",
    "fear": "😨😱😰😧😦🫢🤯🫣",
    "surprise": "😲😮🫨😧😦😯😬🤭",
    "neutral": "😉🙂🤗🫡😐😶🤐😌🫠"
}

# ---------------- Floating emoji CSS injector (fixed behind app) ----------------
def inject_floating_emojis(emoji_string: str):
    """
    Inject a fixed background overlay with floating emojis.
    The overlay is placed behind the Streamlit UI using z-index:-1 to
    avoid covering or hiding any content.
    """
    emoji_chars = [c for c in emoji_string]
    emoji_html = "".join(f'<span class="e">{c}</span>' for c in emoji_chars)

    # create per-emoji CSS rules (positions/delays/durations/sizes vary)
    span_rules = []
    for i in range(len(emoji_chars)):
        left = (5 + i * 12) % 92
        delay = round((i * 1.15) % 6, 2)
        duration = 10 + (i % 6) * 1.7
        size = 24 + (i % 5) * 6
        span_rules.append(f".emoji-overlay .e:nth-child({i+1}){{ left:{left}%; animation-delay:{delay}s; animation-duration:{duration}s; font-size:{size}px; }}")
    span_rules_css = "\n".join(span_rules)

    # CSS: overlay uses z-index: -1 so it is behind everything; Streamlit body remains above
    css = f"""
<style>
/* Make sure the app's main container sits above the overlay */
[data-testid="stAppViewContainer"], .stApp {{
  position: relative;
  z-index: 1; /* app above overlay */
}}

/* Emoji overlay is behind everything (negative z-index) */
.emoji-overlay {{
  pointer-events: none;
  position: fixed;
  inset: 0;
  z-index: -1;         /* BEHIND the app */
  overflow: hidden;
  background: linear-gradient(135deg, #030712 0%, #071227 45%, rgba(11,18,34,0.9) 100%);
}}

/* emoji element style */
.emoji-overlay .e {{
  position: absolute;
  bottom: -22vh;
  opacity: 0;
  transform: translateY(0) scale(0.85);
  will-change: transform, opacity;
  filter: drop-shadow(0 8px 16px rgba(0,0,0,0.45));
  animation-name: floatUp;
  animation-timing-function: linear;
  animation-iteration-count: infinite;
}}

/* floating animation */
@keyframes floatUp {{
  0%   {{ transform: translateY(0) scale(0.75); opacity: 0; }}
  8%   {{ opacity: 1; }}
  92%  {{ opacity: 1; }}
  100% {{ transform: translateY(-130vh) scale(1.05); opacity: 0; }}
}}

{span_rules_css}

</style>

<div class="emoji-overlay">
{emoji_html}
</div>
"""
    st.markdown(css, unsafe_allow_html=True)

# ---------------- Streamlit app layout ----------------
st.set_page_config(page_title="Emotion Recommender", layout="centered")

# Header (always visible)
st.markdown("<h1 style='text-align:center;color:#FFD93D;'>🎵 Emotion-Based Quote & Song Recommender</h1>", unsafe_allow_html=True)

# Try to load optional trained model (if you have model.pkl)
MODEL_PATH = pathlib.Path("models/model.pkl")
model = None
if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        st.success("✅ Model loaded successfully.")
    except Exception:
        st.warning("⚠ Model found but failed to load — using fallback detector.")
else:
    st.info("ℹ Model not found — using fallback detector.")

# Controls row
left, right = st.columns([1,1])
with left:
    enable_anim = st.checkbox("Enable animated background", value=True)
with right:
    use_model = st.checkbox("Use trained model (if available).", value=False)

st.markdown("### 🧠 Type how you feel:")
user_text = st.text_area("Try: 'I feel very happy today!'", height=140)

# persistent container for results so UI doesn't shift away
result_container = st.container()

# Recommend action
if st.button("🎧 Recommend"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # pick prediction
        if use_model and model is not None:
            try:
                emotion = model.predict([user_text])[0]
            except Exception:
                emotion = fallback_predict(user_text)
        else:
            emotion = fallback_predict(user_text)

        # show results inside the persistent container
        with result_container:
            st.subheader(f"Detected Emotion: {emotion.capitalize()}")

            # load quotes/songs
            utils_path = pathlib.Path("utils/quotes_songs.json")
            if utils_path.exists():
                try:
                    with open(utils_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {}
            else:
                data = {}

            bucket = emotion if emotion in data else "neutral"
            bucket_data = data.get(bucket, data.get("neutral", {"quotes":["Be yourself."],"songs":["https://www.youtube.com/watch?v=60ItHLz5WEA"]}))
            quote = random.choice(bucket_data.get("quotes", ["Be yourself."]))
            song = random.choice(bucket_data.get("songs", ["https://www.youtube.com/watch?v=60ItHLz5WEA"]))

            st.markdown(f"💬 Quote:** {quote}")
            st.markdown(f"🎶 Song:** [Listen on YouTube]({song})")

        # keep a record in session state for continuous background
        st.session_state["last_emotion"] = emotion

# ensure session key exists
if "last_emotion" not in st.session_state:
    st.session_state["last_emotion"] = "neutral"

# inject emoji overlay behind the app (negative z-index)
if enable_anim:
    inject_floating_emojis(EMOJI_SETS.get(st.session_state["last_emotion"], EMOJI_SETS["neutral"]))

# footer small
st.markdown("<p style='text-align:center; color:#9AE6B4; font-size:13px; margin-top:30px;'>Made with ❤ by <b>Jenevive Riya DSilva</b></p>", unsafe_allow_html=True)
st.caption("✨ Emojis float continuously based on your detected emotion.")