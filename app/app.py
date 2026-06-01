"""
Flask backend for the Mobile Sentiment Analysis Dashboard.

Routes:
  GET  /                    -> dashboard page
  GET  /predict             -> live prediction page
  POST /api/run_model       -> run a single model on dataset
  POST /api/compare_all     -> run all 5 models for comparison
  POST /api/predict         -> live prediction on user-typed text
  POST /api/upload          -> accept user-uploaded dataset
"""

import os
import sys
import re
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import Counter
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Add project root so we can import config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
import config


# ===== RNN classes (must match train_rnn.py) =====
class SimpleTokenizer:
    def __init__(self, max_vocab=10000):
        self.max_vocab = max_vocab
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}

    def fit(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(str(text).split())
        most_common = counter.most_common(self.max_vocab - 2)
        for word, _ in most_common:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

    def encode(self, text, max_len=100):
        tokens = str(text).split()[:max_len]
        ids = [self.word_to_idx.get(t, 1) for t in tokens]
        ids += [0] * (max_len - len(ids))
        return ids

    def vocab_size(self):
        return len(self.word_to_idx)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (hidden, _) = self.lstm(emb)
        h = torch.cat([hidden[0], hidden[1]], dim=1)
        h = self.dropout(h)
        return self.fc(h)


# ===== Text cleaning (mirrors cleaner.py) =====
NEGATION_WORDS = {"not", "no", "never", "none", "neither", "nor",
                  "hardly", "barely", "scarcely", "without"}
CUSTOM_STOPWORDS = set(stopwords.words("english")) - NEGATION_WORDS
lemmatizer = WordNetLemmatizer()

CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't": "will not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "haven't": "have not",
    "hasn't": "has not", "hadn't": "had not", "can't": "can not",
    "cannot": "can not", "couldn't": "could not", "wouldn't": "would not",
    "shouldn't": "should not", "i'm": "i am", "you're": "you are",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "they're": "they are",
    "dont": "do not", "doesnt": "does not", "didnt": "did not",
    "wont": "will not", "isnt": "is not", "arent": "are not",
    "wasnt": "was not", "werent": "were not", "havent": "have not",
    "hasnt": "has not", "hadnt": "had not", "cant": "can not",
    "couldnt": "could not", "wouldnt": "would not", "shouldnt": "should not",
}


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    for short, full in CONTRACTIONS.items():
        text = text.replace(short, full)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^\w\s!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    cleaned = []
    for tok in tokens:
        if tok in {"!", "?"}:
            cleaned.append(tok)
            continue
        if tok in CUSTOM_STOPWORDS or len(tok) < 2:
            continue
        lemma = lemmatizer.lemmatize(tok, pos=wordnet.VERB)
        lemma = lemmatizer.lemmatize(lemma)
        cleaned.append(lemma)
    return " ".join(cleaned)


# ===== Labeling for UPLOADED data (mirrors the project's labeling pipeline) =====
# VADER: lightweight, loaded once. BERT: heavy, lazy-loaded only when first used.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
_vader = SentimentIntensityAnalyzer()

_bert_labeler = {"tokenizer": None, "model": None}  # filled on first BERT use

POS_THR = config.POSITIVE_THRESHOLD
NEG_THR = config.NEGATIVE_THRESHOLD


def _score_to_label(score):
    if score > POS_THR:
        return "positive"
    if score < NEG_THR:
        return "negative"
    return "neutral"


def vader_label_sentences(raw_sentences):
    """Label uploaded sentences with VADER (uses RAW text, like the pipeline)."""
    return [_score_to_label(_vader.polarity_scores(str(t))["compound"])
            for t in raw_sentences]


def _ensure_bert_loaded():
    if _bert_labeler["model"] is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        print("  Loading RoBERTa labeler for uploads (first use, ~500MB)...")
        _bert_labeler["tokenizer"] = AutoTokenizer.from_pretrained(name)
        m = AutoModelForSequenceClassification.from_pretrained(name)
        m.eval()
        _bert_labeler["model"] = m


def bert_label_sentences(raw_sentences):
    """Label uploaded sentences with RoBERTa (uses RAW text, like the pipeline)."""
    _ensure_bert_loaded()
    tok = _bert_labeler["tokenizer"]
    model = _bert_labeler["model"]
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    out = []
    texts = [str(t) for t in raw_sentences]
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=256)
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            out.extend(id2label[p] for p in preds)
    return out


def label_uploaded(raw_sentences, label_source):
    if label_source == "bert":
        return bert_label_sentences(raw_sentences)
    return vader_label_sentences(raw_sentences)


# In-memory cache of uploaded+labeled data, keyed by a token we hand the client.
# Avoids re-labeling (esp. slow BERT) every time the user picks a model.
UPLOAD_CACHE = {}


# ===== Load all models once at startup =====
print("Loading models...")
LOADED_MODELS = {}
for label_source, models_dir in [("vader", config.MODELS_DIR_VADER),
                                  ("bert", config.MODELS_DIR_BERT)]:
    print(f"  {label_source.upper()} from {models_dir}")
    bundle = {
        "tfidf": joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl")),
        "naive_bayes": joblib.load(os.path.join(models_dir, "naive_bayes.pkl")),
        "logistic_regression": joblib.load(os.path.join(models_dir, "logistic_regression.pkl")),
        "svm": joblib.load(os.path.join(models_dir, "svm.pkl")),
        "kmeans": joblib.load(os.path.join(models_dir, "kmeans.pkl")),
        "kmeans_vectorizer": joblib.load(os.path.join(models_dir, "kmeans_vectorizer.pkl")),
        "kmeans_label_map": joblib.load(os.path.join(models_dir, "kmeans_label_map.pkl")),
        "rnn_tokenizer": joblib.load(os.path.join(models_dir, "rnn_tokenizer.pkl")),
        "rnn_label_encoder": joblib.load(os.path.join(models_dir, "rnn_label_encoder.pkl")),
    }
    rnn_cfg = joblib.load(os.path.join(models_dir, "rnn_config.pkl"))
    bundle["rnn_config"] = rnn_cfg
    rnn = BiLSTMClassifier(
        vocab_size=rnn_cfg["vocab_size"],
        embed_dim=rnn_cfg["embed_dim"],
        hidden_dim=rnn_cfg["hidden_dim"],
        num_classes=rnn_cfg["num_classes"]
    )
    rnn.load_state_dict(torch.load(os.path.join(models_dir, "rnn.pt"),
                                   map_location="cpu"))
    rnn.eval()
    bundle["rnn"] = rnn
    LOADED_MODELS[label_source] = bundle

print("Models loaded.")

# Load default datasets (full data — used for stats and for live prediction context)
print("Loading default datasets...")
DEFAULT_DATA = {
    "vader": pd.read_excel(config.LABELED_FILE),
    "bert": pd.read_excel(config.LABELED_FILE_BERT),
}
for k in DEFAULT_DATA:
    DEFAULT_DATA[k] = DEFAULT_DATA[k].dropna(
        subset=["Cleaned_Sentence", "Final_Label"]
    ).reset_index(drop=True)
print("Datasets loaded.")

# Build TEST sets (the 20% held out during training).
# We reproduce the EXACT same split used in train.py by using the same random_state.
TEST_DATA = {}
for label_source, df in DEFAULT_DATA.items():
    X = df["Cleaned_Sentence"].astype(str)
    y = df["Final_Label"]
    _, X_test_text, _, _ = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    # The split returns the same index as the original df; use it to recover Brand column.
    test_indices = X_test_text.index
    TEST_DATA[label_source] = df.loc[test_indices].reset_index(drop=True)
print(f"Test sets ready: VADER={len(TEST_DATA['vader'])}, BERT={len(TEST_DATA['bert'])}\n")


def get_test_set(label_source):
    """Return the held-out test set (not used during training)."""
    return TEST_DATA[label_source].copy()


# ===== Model name display mapping =====
MODEL_DISPLAY = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "svm": "Linear SVM",
    "kmeans": "K-Means (Unsupervised)",
    "rnn": "RNN (Bi-LSTM)",
}
ALL_MODELS = ["naive_bayes", "logistic_regression", "svm", "kmeans", "rnn"]


# ===== Prediction functions =====
def predict_classical(model_name, label_source, cleaned_texts):
    b = LOADED_MODELS[label_source]
    X = b["tfidf"].transform(cleaned_texts)
    return b[model_name].predict(X)


def predict_kmeans(label_source, cleaned_texts):
    b = LOADED_MODELS[label_source]
    X = b["kmeans_vectorizer"].transform(cleaned_texts)
    cluster_ids = b["kmeans"].predict(X)
    return np.array([b["kmeans_label_map"][c] for c in cluster_ids])


def predict_rnn(label_source, cleaned_texts):
    b = LOADED_MODELS[label_source]
    tok = b["rnn_tokenizer"]
    cfg = b["rnn_config"]
    enc = b["rnn_label_encoder"]
    ids = [tok.encode(t, max_len=cfg["max_len"]) for t in cleaned_texts]
    X = torch.tensor(ids, dtype=torch.long)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            batch = X[i:i+64]
            logits = b["rnn"](batch)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    return enc.inverse_transform(preds)


def get_predictions(model_name, label_source, cleaned_texts):
    if model_name in ["naive_bayes", "logistic_regression", "svm"]:
        return predict_classical(model_name, label_source, cleaned_texts)
    if model_name == "kmeans":
        return predict_kmeans(label_source, cleaned_texts)
    if model_name == "rnn":
        return predict_rnn(label_source, cleaned_texts)
    raise ValueError(f"Unknown model: {model_name}")


def compute_metrics(y_true, y_pred):
    labels = ["negative", "neutral", "positive"]
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def filter_df_by_brand(df, brand):
    if brand and brand.lower() != "all":
        return df[df["Brand"].str.lower() == brand.lower()].reset_index(drop=True)
    return df


# ===== Flask app =====
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = config.UPLOAD_DIR
os.makedirs(config.UPLOAD_DIR, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/api/run_model", methods=["POST"])
def api_run_model():
    data = request.get_json()
    model_name = data.get("model", "logistic_regression")
    label_source = data.get("label_source", "bert")
    brand = data.get("brand", "all")

    # Use TEST SET only (the 20% held out during training).
    # Evaluating on training data would give misleadingly high accuracy.
    test_df = get_test_set(label_source)
    test_df = filter_df_by_brand(test_df, brand)
    if len(test_df) == 0:
        return jsonify({"error": f"No test reviews for brand {brand}"}), 400

    cleaned = test_df["Cleaned_Sentence"].astype(str).tolist()
    y_true = test_df["Final_Label"].values
    y_pred = get_predictions(model_name, label_source, cleaned)
    metrics = compute_metrics(y_true, y_pred)

    return jsonify({
        "model": MODEL_DISPLAY[model_name],
        "model_key": model_name,
        "label_source": label_source,
        "brand": brand,
        "num_reviews": len(test_df),
        **metrics,
    })


@app.route("/api/compare_all", methods=["POST"])
def api_compare_all():
    data = request.get_json()
    label_source = data.get("label_source", "bert")
    brand = data.get("brand", "all")

    test_df = get_test_set(label_source)
    test_df = filter_df_by_brand(test_df, brand)
    if len(test_df) == 0:
        return jsonify({"error": f"No test reviews for brand {brand}"}), 400

    cleaned = test_df["Cleaned_Sentence"].astype(str).tolist()
    y_true = test_df["Final_Label"].values

    labels = ["negative", "neutral", "positive"]
    results = []
    overall_cm = np.zeros((len(labels), len(labels)), dtype=int)
    for model_name in ALL_MODELS:
        y_pred = get_predictions(model_name, label_source, cleaned)
        m = compute_metrics(y_true, y_pred)
        results.append({
            "model": MODEL_DISPLAY[model_name],
            "model_key": model_name,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        })
        # accumulate this model's confusion matrix into the overall total
        overall_cm += np.array(m["confusion_matrix"])

    # Average metrics across all 5 models
    n = len(results)
    average = {
        "accuracy": round(sum(r["accuracy"] for r in results) / n, 4),
        "precision": round(sum(r["precision"] for r in results) / n, 4),
        "recall": round(sum(r["recall"] for r in results) / n, 4),
        "f1": round(sum(r["f1"] for r in results) / n, 4),
    }

    return jsonify({
        "label_source": label_source,
        "brand": brand,
        "num_reviews": len(test_df),
        "results": results,
        "average": average,
        "overall_cm": overall_cm.tolist(),
        "labels": labels,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    label_source = data.get("label_source", "bert")

    if not text:
        return jsonify({"error": "Please type a review"}), 400

    cleaned = clean_text(text)
    if not cleaned or len(cleaned) < 3:
        return jsonify({"error": "Text too short after cleaning"}), 400

    predictions = {}
    for model_name in ALL_MODELS:
        pred = get_predictions(model_name, label_source, [cleaned])[0]
        predictions[model_name] = {
            "name": MODEL_DISPLAY[model_name],
            "label": str(pred),
        }

    return jsonify({
        "original": text,
        "cleaned": cleaned,
        "label_source": label_source,
        "predictions": predictions,
    })


@app.route("/api/upload_analyze", methods=["POST"])
def api_upload_analyze():
    """
    Step 1 of the upload pipeline.
    Takes a file of sentences (no labels needed) + label_source (vader/bert).
    Cleans -> labels with VADER/BERT (like the real pipeline) -> returns the
    label distribution (for the first pie + numbers) and a token for step 2.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = f.filename.rsplit(".", 1)[-1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        return jsonify({"error": f"File type .{ext} not allowed"}), 400

    label_source = request.form.get("label_source", "vader")

    safe_name = secure_filename(f.filename)
    save_path = os.path.join(config.UPLOAD_DIR, safe_name)
    f.save(save_path)

    try:
        df = pd.read_csv(save_path) if ext == "csv" else pd.read_excel(save_path)
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    if "Sentence" not in df.columns:
        return jsonify({"error": "File must have a 'Sentence' column"}), 400

    df = df.dropna(subset=["Sentence"]).reset_index(drop=True)
    if len(df) == 0:
        return jsonify({"error": "No sentences found in file"}), 400

    raw = df["Sentence"].astype(str).tolist()
    cleaned = [clean_text(t) for t in raw]

    # Generate labels with the chosen labeler (this is the pipeline's labeling step)
    labels = label_uploaded(raw, label_source)

    # Cache cleaned text + generated labels for the model step
    token = secure_filename(safe_name) + "__" + label_source
    UPLOAD_CACHE[token] = {
        "cleaned": cleaned,
        "labels": labels,
        "label_source": label_source,
    }

    dist = {k: int(v) for k, v in pd.Series(labels).value_counts().items()}
    for c in ["positive", "neutral", "negative"]:
        dist.setdefault(c, 0)

    return jsonify({
        "token": token,
        "filename": safe_name,
        "label_source": label_source,
        "num_rows": len(df),
        "label_distribution": dist,
    })


@app.route("/api/upload_run_model", methods=["POST"])
def api_upload_run_model():
    """
    Step 2 of the upload pipeline.
    Takes the token + chosen model. Runs the trained model on the uploaded
    sentences and compares its predictions against the VADER/BERT labels
    generated in step 1 -> confusion matrix + metrics + predicted distribution.
    """
    data = request.get_json()
    token = data.get("token")
    model_name = data.get("model", "logistic_regression")

    cached = UPLOAD_CACHE.get(token)
    if not cached:
        return jsonify({"error": "Upload expired. Please re-upload the file."}), 400

    label_source = cached["label_source"]
    cleaned = cached["cleaned"]
    y_true = np.array(cached["labels"])  # the VADER/BERT labels

    y_pred = get_predictions(model_name, label_source, cleaned)
    metrics = compute_metrics(y_true, y_pred)

    pred_dist = {k: int(v) for k, v in pd.Series(y_pred).value_counts().items()}
    for c in ["positive", "neutral", "negative"]:
        pred_dist.setdefault(c, 0)

    return jsonify({
        "model": MODEL_DISPLAY[model_name],
        "model_key": model_name,
        "label_source": label_source,
        "num_rows": len(cleaned),
        "predicted_distribution": pred_dist,
        **metrics,
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = f.filename.rsplit(".", 1)[-1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        return jsonify({"error": f"File type .{ext} not allowed"}), 400

    safe_name = secure_filename(f.filename)
    save_path = os.path.join(config.UPLOAD_DIR, safe_name)
    f.save(save_path)

    try:
        if ext == "csv":
            df = pd.read_csv(save_path)
        else:
            df = pd.read_excel(save_path)
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    if "Sentence" not in df.columns:
        return jsonify({"error": "File must have a 'Sentence' column"}), 400

    return jsonify({
        "message": "File uploaded successfully",
        "filename": safe_name,
        "rows": len(df),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)