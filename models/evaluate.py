"""
Evaluates ALL 5 trained models on the held-out test set, for BOTH labeled datasets.

Models:
  - Multinomial Naive Bayes (supervised, classical)
  - Logistic Regression (supervised, classical)
  - Linear SVM (supervised, classical)
  - K-Means Clustering (unsupervised)
  - Bi-LSTM RNN (supervised, deep learning)

Reports Precision, Recall, F1, Accuracy, Confusion Matrix per model.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Standard label order for confusion matrix
LABEL_ORDER = ["negative", "neutral", "positive"]


# ===== Tokenizer class (must match train_rnn.py) =====
from collections import Counter

class SimpleTokenizer:
    """Tiny tokenizer that converts text to integer sequences."""
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

# ===== RNN model definition (must match train_rnn.py) =====
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


def print_metrics_block(model_name, y_test, y_pred):
    """Pretty print metrics + confusion matrix for one model."""
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n  --- {model_name} ---")
    print(f"  Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}  (macro-avg)")
    print(f"  Recall:    {rec:.4f}  (macro-avg)")
    print(f"  F1-score:  {f1:.4f}  (macro-avg)")

    print(f"\n  Per-class breakdown:")
    print(classification_report(y_test, y_pred,
                                labels=LABEL_ORDER,
                                target_names=LABEL_ORDER,
                                zero_division=0, digits=3))

    cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
    print(f"  Confusion Matrix (rows = actual, cols = predicted):")
    print(f"                 {LABEL_ORDER[0]:<10}{LABEL_ORDER[1]:<10}{LABEL_ORDER[2]:<10}")
    for i, label in enumerate(LABEL_ORDER):
        row = "  ".join(f"{cm[i][j]:<8}" for j in range(len(LABEL_ORDER)))
        print(f"    {label:<10} {row}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def evaluate_classical(model_path, X_test, y_test):
    """Evaluate any classical model (NB, LogReg, SVM) on TF-IDF test data."""
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    return y_pred


def evaluate_kmeans(models_dir, X_test_text, y_test):
    """K-Means uses its own vectorizer and a cluster->label map."""
    kmeans = joblib.load(os.path.join(models_dir, "kmeans.pkl"))
    label_map = joblib.load(os.path.join(models_dir, "kmeans_label_map.pkl"))
    vectorizer = joblib.load(os.path.join(models_dir, "kmeans_vectorizer.pkl"))

    X_test_vec = vectorizer.transform(X_test_text)
    cluster_ids = kmeans.predict(X_test_vec)
    y_pred = np.array([label_map[c] for c in cluster_ids])
    return y_pred


def evaluate_rnn(models_dir, X_test_text, y_test):
    """RNN uses its own tokenizer, model checkpoint, and label encoder."""
    tokenizer = joblib.load(os.path.join(models_dir, "rnn_tokenizer.pkl"))
    label_encoder = joblib.load(os.path.join(models_dir, "rnn_label_encoder.pkl"))
    cfg = joblib.load(os.path.join(models_dir, "rnn_config.pkl"))

    model = BiLSTMClassifier(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=cfg["num_classes"]
    )
    model.load_state_dict(torch.load(os.path.join(models_dir, "rnn.pt"),
                                     map_location="cpu"))
    model.eval()

    # Tokenize all test texts
    all_ids = [tokenizer.encode(t, max_len=cfg["max_len"]) for t in X_test_text]
    X = torch.tensor(all_ids, dtype=torch.long)

    # Predict in batches
    all_preds = []
    BATCH = 64
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            batch = X[i:i+BATCH]
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    y_pred = label_encoder.inverse_transform(all_preds)
    return y_pred


def evaluate_set(label_name, labeled_file, models_dir):
    """Run all 5 models on one labeled dataset's test split."""
    print(f"\n{'='*70}")
    print(f"  EVALUATING MODELS TRAINED ON: {label_name}")
    print(f"  Models directory: {models_dir}")
    print(f"{'='*70}")

    if not os.path.exists(models_dir):
        print(f"  Directory not found — skipping.")
        return {}

    # Reproduce same train/test split (same random_state) to get the test set
    df = pd.read_excel(labeled_file)
    df = df.dropna(subset=["Cleaned_Sentence", "Final_Label"]).reset_index(drop=True)
    X = df["Cleaned_Sentence"].astype(str)
    y = df["Final_Label"]

    _, X_test_text, _, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y
    )
    X_test_text = X_test_text.tolist()
    y_test = y_test.reset_index(drop=True).values

    # TF-IDF vectorization for classical models
    vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    X_test_tfidf = vectorizer.transform(X_test_text)

    print(f"\n  Test samples: {len(X_test_text)}")
    print(f"  Test class distribution:")
    for label in LABEL_ORDER:
        count = (y_test == label).sum()
        print(f"    {label:<10} {count}")

    results = {}

    # === Naive Bayes ===
    y_pred = evaluate_classical(os.path.join(models_dir, "naive_bayes.pkl"),
                                X_test_tfidf, y_test)
    results["Naive Bayes"] = print_metrics_block("Naive Bayes", y_test, y_pred)

    # === Logistic Regression ===
    y_pred = evaluate_classical(os.path.join(models_dir, "logistic_regression.pkl"),
                                X_test_tfidf, y_test)
    results["Logistic Regression"] = print_metrics_block("Logistic Regression", y_test, y_pred)

    # === Linear SVM ===
    y_pred = evaluate_classical(os.path.join(models_dir, "svm.pkl"),
                                X_test_tfidf, y_test)
    results["Linear SVM"] = print_metrics_block("Linear SVM", y_test, y_pred)

    # === K-Means (unsupervised) ===
    y_pred = evaluate_kmeans(models_dir, X_test_text, y_test)
    results["K-Means"] = print_metrics_block("K-Means (Unsupervised)", y_test, y_pred)

    # === RNN ===
    y_pred = evaluate_rnn(models_dir, X_test_text, y_test)
    results["RNN (Bi-LSTM)"] = print_metrics_block("RNN (Bi-LSTM)", y_test, y_pred)

    return results


def print_comparison(results_vader, results_bert):
    """Side-by-side comparison of both training runs."""
    print(f"\n\n{'='*78}")
    print(f"  FINAL COMPARISON — 5 Models × 2 Labeled Datasets")
    print(f"{'='*78}\n")

    print(f"  {'Model':<22}{'Metric':<14}{'VADER':<12}{'BERT':<12}{'Δ':<10}")
    print(f"  {'-'*70}")

    models = ["Naive Bayes", "Logistic Regression", "Linear SVM", "K-Means", "RNN (Bi-LSTM)"]
    metrics = [("accuracy", "Accuracy"), ("f1", "F1 (macro)"),
               ("precision", "Precision"), ("recall", "Recall")]

    for model in models:
        for key, label in metrics:
            v = results_vader.get(model, {}).get(key, 0)
            b = results_bert.get(model, {}).get(key, 0)
            diff = b - v
            arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
            print(f"  {model:<22}{label:<14}{v:.3f}       {b:.3f}       {arrow}{abs(diff):.3f}")
        print()


def main():
    results_vader = evaluate_set("VADER + TextBlob",
                                  config.LABELED_FILE,
                                  config.MODELS_DIR_VADER)
    results_bert = evaluate_set("BERT (RoBERTa)",
                                 config.LABELED_FILE_BERT,
                                 config.MODELS_DIR_BERT)

    if results_vader and results_bert:
        print_comparison(results_vader, results_bert)


if __name__ == "__main__":
    main()