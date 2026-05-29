"""
Unified training script for the Mobile Sentiment Analysis project.

Trains ALL FIVE models on BOTH labeled datasets in a single run:

    Models:
      1. Multinomial Naive Bayes   (supervised, TF-IDF)
      2. Logistic Regression       (supervised, TF-IDF)
      3. Linear SVM                (supervised, TF-IDF)
      4. K-Means                   (unsupervised, TF-IDF + majority-vote mapping)
      5. Bi-LSTM RNN               (deep learning, PyTorch)

    Datasets:
      - VADER + TextBlob labels  -> models/saved_vader/
      - BERT (RoBERTa) labels    -> models/saved_bert/

Run it once:

    python models/train.py

It will train 5 models x 2 datasets = 10 models total and save everything
to the correct folders. No manual editing or flipping required.

All hyperparameters below are the tuned values used in the final report.
The same TF-IDF settings are shared by the classical models and K-Means so
that the VADER-vs-BERT comparison reflects only the labels, not the features.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ============================================================
#   HYPERPARAMETERS  (tuned values used for the final results)
# ============================================================

# Shared TF-IDF settings (classical models + K-Means use the same features)
TFIDF_PARAMS = dict(
    max_features=15000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

# Classical model hyperparameters (from grid-search tuning)
NB_ALPHA = 0.5
LOGREG_PARAMS = dict(C=2.0, class_weight="balanced", max_iter=2000, solver="lbfgs",
                     random_state=config.RANDOM_STATE)
SVM_PARAMS = dict(C=0.5, class_weight="balanced", max_iter=3000,
                  random_state=config.RANDOM_STATE)

# K-Means
N_CLUSTERS = 3

# Bi-LSTM RNN
MAX_VOCAB = 10000
MAX_LEN = 100
EMBED_DIM = 100
HIDDEN_DIM = 128
NUM_CLASSES = 3
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001


# The two datasets we train on. Add/remove here if needed.
LABEL_SOURCES = {
    "VADER + TextBlob": (config.LABELED_FILE,      config.MODELS_DIR_VADER),
    "BERT (RoBERTa)":   (config.LABELED_FILE_BERT, config.MODELS_DIR_BERT),
}


# ============================================================
#   RNN COMPONENTS  (tokenizer, dataset, model)
# ============================================================

class SimpleTokenizer:
    """Tiny tokenizer that converts text to integer sequences."""
    def __init__(self, max_vocab=MAX_VOCAB):
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

    def encode(self, text, max_len=MAX_LEN):
        tokens = str(text).split()[:max_len]
        ids = [self.word_to_idx.get(t, 1) for t in tokens]
        ids += [0] * (max_len - len(ids))   # pad
        return ids

    def vocab_size(self):
        return len(self.word_to_idx)


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.texts[idx])
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))


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
        h = torch.cat([hidden[0], hidden[1]], dim=1)   # both directions
        h = self.dropout(h)
        return self.fc(h)


# ============================================================
#   HELPER: load a labeled dataset
# ============================================================

def load_dataset(train_file):
    df = pd.read_excel(train_file)
    df = df.dropna(subset=["Cleaned_Sentence", "Final_Label"]).reset_index(drop=True)
    return df


# ============================================================
#   PART 1: classical models (NB, LogReg, SVM)
# ============================================================

def train_classical(df, output_dir):
    print("\n--- Classical models (Naive Bayes, Logistic Regression, SVM) ---")

    X = df["Cleaned_Sentence"].astype(str)
    y = df["Final_Label"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y
    )

    print("Building TF-IDF features...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test:  {X_test.shape[0]} samples")

    models = {}

    print("  Training Multinomial Naive Bayes...")
    nb = MultinomialNB(alpha=NB_ALPHA)
    nb.fit(X_train, y_train)
    models["naive_bayes"] = nb

    print("  Training Logistic Regression...")
    lr = LogisticRegression(**LOGREG_PARAMS)
    lr.fit(X_train, y_train)
    models["logistic_regression"] = lr

    print("  Training Linear SVM...")
    svm = LinearSVC(**SVM_PARAMS)
    svm.fit(X_train, y_train)
    models["svm"] = svm

    # Save
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
    for name, mdl in models.items():
        joblib.dump(mdl, os.path.join(output_dir, f"{name}.pkl"))
    joblib.dump({"X_test": X_test, "y_test": y_test},
                os.path.join(output_dir, "test_data.pkl"))

    # Quick scores for the summary table
    scores = {}
    for name, mdl in models.items():
        pred = mdl.predict(X_test)
        scores[name] = (accuracy_score(y_test, pred),
                        f1_score(y_test, pred, average="macro", zero_division=0))
        print(f"    {name:<22} acc={scores[name][0]:.4f}  f1={scores[name][1]:.4f}")

    return scores


# ============================================================
#   PART 2: K-Means (unsupervised)
# ============================================================

def map_clusters_to_labels(cluster_ids, true_labels):
    mapping = {}
    for cluster_id in range(N_CLUSTERS):
        labels_in_cluster = true_labels[cluster_ids == cluster_id]
        if len(labels_in_cluster) == 0:
            mapping[cluster_id] = "neutral"
        else:
            mapping[cluster_id] = Counter(labels_in_cluster).most_common(1)[0][0]
    return mapping


def train_kmeans(df, output_dir):
    print("\n--- K-Means (unsupervised) ---")

    X = df["Cleaned_Sentence"].astype(str)
    y = df["Final_Label"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y
    )

    # Same TF-IDF settings as classical models, but fit a separate vectorizer
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print(f"Training K-Means (k={N_CLUSTERS})...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=config.RANDOM_STATE,
                    n_init=10, max_iter=300)
    kmeans.fit(X_train)

    train_cluster_ids = kmeans.predict(X_train)
    cluster_to_label = map_clusters_to_labels(
        train_cluster_ids, y_train.reset_index(drop=True).values
    )
    print(f"Cluster -> Label mapping: {cluster_to_label}")

    test_cluster_ids = kmeans.predict(X_test)
    y_pred = np.array([cluster_to_label[c] for c in test_cluster_ids])

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"    kmeans                 acc={acc:.4f}  f1={f1:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(kmeans, os.path.join(output_dir, "kmeans.pkl"))
    joblib.dump(cluster_to_label, os.path.join(output_dir, "kmeans_label_map.pkl"))
    joblib.dump(vectorizer, os.path.join(output_dir, "kmeans_vectorizer.pkl"))

    return {"kmeans": (acc, f1)}


# ============================================================
#   PART 3: Bi-LSTM RNN (PyTorch)
# ============================================================

def train_rnn(df, output_dir):
    print("\n--- Bi-LSTM RNN (PyTorch) ---")

    X = df["Cleaned_Sentence"].astype(str).tolist()
    y_str = df["Final_Label"].tolist()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y
    )

    print("Building tokenizer...")
    tokenizer = SimpleTokenizer(max_vocab=MAX_VOCAB)
    tokenizer.fit(X_train)
    print(f"Vocabulary size: {tokenizer.vocab_size()}")

    train_loader = DataLoader(ReviewDataset(X_train, y_train, tokenizer),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ReviewDataset(X_test, y_test, tokenizer),
                             batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = BiLSTMClassifier(
        vocab_size=tokenizer.vocab_size(),
        embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        start = time.time()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} "
              f"| Time: {time.time()-start:.1f}s")

    # Evaluate
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = torch.argmax(model(x_batch), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    y_pred_str = label_encoder.inverse_transform(all_preds)
    y_test_str = label_encoder.inverse_transform(all_true)
    acc = accuracy_score(y_test_str, y_pred_str)
    f1 = f1_score(y_test_str, y_pred_str, average="macro", zero_division=0)
    print(f"    rnn                    acc={acc:.4f}  f1={f1:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "rnn.pt"))
    joblib.dump(tokenizer, os.path.join(output_dir, "rnn_tokenizer.pkl"))
    joblib.dump(label_encoder, os.path.join(output_dir, "rnn_label_encoder.pkl"))
    joblib.dump({
        "vocab_size": tokenizer.vocab_size(),
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_classes": NUM_CLASSES,
        "max_len": MAX_LEN,
    }, os.path.join(output_dir, "rnn_config.pkl"))

    return {"rnn": (acc, f1)}


# ============================================================
#   DRIVER
# ============================================================

def train_all_for_dataset(label_source, train_file, output_dir):
    print(f"\n{'='*70}")
    print(f"  TRAINING ALL MODELS ON: {label_source}")
    print(f"  Source file : {train_file}")
    print(f"  Output dir  : {output_dir}")
    print(f"{'='*70}")

    df = load_dataset(train_file)
    print(f"Total samples: {len(df)}")
    print("Label distribution:")
    print(df["Final_Label"].value_counts().to_string())

    scores = {}
    scores.update(train_classical(df, output_dir))
    scores.update(train_kmeans(df, output_dir))
    scores.update(train_rnn(df, output_dir))
    return scores


def main():
    all_scores = {}
    for label_source, (train_file, output_dir) in LABEL_SOURCES.items():
        all_scores[label_source] = train_all_for_dataset(
            label_source, train_file, output_dir
        )

    # Final summary table
    print("\n\n" + "=" * 70)
    print("  FINAL RESULTS — ALL MODELS, BOTH DATASETS")
    print("=" * 70)
    model_order = ["naive_bayes", "logistic_regression", "svm", "kmeans", "rnn"]
    for label_source, scores in all_scores.items():
        print(f"\n  {label_source}")
        print(f"  {'Model':<24}{'Accuracy':>12}{'F1 (macro)':>14}")
        print("  " + "-" * 50)
        for m in model_order:
            if m in scores:
                acc, f1 = scores[m]
                print(f"  {m:<24}{acc:>12.4f}{f1:>14.4f}")

    print("\nDone. All 10 models saved to their respective folders.")


if __name__ == "__main__":
    main()