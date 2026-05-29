"""
BERT-based labeler.
Uses a pre-trained sentiment model (cardiffnlp/twitter-roberta-base-sentiment-latest)
to label all reviews. Much more accurate than VADER on diverse text.

Reads cleaned_reviews.xlsx, saves to labeled_reviews_bert.xlsx.
"""

import os
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 32  # process this many reviews at once for speed
MAX_LENGTH = 256  # truncate longer reviews

# Model output labels (in order): 0=negative, 1=neutral, 2=positive
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def setup_model():
    """Load tokenizer and model. Downloads ~500MB on first run."""
    print("Loading BERT model (this may take a moment on first run)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()  # set to inference mode

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on: {device}")

    return tokenizer, model, device


def classify_batch(texts, tokenizer, model, device):
    """Run BERT inference on a batch of texts. Returns list of labels."""
    # Tokenize the batch
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference (no gradient needed)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).cpu().tolist()

    return [LABEL_MAP[p] for p in predictions]


def label_all():
    """Main labeling pipeline using BERT."""
    print("Loading cleaned reviews...")
    df = pd.read_excel(config.CLEANED_FILE)
    total = len(df)
    print(f"Loaded {total} reviews\n")

    tokenizer, model, device = setup_model()

    print(f"\nLabeling {total} reviews in batches of {BATCH_SIZE}...")
    all_labels = []
    sentences = df["Sentence"].astype(str).tolist()

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = sentences[start:end]
        batch_labels = classify_batch(batch, tokenizer, model, device)
        all_labels.extend(batch_labels)

        # Progress every 10 batches
        if (start // BATCH_SIZE) % 10 == 0:
            print(f"   Processed {end}/{total} ({end/total*100:.1f}%)")

    df["Final_Label"] = all_labels

    # Reorder columns
    df = df[["Sentence", "Cleaned_Sentence", "Source_File", "Brand", "Final_Label"]]

    # Save
    output_path = os.path.join(config.PROCESSED_DIR, "labeled_reviews_bert.xlsx")
    print(f"\nSaving to {output_path}...")
    df.to_excel(output_path, index=False, engine="openpyxl")

    # Stats
    print("\n=== Label Distribution ===")
    print(df["Final_Label"].value_counts().to_string())
    print(f"\nTotal labeled: {len(df)}")
    print("\nDone.")


if __name__ == "__main__":
    label_all()