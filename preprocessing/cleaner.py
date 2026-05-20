"""
Cleaner: takes parsed_reviews.xlsx and produces cleaned_reviews.xlsx.

Adds a 'Cleaned_Sentence' column with normalized text ready for ML models.
Drops reviews that become too short or empty after cleaning.

Cleaning steps:
  1. Strip quoted parent comments (e.g., "UserName, 12 May 2026...")
  2. Lowercase
  3. Remove URLs
  4. Remove emojis and other non-text characters
  5. Expand common contractions (don't -> do not)
  6. Reduce repeated characters (sooooo -> soo)
  7. Remove punctuation except ! and ?
  8. Tokenize using NLTK
  9. Remove stopwords BUT keep negations (not, no, never, etc.)
 10. Lemmatize using WordNet
 11. Drop rows shorter than 15 characters after cleaning
"""

import os
import re
import sys
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ===== SETUP =====

# Negation words we DO NOT want to remove (they flip sentiment)
NEGATION_WORDS = {
    "not", "no", "never", "none", "neither", "nor",
    "hardly", "barely", "scarcely", "without"
}

# Build custom stopword list (NLTK list minus negations)
CUSTOM_STOPWORDS = set(stopwords.words("english")) - NEGATION_WORDS

# Common contractions for expansion
CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
    "couldn't": "could not", "can't": "can not", "cannot": "can not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
    "it's": "it is", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will",
    "i'd": "i would", "you'd": "you would", "he'd": "he would", "she'd": "she would",
    "dont": "do not", "doesnt": "does not", "didnt": "did not",
    "wont": "will not", "wouldnt": "would not", "shouldnt": "should not",
    "couldnt": "could not", "cant": "can not",
    "isnt": "is not", "arent": "are not", "wasnt": "was not", "werent": "were not",
    "havent": "have not", "hasnt": "has not", "hadnt": "had not",
}

lemmatizer = WordNetLemmatizer()


# ===== CLEANING FUNCTIONS =====

def strip_quote_prefix(text):
    """
    Remove patterns like 'UserName, 12 May 2026' or 'Anonymous, 12 May 2026'
    that sometimes leak from quoted parent comments.
    """
    # Pattern: anything followed by ", DD Mon YYYY" at the start of the text
    pattern = r"^[A-Za-z][A-Za-z0-9_\-\.\s]{0,30},\s*\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}"
    return re.sub(pattern, "", text).strip()


def remove_urls(text):
    """Remove http(s) and www links."""
    return re.sub(r"http[s]?://\S+|www\.\S+", " ", text)


def remove_emojis(text):
    """Remove emojis and other non-ASCII characters."""
    return re.sub(r"[^\x00-\x7F]+", " ", text)


def expand_contractions(text):
    """Replace common contractions with their expanded form."""
    for short, full in CONTRACTIONS.items():
        text = text.replace(short, full)
    return text


def reduce_repeats(text):
    """Reduce repeated characters: 'sooooo' -> 'soo', 'yessss' -> 'yess'."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def remove_punctuation(text):
    """Remove punctuation EXCEPT ! and ? (they carry sentiment)."""
    # Keep word characters, whitespace, exclamation, question mark
    return re.sub(r"[^\w\s!?]", " ", text)


def tokenize_and_filter(text):
    """Tokenize, remove stopwords (keeping negations), lemmatize."""
    tokens = word_tokenize(text)
    cleaned_tokens = []
    for token in tokens:
        # skip pure punctuation tokens
        if token in {"!", "?"}:
            cleaned_tokens.append(token)
            continue
        # skip stopwords
        if token in CUSTOM_STOPWORDS:
            continue
        # skip single characters and empty tokens
        if len(token) < 2:
            continue
        # lemmatize
        lemma = lemmatizer.lemmatize(token, pos=wordnet.VERB)
        lemma = lemmatizer.lemmatize(lemma)  # default = noun
        cleaned_tokens.append(lemma)
    return " ".join(cleaned_tokens)


def clean_sentence(text):
    """Run the full cleaning pipeline on one sentence."""
    if not isinstance(text, str):
        return ""
    text = strip_quote_prefix(text)
    text = text.lower()
    text = remove_urls(text)
    text = remove_emojis(text)
    text = expand_contractions(text)
    text = reduce_repeats(text)
    text = remove_punctuation(text)
    text = re.sub(r"\s+", " ", text).strip()  # collapse multiple spaces
    text = tokenize_and_filter(text)
    return text


# ===== MAIN =====

def main():
    print("Loading parsed reviews...")
    df = pd.read_excel(config.PARSED_FILE)
    original_count = len(df)
    print(f"Original rows: {original_count}")

    print("\nCleaning sentences (this may take a minute)...")
    df["Cleaned_Sentence"] = df["Sentence"].apply(clean_sentence)

    print("Dropping rows shorter than 15 characters after cleaning...")
    df = df[df["Cleaned_Sentence"].str.len() >= 15].reset_index(drop=True)
    final_count = len(df)
    dropped = original_count - final_count
    print(f"Dropped: {dropped} rows ({dropped / original_count * 100:.1f}%)")
    print(f"Remaining: {final_count} rows")

    # Reorder columns for clarity
    df = df[["Sentence", "Cleaned_Sentence", "Source_File", "Brand"]]

    print("\nSaving cleaned dataset...")
    df.to_excel(config.CLEANED_FILE, index=False, engine="openpyxl")
    print(f"Saved to: {config.CLEANED_FILE}")

    print("\nRow distribution by brand:")
    print(df["Brand"].value_counts().to_string())


if __name__ == "__main__":
    main()