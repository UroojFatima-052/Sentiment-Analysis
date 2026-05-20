"""
Labeler: assigns sentiment labels to each cleaned review using a hybrid strategy.

Uses TWO independent sentiment scorers (VADER and TextBlob).
  - If both agree -> auto-accept the label
  - If they disagree -> flag the row for manual review
  - If VADER is very uncertain (compound near 0) -> also flag for review

Adds columns: VADER_Label, TextBlob_Label, Final_Label, Needs_Review
Reads from cleaned_reviews.xlsx, saves to labeled_reviews.xlsx.
"""

import os
import sys
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Initialize VADER once (loading the lexicon is slow)
vader = SentimentIntensityAnalyzer()


def label_from_score(score):
    """
    Convert a continuous sentiment score (-1 to +1) to a label.
    Uses thresholds defined in config.
    """
    if score > config.POSITIVE_THRESHOLD:
        return "positive"
    elif score < config.NEGATIVE_THRESHOLD:
        return "negative"
    else:
        return "neutral"


def get_vader_label_and_score(text):
    """
    Returns (label, compound_score) from VADER.
    Uses the ORIGINAL sentence so VADER sees punctuation/intensifiers.
    """
    scores = vader.polarity_scores(str(text))
    compound = scores["compound"]
    return label_from_score(compound), compound


def get_textblob_label(text):
    """
    Returns label from TextBlob polarity.
    """
    polarity = TextBlob(str(text)).sentiment.polarity
    return label_from_score(polarity)


def label_all():
    """Main labeling pipeline."""
    print("Loading cleaned reviews...")
    df = pd.read_excel(config.CLEANED_FILE)
    print(f"Loaded {len(df)} reviews\n")

    # Score every review with both tools
    print("Scoring with VADER...")
    vader_results = df["Sentence"].apply(get_vader_label_and_score)
    df["VADER_Label"] = vader_results.apply(lambda x: x[0])
    df["VADER_Score"] = vader_results.apply(lambda x: x[1])

    print("Scoring with TextBlob...")
    df["TextBlob_Label"] = df["Sentence"].apply(get_textblob_label)

    # Decide final label and which rows need manual review
    print("Comparing labels and flagging disagreements...")

    final_labels = []
    needs_review = []
    for _, row in df.iterrows():
        vader_lbl = row["VADER_Label"]
        tb_lbl = row["TextBlob_Label"]
        vader_compound = row["VADER_Score"]
        vader_strength = abs(vader_compound)

        # Default: trust VADER (it's tuned for short opinion text)
        final_labels.append(vader_lbl)

        # Decide if this row needs manual review
        if vader_lbl == tb_lbl:
            # Both agree -> trust the label
            review = False
        elif {vader_lbl, tb_lbl} == {"positive", "negative"}:
            # Hard conflict (pos vs neg) -> always flag
            review = True
        elif vader_strength >= 0.4:
            # One said neutral, other didn't, BUT VADER is confident -> trust VADER
            review = False
        else:
            # Soft disagreement with weak VADER signal -> flag
            review = True

        needs_review.append(review)

    df["Final_Label"] = final_labels
    df["Needs_Review"] = needs_review

    # Reorder columns for clarity
    df = df[[
        "Sentence", "Cleaned_Sentence", "Source_File", "Brand",
        "VADER_Label", "TextBlob_Label", "Final_Label", "Needs_Review"
    ]]

    print("\nSaving labeled dataset...")
    df.to_excel(config.LABELED_FILE, index=False, engine="openpyxl")
    print(f"Saved to: {config.LABELED_FILE}\n")

    # Report stats
    print("=== Label Distribution (Final_Label) ===")
    print(df["Final_Label"].value_counts().to_string())

    print("\n=== Agreement Stats ===")
    agree_count = (df["VADER_Label"] == df["TextBlob_Label"]).sum()
    total = len(df)
    print(f"VADER and TextBlob agreed: {agree_count} / {total} ({agree_count/total*100:.1f}%)")

    print("\n=== Manual Review Workload ===")
    flagged = df["Needs_Review"].sum()
    print(f"Rows flagged for manual review: {flagged} ({flagged/total*100:.1f}%)")
    print(f"Auto-accepted (no review needed): {total - flagged}")

    print("\nDone. Open the Excel and filter by Needs_Review = TRUE to start manual review.")


if __name__ == "__main__":
    label_all()