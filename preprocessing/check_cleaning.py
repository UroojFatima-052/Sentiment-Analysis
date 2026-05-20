"""
Quick spot-check of cleaning quality.
Shows random samples of original vs cleaned text side-by-side.
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def check():
    df = pd.read_excel(config.CLEANED_FILE)
    print(f"Total cleaned rows: {len(df)}\n")

    samples = df.sample(15, random_state=42)

    for i, row in enumerate(samples.itertuples(), 1):
        original = str(row.Sentence)[:150]
        cleaned = str(row.Cleaned_Sentence)[:150]
        print(f"--- Sample {i} ---")
        print(f"ORIGINAL: {original}")
        print(f"CLEANED:  {cleaned}")
        print()


if __name__ == "__main__":
    check()