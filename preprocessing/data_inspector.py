"""
Quick data quality inspector.
Scans parsed_reviews.xlsx and reports stats so we know what cleaning issues exist.
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def inspect():
    df = pd.read_excel(config.PARSED_FILE)
    total = len(df)
    print(f"Total reviews: {total}\n")

    # Add a length column once, reuse it
    df["length"] = df["Sentence"].astype(str).str.len()

    # Length stats
    print("=== LENGTH STATS ===")
    print(f"Shortest: {df['length'].min()} chars")
    print(f"Longest:  {df['length'].max()} chars")
    print(f"Average:  {df['length'].mean():.0f} chars")
    print(f"Reviews under 20 chars: {(df['length'] < 20).sum()}")
    print(f"Reviews over 500 chars: {(df['length'] > 500).sum()}")

    # Pattern checks
    print("\n=== PATTERN CHECKS ===")
    has_url = df["Sentence"].str.contains(r"http[s]?://|www\.", regex=True, na=False).sum()
    print(f"Reviews containing URLs: {has_url}")

    has_emoji = df["Sentence"].apply(lambda x: any(ord(c) > 10000 for c in str(x))).sum()
    print(f"Reviews containing emojis: {has_emoji}")

    has_html = df["Sentence"].str.contains(r"<[^>]+>", regex=True, na=False).sum()
    print(f"Reviews with HTML tags: {has_html}")

    all_caps = df["Sentence"].apply(lambda x: str(x).isupper() and len(str(x)) > 10).sum()
    print(f"Reviews in ALL CAPS: {all_caps}")

    has_repeats = df["Sentence"].str.contains(r"(.)\1{3,}", regex=True, na=False).sum()
    print(f"Reviews with repeated chars (like 'sooooo'): {has_repeats}")

    # 10 random samples
    print("\n=== 10 RANDOM SAMPLES ===")
    samples = df.sample(10, random_state=1)["Sentence"].tolist()
    for i, s in enumerate(samples, 1):
        s = str(s)
        preview = s[:120] + "..." if len(s) > 120 else s
        print(f"{i}. {preview}")

    # 5 shortest reviews
    print("\n=== 5 SHORTEST REVIEWS ===")
    for s in df.nsmallest(5, "length")["Sentence"]:
        print(f"  - {s!r}")

    # 3 longest reviews
    print("\n=== 3 LONGEST REVIEWS (first 200 chars) ===")
    for s in df.nlargest(3, "length")["Sentence"]:
        s = str(s)
        print(f"  - {s[:200]}...")


if __name__ == "__main__":
    inspect()