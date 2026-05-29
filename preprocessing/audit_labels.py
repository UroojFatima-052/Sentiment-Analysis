"""
Audit script: pulls a stratified sample of labeled reviews so we can evaluate
how well the labeling actually performed.

Samples 10 rows from each label group (positive, negative, neutral).
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# File to audit — change this path if needed
AUDIT_FILE = os.path.join(config.PROCESSED_DIR, "labeled_reviews_bert.xlsx")


def sample_for_audit(df, label, n=10):
    """Get up to n random samples matching the given label."""
    subset = df[df["Final_Label"] == label]
    return subset.sample(min(n, len(subset)), random_state=42)


def main():
    print(f"Loading: {AUDIT_FILE}\n")
    df = pd.read_excel(AUDIT_FILE)

    print(f"Total rows: {len(df)}")
    print(f"\nLabel distribution:")
    print(df["Final_Label"].value_counts().to_string())

    print("\nPulling stratified samples for audit...\n")

    output_lines = []

    for label in ["positive", "negative", "neutral"]:
        samples = sample_for_audit(df, label, n=10)
        output_lines.append(f"\n{'='*70}")
        output_lines.append(f"  {label.upper()} samples ({len(samples)})")
        output_lines.append(f"{'='*70}\n")

        for i, row in enumerate(samples.itertuples(), 1):
            sentence = str(row.Sentence)[:200]
            output_lines.append(f"{i}. [{label}] {sentence}")
        output_lines.append("")

    full_output = "\n".join(output_lines)
    print(full_output)

    # Save to text file too
    audit_path = os.path.join(config.PROCESSED_DIR, "audit_sample_bert.txt")
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write(full_output)
    print(f"\nAudit saved to: {audit_path}")


if __name__ == "__main__":
    main()