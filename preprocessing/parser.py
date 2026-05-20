"""
Parser: reads all .txt review files and combines them into one master Excel file.

For every line in every .txt file, creates one row in the Excel with:
  - Sentence: the review text
  - Source_File: which .txt file it came from
  - Brand: which brand folder it was in (samsung/iphone/xiaomi/mixed)
"""

import os
import sys
import pandas as pd

# Add parent folder to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def read_reviews_from_file(filepath):
    """
    Read one .txt file and return a list of non-empty review lines.
    """
    reviews = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:  # skip empty lines
                reviews.append(line)
    return reviews


def parse_all_files():
    """
    Walk through every brand folder, read every .txt file,
    and build a list of dictionaries (one per review line).
    """
    all_rows = []

    for brand in config.BRAND_FOLDERS:
        brand_folder = os.path.join(config.RAW_DATA_DIR, brand)

        if not os.path.exists(brand_folder):
            print(f"   Warning: folder not found - {brand_folder}")
            continue

        txt_files = [f for f in os.listdir(brand_folder) if f.endswith(".txt")]
        print(f"\n{brand.upper()}: {len(txt_files)} files")

        brand_review_count = 0
        for txt_file in txt_files:
            filepath = os.path.join(brand_folder, txt_file)
            reviews = read_reviews_from_file(filepath)

            for review in reviews:
                all_rows.append({
                    "Sentence": review,
                    "Source_File": txt_file,
                    "Brand": brand
                })

            brand_review_count += len(reviews)

        print(f"   Total reviews: {brand_review_count}")

    return all_rows


def save_to_excel(rows):
    """
    Convert the list of dictionaries to a DataFrame and save as Excel.
    """
    df = pd.DataFrame(rows)

    # Make sure the processed folder exists
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    df.to_excel(config.PARSED_FILE, index=False, engine="openpyxl")
    return df


def main():
    print("Starting parser...")
    rows = parse_all_files()
    df = save_to_excel(rows)

    print(f"\nDONE.")
    print(f"Total rows in Excel: {len(df)}")
    print(f"Saved to: {config.PARSED_FILE}")
    print(f"\nRow distribution by brand:")
    print(df["Brand"].value_counts().to_string())


if __name__ == "__main__":
    main()