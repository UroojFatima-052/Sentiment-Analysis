"""
Central configuration for the Sentiment Analysis project.
All paths and settings live here so individual scripts stay clean.
"""

import os

# Base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== Data =====
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw_txt")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

BRAND_FOLDERS = ["samsung", "iphone", "xiaomi", "mixed"]

PARSED_FILE       = os.path.join(PROCESSED_DIR, "parsed_reviews.xlsx")
CLEANED_FILE      = os.path.join(PROCESSED_DIR, "cleaned_reviews.xlsx")
LABELED_FILE      = os.path.join(PROCESSED_DIR, "labeled_reviews.xlsx")        # VADER + TextBlob
LABELED_FILE_BERT = os.path.join(PROCESSED_DIR, "labeled_reviews_bert.xlsx")   # BERT (RoBERTa)

# ===== Models =====
MODELS_DIR_VADER = os.path.join(BASE_DIR, "models", "saved_vader")
MODELS_DIR_BERT  = os.path.join(BASE_DIR, "models", "saved_bert")

# ===== Labeling (VADER compound score thresholds) =====
POSITIVE_THRESHOLD =  0.05   # ← keep whatever value is in your current config.py
NEGATIVE_THRESHOLD = -0.05   # ← same

# ===== Train / test split =====
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ===== Scraper =====
SCRAPER_DELAY_SECONDS = 2

# ===== Flask app =====
UPLOAD_DIR = os.path.join(BASE_DIR, "app", "uploads")
ALLOWED_EXTENSIONS = {"xlsx", "csv"}