"""
Central configuration for the Sentiment Analysis project.
All paths and settings live here so individual scripts stay clean.
"""

import os

# Base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw_txt")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Brand folders
BRAND_FOLDERS = ["samsung", "iphone", "xiaomi", "mixed"]

# Processed file names
PARSED_FILE = os.path.join(PROCESSED_DIR, "parsed_reviews.xlsx")
LABELED_FILE = os.path.join(PROCESSED_DIR, "labeled_reviews.xlsx")
VERIFIED_FILE = os.path.join(PROCESSED_DIR, "verified_reviews.xlsx")
CLEANED_FILE = os.path.join(PROCESSED_DIR, "cleaned_reviews.xlsx")

# Model directory and file names
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
NB_MODEL_PATH = os.path.join(MODELS_DIR, "naive_bayes.pkl")
LR_MODEL_PATH = os.path.join(MODELS_DIR, "logistic_regression.pkl")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm.pkl")

# Sentiment label thresholds (VADER compound score)
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Scraper settings
SCRAPER_DELAY_SECONDS = 2

# Flask
UPLOAD_DIR = os.path.join(BASE_DIR, "app", "uploads")
ALLOWED_EXTENSIONS = {"xlsx", "csv"}