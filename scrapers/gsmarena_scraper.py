"""
GSMArena scraper for mobile phone user opinions.

Reads phone URLs from phone_urls.py and saves reviews as .txt files,
one file per page of reviews. Saves files to data/raw_txt/<brand>/.
"""

import os
import sys
import time
import requests
from bs4 import BeautifulSoup

# Add parent folder to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from phone_urls import PHONE_URLS, PAGES_PER_PHONE


# Pretend to be a normal browser so GSMArena doesn't block us
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_phone_name_from_url(url):
    """
    Extract a clean phone name from a GSMArena URL.
    Example: 'samsung_galaxy_s24_ultra-reviews-12771.php' -> 'samsung_galaxy_s24_ultra'
    """
    filename = url.split("/")[-1]                # samsung_galaxy_s24_ultra-reviews-12771.php
    phone_name = filename.split("-reviews-")[0]  # samsung_galaxy_s24_ultra
    return phone_name


def build_page_url(base_url, page_number):
    """
    Build the URL for a specific page of reviews.
    Page 1 has no suffix; pages 2+ append p2, p3, etc. before .php.
    """
    if page_number == 1:
        return base_url
    # Replace '.php' with 'p<N>.php'
    return base_url.replace(".php", f"p{page_number}.php")


def scrape_page(url):
    """
    Download one review page and extract clean review texts.
    Removes quoted parent comments and URLs from each review.
    Returns a list of review strings.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"   Failed to fetch {url}: {error}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    review_paragraphs = soup.find_all("p", class_="uopin")

    reviews = []
    for paragraph in review_paragraphs:
        # Remove the quoted parent comment elements before extracting text
        for quoted_link in paragraph.find_all("a", class_="uinreply"):
            quoted_link.decompose()
        for quoted_msg in paragraph.find_all("span", class_="uinreply-msg"):
            quoted_msg.decompose()

        # Now extract clean text
        text = paragraph.get_text(strip=True)

        # Skip empty or very short reviews
        if not text or len(text) < 10:
            continue

        # Skip reviews that are mostly just a URL
        if text.startswith("http") and " " not in text:
            continue

        reviews.append(text)

    return reviews


def save_reviews_to_file(reviews, brand, phone_name, page_number):
    """
    Save a list of reviews to a .txt file, one review per line.
    File goes to data/raw_txt/<brand>/<phone_name>_pN.txt
    """
    folder = os.path.join(config.RAW_DATA_DIR, brand)
    os.makedirs(folder, exist_ok=True)

    filename = f"{phone_name}_p{page_number}.txt"
    filepath = os.path.join(folder, filename)

    with open(filepath, "w", encoding="utf-8") as file:
        for review in reviews:
            file.write(review + "\n")

    return filepath


def scrape_phone(brand, phone_url):
    """
    Scrape all configured pages for one phone.
    Returns the total number of reviews collected.
    """
    phone_name = get_phone_name_from_url(phone_url)
    print(f"\nScraping {brand} / {phone_name}")
    total_reviews = 0

    for page_number in range(1, PAGES_PER_PHONE + 1):
        page_url = build_page_url(phone_url, page_number)
        reviews = scrape_page(page_url)

        if not reviews:
            print(f"   Page {page_number}: no reviews found, stopping this phone")
            break

        filepath = save_reviews_to_file(reviews, brand, phone_name, page_number)
        print(f"   Page {page_number}: {len(reviews)} reviews -> {os.path.basename(filepath)}")
        total_reviews += len(reviews)

        # Polite delay between requests so we don't get blocked
        time.sleep(config.SCRAPER_DELAY_SECONDS)

    return total_reviews


def main():
    """Run the scraper for all brands and all phones."""
    print("Starting GSMArena scraper...")
    print(f"Brands: {list(PHONE_URLS.keys())}")
    print(f"Pages per phone: {PAGES_PER_PHONE}")

    grand_total = 0
    for brand, urls in PHONE_URLS.items():
        print(f"\n=== Brand: {brand.upper()} ({len(urls)} phones) ===")
        brand_total = 0
        for url in urls:
            brand_total += scrape_phone(brand, url)
        print(f"\n{brand}: {brand_total} reviews collected")
        grand_total += brand_total

    print(f"\nDONE. Total reviews collected: {grand_total}")


if __name__ == "__main__":
    main()