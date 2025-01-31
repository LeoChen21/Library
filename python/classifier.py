from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import csv

# Load improved models
classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-base")
ner = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english")

media_categories = ["Anime", "Manga", "Novel"]

# Extract text & metadata
def extract_info_from_website(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        text = " ".join([p.get_text() for p in soup.find_all("p")])[:1000]  # Limit to 1000 chars
        title = soup.title.string if soup.title else "Unknown"
        author = "Unknown"

        return text, title, author

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None, "Unknown", "Unknown"

# Classify website and extract media title
def classify_website(url):
    text, title, author = extract_info_from_website(url)
    if not text:
        return url, title, author, "Unknown"

    # Classify the content
    classification = classifier(text, candidate_labels=media_categories)
    category = classification["labels"][0]

    # Extract named entities (possible media title)
    ner_results = ner(text)
    extracted_titles = [ent["word"] for ent in ner_results if ent["entity"].startswith("B-")]
    if extracted_titles:
        title = extracted_titles[0]  # Take first found name

    return url, title, author, category

# Process and save results
websites = [
    "https://anitaku.io/series/ao-no-hako/",
    "https://www.royalroad.com/fiction/94398/matabar/",
    "https://asuracomic.net/series/the-greatest-estate-developer-5c0c7ac9",
]

results = [classify_website(site) for site in websites]

csv_filename = "website_classification.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Website", "Title", "Author", "Category"])
    writer.writerows(results)

print(f"Results saved to {csv_filename}")
