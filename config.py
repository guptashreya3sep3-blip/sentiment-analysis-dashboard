import os
from dotenv import load_dotenv

load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "demo_mode")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "demo_mode")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "SentimentBot/1.0")

# Data paths
RAW_DATA_PATH = "data/raw_data.csv"
PROCESSED_DATA_PATH = "data/processed_data.csv"
SCORED_DATA_PATH = "data/scored_data.csv"

# Scraping settings
DEFAULT_SUBREDDIT = "all"
DEFAULT_POST_LIMIT = 100