import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get API key securely
API_KEY = os.getenv("NEWS_API_KEY")


def scrape_news(topic: str, limit: int = 50) -> pd.DataFrame:
    """
    Fetch live news articles using NewsAPI
    and return them as a pandas DataFrame.
    """
    # Check API key
    if not API_KEY:
        print("❌ NEWS_API_KEY not found in .env file")
        return pd.DataFrame()

    try:
        # Initialize NewsAPI client
        newsapi = NewsApiClient(api_key=API_KEY)

        # Date range (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        print(f"🔍 Fetching news for: {topic}")

        # Fetch articles
        response = newsapi.get_everything(
            q=topic,
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='publishedAt',
            page_size=min(limit, 100)
        )

        articles = response.get("articles", [])

        # No articles found
        if not articles:
            print("⚠️ No articles found")
            return pd.DataFrame()

        posts = []

        # Process each article
        for i, article in enumerate(articles):

            title = article.get("title", "") or ""
            description = article.get("description", "") or ""

            # Combine title + description
            text = f"{title} {description}".strip()

            if not text:
                continue

            # Published date
            published = article.get("publishedAt", "")

            try:
                date_obj = datetime.strptime(
                    published[:10],
                    "%Y-%m-%d"
                )

                date_str = date_obj.strftime("%Y-%m-%d")

            except:
                date_str = datetime.now().strftime("%Y-%m-%d")

            # Source name
            source = article.get("source", {})
            source_name = source.get("name", "News")

            posts.append({
                "id": f"news_{i}",
                "title": title,
                "text": text,
                "date": date_str,
                "source": source_name,
                "url": article.get("url", ""),
                "author": article.get("author", "Unknown"),
                "topic": topic
            })

        # Create DataFrame
        df = pd.DataFrame(posts)

        # Create data folder if not exists
        os.makedirs("data", exist_ok=True)

        # Save raw data
        df.to_csv("data/raw_data.csv", index=False)

        print(f"✅ Successfully fetched {len(df)} articles")

        return df

    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()


# Testing
if __name__ == "__main__":

    topic = input("Enter topic: ")

    df = scrape_news(topic)

    if not df.empty:
        print(df.head())

    else:
        print("No data fetched.")