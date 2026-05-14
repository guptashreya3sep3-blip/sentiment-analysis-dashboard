import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()


def scrape_news(topic: str, limit: int = 100) -> pd.DataFrame:
    """
    Scrape real news articles for any topic using NewsAPI.
    """
    api_key = os.getenv("NEWS_API_KEY", "")

    if not api_key or api_key == "paste_your_actual_newsapi_key_here":
        print("No NewsAPI key found, falling back to demo data")
        return pd.DataFrame()

    try:
        newsapi = NewsApiClient(api_key=api_key)

        end_date   = datetime.now()
        start_date = end_date - timedelta(days=30)

        print(f"Fetching real news for '{topic}'...")

        response = newsapi.get_everything(
            q=topic,
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='publishedAt',
            page_size=min(limit, 100)
        )

        articles = response.get('articles', [])

        if not articles:
            print("No articles found for this topic")
            return pd.DataFrame()

        posts = []
        for i, article in enumerate(articles):
            # Combine title and description for better sentiment analysis
            title = article.get('title', '') or ''
            desc  = article.get('description', '') or ''
            text  = (title + ' ' + desc).strip()

            if not text or text == ' ':
                continue

            # Parse published date
            published = article.get('publishedAt', '')
            try:
                date_obj = datetime.strptime(published[:10], '%Y-%m-%d')
                date_str = date_obj.strftime('%Y-%m-%d')
            except Exception:
                date_str = datetime.now().strftime('%Y-%m-%d')

            # Get source name as subreddit equivalent
            source = article.get('source', {})
            source_name = source.get('name', 'News') if source else 'News'

            posts.append({
                "id":           f"news_{i}_{topic[:5]}",
                "title":        title,
                "text":         text,
                "score":        0,
                "num_comments": 0,
                "created_utc":  published,
                "date":         date_str,
                "subreddit":    source_name,
                "url":          article.get('url', ''),
                "author":       article.get('author', 'unknown') or 'unknown',
                "topic":        topic
            })

        df = pd.DataFrame(posts)

        if df.empty:
            return pd.DataFrame()

        os.makedirs("data", exist_ok=True)
        df.to_csv("data/raw_data.csv", index=False)
        print(f"✅ Fetched {len(df)} real news articles for '{topic}'")

        return df

    except Exception as e:
        print(f"NewsAPI error: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    topic = input("Enter topic: ")
    df = scrape_news(topic, limit=100)
    if not df.empty:
        print(df[['title', 'date', 'subreddit']].head(10))
    else:
        print("No data returned")