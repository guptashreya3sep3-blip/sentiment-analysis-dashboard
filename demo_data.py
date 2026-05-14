import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import hashlib

POSITIVE_TEMPLATES = [
    "{topic} is absolutely amazing, changed my life completely!",
    "Just started using {topic} and I love it so much!",
    "The future of {topic} looks incredibly bright and promising.",
    "{topic} exceeded all my expectations, highly recommend!",
    "I am so impressed with how {topic} has evolved recently.",
    "Best decision I ever made was getting into {topic}!",
    "{topic} community is so helpful and supportive, love it!",
    "Finally tried {topic} and wow, it is better than I thought!",
    "The improvements in {topic} this year have been outstanding.",
    "{topic} just keeps getting better and better every day!",
    "Really happy with my experience with {topic} so far.",
    "{topic} solved all my problems, cannot recommend enough!",
    "Incredible results with {topic}, exceeded my expectations.",
    "The {topic} team is doing a fantastic job, keep it up!",
    "{topic} is the best thing that happened to me this year.",
    "Loving every moment with {topic}, truly exceptional!",
    "{topic} has completely transformed the way I work daily.",
    "So grateful for {topic}, it makes everything so much easier.",
    "The {topic} experience is unlike anything I have seen before.",
    "Everyone should try {topic}, you will not regret it at all!",
]

NEGATIVE_TEMPLATES = [
    "{topic} is completely overrated and not worth the hype.",
    "I am really disappointed with {topic}, expected much better.",
    "{topic} has so many problems, when will they fix it?",
    "Tried {topic} for a week and honestly it was terrible.",
    "The worst experience I had was with {topic}, avoid it.",
    "{topic} is going downhill fast, very sad to see this.",
    "Cannot believe how bad {topic} has become recently.",
    "So frustrated with {topic}, nothing works as advertised.",
    "{topic} keeps crashing and support is completely useless.",
    "Totally regret spending money on {topic}, waste of time.",
    "{topic} is broken and the developers do not care at all.",
    "The quality of {topic} has dropped significantly lately.",
    "Awful experience with {topic}, would not recommend to anyone.",
    "{topic} promised a lot but delivered absolutely nothing.",
    "Giving up on {topic}, too many issues and no fixes coming.",
    "Do not waste your time on {topic}, complete disappointment.",
    "{topic} support is the worst I have ever dealt with.",
    "Seriously regretting my decision to try {topic} at all.",
    "The problems with {topic} just keep piling up endlessly.",
    "{topic} is a scam, stay far away from it please.",
]

NEUTRAL_TEMPLATES = [
    "Anyone else been following the latest news about {topic}?",
    "What do you all think about the recent changes in {topic}?",
    "I have been using {topic} for a while now, it is okay.",
    "Interesting article about {topic} I read today.",
    "Has anyone compared {topic} with other alternatives?",
    "{topic} seems to be trending a lot lately, thoughts?",
    "Looking for more information about {topic}, any suggestions?",
    "Just read about {topic} in the news, what do you think?",
    "Been thinking about trying {topic}, is it worth it?",
    "What is everyone's experience with {topic} so far?",
    "Saw a discussion about {topic} earlier, pretty interesting.",
    "Can someone explain how {topic} actually works in detail?",
    "There are pros and cons to {topic} that people miss.",
    "The debate around {topic} continues, no clear winner yet.",
    "Statistics about {topic} are quite mixed and confusing.",
    "Just discovered {topic} today, still forming my opinion.",
    "Has {topic} changed much over the past few months?",
    "Not sure what to think about {topic} honestly.",
    "Heard a lot about {topic} but never tried it myself.",
    "The reviews for {topic} are all over the place online.",
]

SUBREDDITS = [
    "technology", "news", "worldnews", "tech", "science",
    "programming", "artificial", "MachineLearning", "Python",
    "datascience", "business", "investing", "stocks", "finance",
    "gaming", "movies", "music", "sports", "health", "education"
]


def get_topic_seed(topic: str) -> int:
    """Generate a unique seed based on the topic string."""
    return int(hashlib.md5(topic.lower().encode()).hexdigest()[:8], 16)


def get_topic_distribution(topic: str):
    """
    Give different sentiment distributions for different topics.
    Makes results feel realistic and unique per topic.
    """
    topic_lower = topic.lower()

    # Positive-leaning topics
    positive_topics = ['cricket', 'virat kohli', 'ms dhoni', 'music', 'nature',
                       'python', 'space', 'science', 'dogs', 'cats', 'food']
    # Negative-leaning topics
    negative_topics = ['inflation', 'recession', 'war', 'pollution', 'crime',
                       'corruption', 'unemployment', 'poverty', 'disease']
    # Neutral-leaning topics
    neutral_topics = ['politics', 'government', 'economy', 'news', 'stock',
                      'bitcoin', 'crypto', 'election', 'policy']

    if any(t in topic_lower for t in positive_topics):
        return 0.55, 0.20, 0.25   # pos, neg, neu
    elif any(t in topic_lower for t in negative_topics):
        return 0.20, 0.55, 0.25
    elif any(t in topic_lower for t in neutral_topics):
        return 0.30, 0.30, 0.40
    else:
        # Default: slightly positive
        seed = get_topic_seed(topic) % 100
        pos = 0.35 + (seed % 20) / 100
        neg = 0.25 + (seed % 15) / 100
        neu = max(0.1, 1.0 - pos - neg)
        return pos, neg, neu


def generate_demo_data(topic: str, num_posts: int = 150) -> pd.DataFrame:
    """
    Generate realistic fake Reddit posts for any topic.
    Each topic gets unique stats and sentiment distribution.
    """
    # Use topic-based seed so same topic = same results, different topic = different results
    seed = get_topic_seed(topic)
    random.seed(seed)
    np.random.seed(seed)

    pos_ratio, neg_ratio, neu_ratio = get_topic_distribution(topic)

    posts = []
    end_date = datetime.now()

    for i in range(num_posts):
        rand = random.random()
        if rand < pos_ratio:
            template = random.choice(POSITIVE_TEMPLATES)
            sentiment_hint = "positive"
        elif rand < pos_ratio + neg_ratio:
            template = random.choice(NEGATIVE_TEMPLATES)
            sentiment_hint = "negative"
        else:
            template = random.choice(NEUTRAL_TEMPLATES)
            sentiment_hint = "neutral"

        text = template.format(topic=topic)

        random_days  = random.randint(0, 30)
        random_hours = random.randint(0, 23)
        post_date = end_date - timedelta(days=random_days, hours=random_hours)

        posts.append({
            "id": f"demo_{i}_{random.randint(10000, 99999)}",
            "title": text,
            "text": text,
            "score": random.randint(1, 5000),
            "num_comments": random.randint(0, 500),
            "created_utc": post_date.strftime("%Y-%m-%d %H:%M:%S"),
            "date": post_date.strftime("%Y-%m-%d"),
            "subreddit": random.choice(SUBREDDITS),
            "url": f"https://reddit.com/r/demo/post_{i}",
            "author": f"user_{random.randint(1000, 9999)}",
            "topic": topic,
            "sentiment_hint": sentiment_hint
        })

    df = pd.DataFrame(posts)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_data.csv", index=False)
    print(f"Generated {len(df)} demo posts for topic: '{topic}'")

    return df


if __name__ == "__main__":
    topic = input("Enter a topic to generate demo data for: ")
    df = generate_demo_data(topic, num_posts=150)
    print(df[['title', 'date', 'subreddit', 'score']].head(10))