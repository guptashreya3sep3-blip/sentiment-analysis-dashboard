import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VADER = SentimentIntensityAnalyzer()
MODEL_PATH = "ml/sentiment_pipeline.pkl"


def get_vader_sentiment(text: str) -> dict:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0, "label": "Neutral"}

    scores = VADER.polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "pos": round(scores['pos'], 4),
        "neg": round(scores['neg'], 4),
        "neu": round(scores['neu'], 4),
        "compound": round(compound, 4),
        "label": label
    }


def score_dataframe(df: pd.DataFrame, text_column: str = "cleaned_text") -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    results = df[text_column].fillna("").apply(get_vader_sentiment)

    df["sentiment_pos"]      = results.apply(lambda x: x["pos"])
    df["sentiment_neg"]      = results.apply(lambda x: x["neg"])
    df["sentiment_neu"]      = results.apply(lambda x: x["neu"])
    df["sentiment_compound"] = results.apply(lambda x: x["compound"])
    df["sentiment_label"]    = results.apply(lambda x: x["label"])

    return df


def build_sklearn_pipeline():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            strip_accents='unicode',
            analyzer='word'
        )),
        ('clf', LogisticRegression(
            max_iter=500,
            C=1.0,
            solver='lbfgs',
            multi_class='multinomial',
            random_state=42
        ))
    ])
    return pipeline


def train_sklearn_model(df: pd.DataFrame, text_column: str = "cleaned_text"):
    if df.empty or "sentiment_label" not in df.columns:
        print("Need scored data to train.")
        return None

    df_train = df[df[text_column].str.len() > 5].copy()
    X = df_train[text_column]
    y = df_train["sentiment_label"]

    if len(X) < 20:
        print("Not enough data to train")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_sklearn_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))

    os.makedirs("ml", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Model saved to {MODEL_PATH}")
    return pipeline


def load_sklearn_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_sentiment(texts: list, use_sklearn: bool = False) -> list:
    if use_sklearn:
        model = load_sklearn_model()
        if model:
            return model.predict(texts).tolist()
    return [get_vader_sentiment(t)["label"] for t in texts]