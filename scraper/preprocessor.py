import pandas as pd
import re
import nltk
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH

# Download all required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'&amp;|&lt;|&gt;', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_and_filter(text: str) -> str:
    if not text:
        return ""
    try:
        tokens = word_tokenize(text)
        tokens = [
            LEMMATIZER.lemmatize(token)
            for token in tokens
            if token not in STOP_WORDS and len(token) > 2
        ]
        return ' '.join(tokens)
    except Exception:
        return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if 'title' in df.columns:
        df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    else:
        df['combined_text'] = df['text'].fillna('')

    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    df['processed_text'] = df['cleaned_text'].apply(tokenize_and_filter)
    df = df[df['cleaned_text'].str.len() > 5].reset_index(drop=True)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

    return df


def load_and_preprocess(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows from {filepath}")
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()

    df_clean = preprocess_dataframe(df)

    os.makedirs("data", exist_ok=True)
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved {len(df_clean)} cleaned rows to {PROCESSED_DATA_PATH}")

    return df_clean


if __name__ == "__main__":
    df = load_and_preprocess()
    print(df[['cleaned_text', 'processed_text']].head())