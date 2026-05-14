import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_PATH, SCORED_DATA_PATH
from ml.sentiment_model import score_dataframe, train_sklearn_model


def run_training_pipeline():
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"ERROR: {PROCESSED_DATA_PATH} not found.")
        print("Run demo_data.py first!")
        return

    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Loaded {len(df)} rows for training")

    print("Running VADER sentiment scoring...")
    df_scored = score_dataframe(df, text_column="cleaned_text")

    os.makedirs("data", exist_ok=True)
    df_scored.to_csv(SCORED_DATA_PATH, index=False)
    print(f"Saved scored data to {SCORED_DATA_PATH}")

    print("\nTraining sklearn model...")
    model = train_sklearn_model(df_scored, text_column="cleaned_text")

    if model:
        print("Training complete!")

    print("\nSentiment distribution:")
    print(df_scored["sentiment_label"].value_counts())

    return df_scored


if __name__ == "__main__":
    run_training_pipeline()