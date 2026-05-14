import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

# Config to hide plotly toolbar
CHART_CONFIG = {'displayModeBar': False}
# Color palette for consistent theming
COLORS = {
    "Positive": "#2ecc71",
    "Negative": "#e74c3c",
    "Neutral":  "#3498db"
}


def pie_chart(df: pd.DataFrame) -> go.Figure:
    """Sentiment distribution as a donut pie chart."""
    counts = df["sentiment_label"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    
    fig = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        color="Sentiment",
        color_discrete_map=COLORS,
        hole=0.45,
        title="Sentiment Distribution"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        margin=dict(t=50, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )
    return fig


def time_series_chart(df: pd.DataFrame) -> go.Figure:
    """Sentiment counts over time as a line chart."""
    if 'date' not in df.columns:
        return go.Figure()

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Group by date and sentiment
    daily = (
        df.groupby(['date', 'sentiment_label'])
        .size()
        .reset_index(name='count')
    )

    # Fill colors with correct rgba format
    fill_colors = {
        "Positive": "rgba(46, 204, 113, 0.15)",
        "Negative": "rgba(231, 76, 60, 0.15)",
        "Neutral":  "rgba(52, 152, 219, 0.15)"
    }

    fig = go.Figure()

    for sentiment in ["Positive", "Neutral", "Negative"]:
        subset = daily[daily['sentiment_label'] == sentiment]
        fig.add_trace(go.Scatter(
            x=subset['date'],
            y=subset['count'],
            name=sentiment,
            mode='lines+markers',
            line=dict(color=COLORS[sentiment], width=2),
            marker=dict(size=5),
            fill='tozeroy',
            fillcolor=fill_colors[sentiment]
        ))

    fig.update_layout(
        title="Sentiment Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Posts",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        margin=dict(t=50, b=60, l=40, r=20)
    )
    return fig

def bar_chart_by_subreddit(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Sentiment breakdown per top news source."""
    if 'subreddit' not in df.columns:
        return go.Figure()

    top_subs = df['subreddit'].value_counts().head(top_n).index
    df_top   = df[df['subreddit'].isin(top_subs)]

    grouped = (
        df_top.groupby(['subreddit', 'sentiment_label'])
        .size()
        .reset_index(name='count')
    )

    fig = px.bar(
        grouped,
        x='subreddit',
        y='count',
        color='sentiment_label',
        color_discrete_map=COLORS,
        title=f"Sentiment by Top {top_n} News Sources",
        barmode='stack',
        labels={'subreddit': 'News Source', 'count': 'Count'}
    )
    fig.update_layout(
        xaxis_tickangle=-35,
        xaxis_title="News Source",
        legend_title="Sentiment",
        margin=dict(t=50, b=80, l=40, r=20)
    )
    return fig

def compound_score_histogram(df: pd.DataFrame) -> go.Figure:
    """Distribution of VADER compound scores."""
    if 'sentiment_compound' not in df.columns:
        return go.Figure()
    
    fig = px.histogram(
        df,
        x='sentiment_compound',
        nbins=50,
        color_discrete_sequence=["#9b59b6"],
        title="VADER Compound Score Distribution",
        labels={'sentiment_compound': 'Compound Score (-1 = most negative, +1 = most positive)'}
    )
    fig.add_vline(x=0.05,  line_dash="dash", line_color="#2ecc71", annotation_text="Positive threshold")
    fig.add_vline(x=-0.05, line_dash="dash", line_color="#e74c3c", annotation_text="Negative threshold")
    fig.update_layout(margin=dict(t=50, b=40, l=40, r=20))
    return fig


def generate_wordcloud(df: pd.DataFrame, sentiment: str = "all") -> str:
    """
    Generate a word cloud image for given sentiment.
    Returns base64-encoded PNG string for display in Streamlit.
    """
    if sentiment != "all":
        subset = df[df['sentiment_label'] == sentiment]
    else:
        subset = df
    
    col = 'processed_text' if 'processed_text' in subset.columns else 'cleaned_text'
    
    if subset.empty or col not in subset.columns:
        return ""
    
    text = ' '.join(subset[col].dropna().tolist())
    
    if len(text.strip()) < 10:
        return ""
    
    color_map = {
        "Positive": "Greens",
        "Negative": "Reds",
        "Neutral":  "Blues",
        "all":      "viridis"
    }
    
    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap=color_map.get(sentiment, 'viridis'),
        max_words=100,
        collocations=False
    ).generate(text)
    
    # Convert to base64 for Streamlit display
    img_buffer = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    img_buffer.seek(0)
    
    img_b64 = base64.b64encode(img_buffer.read()).decode()
    return img_b64


def top_posts_table(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top N posts by Reddit score."""
    cols = ['title', 'sentiment_label', 'sentiment_compound', 'score', 'date', 'subreddit']
    available = [c for c in cols if c in df.columns]
    
    top = df.sort_values('score', ascending=False).head(n)[available]
    return top.reset_index(drop=True)