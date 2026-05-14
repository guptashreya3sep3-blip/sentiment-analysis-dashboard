import streamlit as st
import pandas as pd
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SCORED_DATA_PATH
from demo_data import generate_demo_data
from scraper.news_scraper import scrape_news
from scraper.preprocessor import preprocess_dataframe
from ml.sentiment_model import score_dataframe
from frontend.charts import (
    pie_chart, time_series_chart, bar_chart_by_subreddit,
    compound_score_histogram, generate_wordcloud, top_posts_table
)

# Hide plotly toolbar on all charts
CHART_CONFIG = {'displayModeBar': False}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    .banner-text h1 { color:#ffffff; font-size:2rem; font-weight:700; margin:0; padding:0; }
    .banner-text p  { color:#a0aec0; font-size:1rem; margin:0.3rem 0 0 0; }
    .banner-badge {
        background: rgba(99,179,237,0.15);
        border: 1px solid rgba(99,179,237,0.3);
        color: #63b3ed;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
        display: inline-block;
    }
    .banner-badge-live {
        background: rgba(72,187,120,0.15);
        border: 1px solid rgba(72,187,120,0.3);
        color: #48bb78;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
        margin-left: 0.5rem;
        display: inline-block;
    }

    .metric-row {
    display:flex;
    flex-wrap:wrap;
    gap:1rem;
    margin:1rem 0;
    }
    .metric-card {
        flex: 1;
        min-width:150px;
        background: #ffffff;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    .metric-card .value { font-size:1.8rem; font-weight:700; color:#1a202c; line-height:1.1; }
    .metric-card .sub   { font-size:0.8rem; color:#a0aec0; margin-top:0.2rem; }
    .metric-card.positive { border-top: 4px solid #48bb78; }
    .metric-card.negative { border-top: 4px solid #fc8181; }
    .metric-card.neutral  { border-top: 4px solid #63b3ed; }
    .metric-card.total    { border-top: 4px solid #9f7aea; }
    .metric-card.mood     { border-top: 4px solid #f6ad55; }

    .post-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .post-card.positive { border-left: 5px solid #48bb78; }
    .post-card.negative { border-left: 5px solid #fc8181; }
    .post-card .post-title { font-size:0.95rem; font-weight:600; color:#2d3748; margin-bottom:0.5rem; }
    .post-card .post-meta  { font-size:0.78rem; color:#a0aec0; }

    .score-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .score-pos { background:#c6f6d5; color:#276749; }
    .score-neg { background:#fed7d7; color:#9b2c2c; }

    .data-source-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .live-badge  { background:#c6f6d5; color:#276749; }
    .demo-badge  { background:#fefcbf; color:#744210; }

    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2d3748;
        padding: 0.5rem 0;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 1rem;
    }

    .vs-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        border-radius: 50%;
        text-align: center;
        width: 45px;
        height: 45px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: auto;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    div[data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
defaults = {
    "df": None, "df2": None,
    "current_topic": "", "current_topic2": "",
    "search_history": [],
    "mode": "Single Topic",
    "data_source": "demo", "data_source2": "demo"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Pipeline ───────────────────────────────────────────────────────────────────
def run_pipeline(topic: str, limit: int):
    progress = st.progress(0)
    status   = st.empty()
    source   = "demo"

    status.text(f"Fetching real news for '{topic}'...")
    progress.progress(15)
    df_raw = scrape_news(topic, limit=limit)

    if df_raw.empty:
        status.text(f"Generating demo data for '{topic}'...")
        df_raw = generate_demo_data(topic, num_posts=limit)
        source = "demo"
    else:
        source = "live"

    if df_raw.empty:
        st.error("Failed to generate data.")
        progress.empty()
        status.empty()
        return None, source

    progress.progress(40)
    status.text("Cleaning text...")
    df_clean = preprocess_dataframe(df_raw)

    progress.progress(70)
    status.text("Running sentiment analysis...")
    df_scored = score_dataframe(df_clean, text_column="cleaned_text")

    progress.progress(90)
    status.text("Saving...")
    os.makedirs("data", exist_ok=True)
    df_scored.to_csv(SCORED_DATA_PATH, index=False)

    progress.progress(100)
    status.text("Done!")
    time.sleep(0.4)
    progress.empty()
    status.empty()

    if topic not in st.session_state.search_history:
        st.session_state.search_history.insert(0, topic)
        st.session_state.search_history = st.session_state.search_history[:8]

    return df_scored, source


# ── UI components ──────────────────────────────────────────────────────────────
def render_source_badge(source: str):
    if source == "live":
        st.markdown(
            '<span class="data-source-badge live-badge">🟢 Live News Data (NewsAPI)</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="data-source-badge demo-badge">🟡 Demo Data — paste real NewsAPI key in .env for live data</span>',
            unsafe_allow_html=True
        )


def render_metric_cards(df, topic_label):
    total     = len(df)
    pos_count = len(df[df['sentiment_label'] == 'Positive'])
    neg_count = len(df[df['sentiment_label'] == 'Negative'])
    neu_count = len(df[df['sentiment_label'] == 'Neutral'])
    avg_comp  = df['sentiment_compound'].mean() if 'sentiment_compound' in df.columns else 0

    if avg_comp > 0.05:
        mood_text, mood_color = "😊 Positive", "#48bb78"
    elif avg_comp < -0.05:
        mood_text, mood_color = "😠 Negative", "#fc8181"
    else:
        mood_text, mood_color = "😐 Neutral", "#63b3ed"

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card total">
            <div class="label">Total Articles</div>
            <div class="value">{total}</div>
            <div class="sub">{topic_label}</div>
        </div>
        <div class="metric-card positive">
            <div class="label">Positive</div>
            <div class="value" style="color:#48bb78">{pos_count}</div>
            <div class="sub">{pos_count/total*100:.1f}%</div>
        </div>
        <div class="metric-card negative">
            <div class="label">Negative</div>
            <div class="value" style="color:#fc8181">{neg_count}</div>
            <div class="sub">{neg_count/total*100:.1f}%</div>
        </div>
        <div class="metric-card neutral">
            <div class="label">Neutral</div>
            <div class="value" style="color:#63b3ed">{neu_count}</div>
            <div class="sub">{neu_count/total*100:.1f}%</div>
        </div>
        <div class="metric-card mood">
            <div class="label">Overall Mood</div>
            <div class="value" style="color:{mood_color};font-size:1.2rem">{mood_text}</div>
            <div class="sub">avg: {avg_comp:.3f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_highlighted_posts(df):
    if 'sentiment_compound' not in df.columns or df.empty:
        return

     # Filter articles related to searched topic only
    topic = str(st.session_state.current_topic).lower()
    filtered_df = df[
        df['title'].fillna('').str.lower().str.contains(topic)
    ]
    # fallback if nothing matches
    if filtered_df.empty:
        filtered_df = df
    # Get most positive and negative article
    best = filtered_df.loc[
        filtered_df['sentiment_compound'].idxmax()
    ]
    worst = filtered_df.loc[
        filtered_df['sentiment_compound'].idxmin()
    ]
    st.markdown(
        '<div class="section-header">🌟 Most Positive vs Most Negative Article</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        title  = str(best.get('title', best.get('cleaned_text', '')))[:250]
        source = best.get('subreddit', 'N/A')
        comp   = best.get('sentiment_compound', 0)
        date   = str(best.get('date', ''))[:10]
        st.markdown(f"""
        <div class="post-card positive">
            <div class="post-title">✅ {title}</div>
            <div class="post-meta">
                <span class="score-badge score-pos">score: {comp:.3f}</span>
                <span>📰 {source} &nbsp;•&nbsp; 📅 {date}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        title  = str(worst.get('title', worst.get('cleaned_text', '')))[:250]
        source = worst.get('subreddit', 'N/A')
        comp   = worst.get('sentiment_compound', 0)
        date   = str(worst.get('date', ''))[:10]
        st.markdown(f"""
        <div class="post-card negative">
            <div class="post-title">❌ {title}</div>
            <div class="post-meta">
                <span class="score-badge score-neg">score: {comp:.3f}</span>
                <span>📰 {source} &nbsp;•&nbsp; 📅 {date}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_full_dashboard(df, topic_label, source="demo"):
    render_source_badge(source)
    render_metric_cards(df, topic_label)
    st.markdown("---")

    # Row 1
    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(pie_chart(df),
                        use_container_width=True, config=CHART_CONFIG)
    with col2:
        st.plotly_chart(time_series_chart(df),
                        use_container_width=True, config=CHART_CONFIG)

    # Row 2
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(compound_score_histogram(df),
                        use_container_width=True, config=CHART_CONFIG)
    with col4:
        if 'subreddit' in df.columns:
            st.plotly_chart(bar_chart_by_subreddit(df),
                            use_container_width=True, config=CHART_CONFIG)

    st.markdown("---")
    render_highlighted_posts(df)
    st.markdown("---")

    # Word clouds
    st.markdown('<div class="section-header">☁️ Word Clouds</div>',
                unsafe_allow_html=True)
    wc_tabs = st.tabs(["All", "Positive", "Negative", "Neutral"])
    for i, sentiment in enumerate(["all", "Positive", "Negative", "Neutral"]):
        with wc_tabs[i]:
            img_b64 = generate_wordcloud(df, sentiment)
            if img_b64:
                st.markdown(
                    f'<img src="data:image/png;base64,{img_b64}" '
                    f'style="width:100%;border-radius:8px"/>',
                    unsafe_allow_html=True
                )
            else:
                st.info("Not enough text for this category.")

    st.markdown("---")

    # Top articles table
    st.markdown('<div class="section-header">🏆 Top Articles</div>',
                unsafe_allow_html=True)
    top_df = top_posts_table(df, n=15)

    def highlight_sentiment(val):
        colors = {"Positive": "#c6f6d5", "Negative": "#fed7d7", "Neutral": "#bee3f8"}
        return f"background-color: {colors.get(val, 'white')}"

    if 'sentiment_label' in top_df.columns:
        st.dataframe(
            top_df.style.map(highlight_sentiment, subset=['sentiment_label']),
            use_container_width=True, height=380
        )
    else:
        st.dataframe(top_df, use_container_width=True)

    st.markdown("---")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download CSV", data=csv,
        file_name=f"sentiment_{topic_label.replace(' ','_')}.csv",
        mime="text/csv"
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0;">
        <div style="font-size:2.5rem;">📊</div>
        <div style="font-size:1.1rem;font-weight:700;">Sentiment Dashboard</div>
        <div style="font-size:0.75rem;color:#a0aec0;margin-top:0.2rem;">
            VADER + sklearn + NewsAPI
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    mode = st.radio("Mode", ["Single Topic", "Compare Two Topics"])
    st.session_state.mode = mode
    st.markdown("---")

    if mode == "Single Topic":
        topic_input  = st.text_input("🔎 Topic / Keyword", value="artificial intelligence")
        post_limit   = st.slider("Number of articles", 20, 100, 50, 5)
        analyze_btn  = st.button("🚀 Analyze Now", type="primary", use_container_width=True)
        topic_input2 = None
    else:
        topic_input  = st.text_input("🔎 Topic 1", value="Tesla")
        topic_input2 = st.text_input("🔎 Topic 2", value="Bitcoin")
        post_limit   = st.slider("Articles per topic", 20, 100, 50, 5)
        analyze_btn  = st.button("⚡ Compare Now", type="primary", use_container_width=True)

    st.markdown("---")
    load_btn = st.button("📂 Load Last Result", use_container_width=True)

    st.markdown("---")
    if st.session_state.search_history:
        st.markdown("**🕐 Recent Searches**")
        for t in st.session_state.search_history:
            st.markdown(f"• {t}")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:#718096;text-align:center;line-height:1.8;">
        🟢 Live data via NewsAPI<br>
        🟡 Demo mode as fallback<br>
        Add NEWS_API_KEY in .env
    </div>
    """, unsafe_allow_html=True)


# ── Banner ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
    <div style="font-size:3rem;">📊</div>
    <div class="banner-text">
        <h1>Sentiment Analysis Dashboard</h1>
        <p>Analyze public opinion from real news using AI-powered sentiment analysis</p>
        <span class="banner-badge">🤖 VADER + scikit-learn + Plotly</span>
        <span class="banner-badge-live">📰 NewsAPI</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Search history quick buttons ───────────────────────────────────────────────
if st.session_state.search_history:
    st.markdown("**Quick search:**")
    cols = st.columns(min(len(st.session_state.search_history), 4))
    for i, topic in enumerate(st.session_state.search_history[:4]):
        with cols[i]:
            if st.button(f"🔍 {topic}", key=f"hist_{i}", use_container_width=True):
                df, src = run_pipeline(topic, 50)
                if df is not None:
                    st.session_state.df            = df
                    st.session_state.df2           = None
                    st.session_state.data_source   = src
                    st.session_state.current_topic = topic
                    st.rerun()


# ── Button handlers ────────────────────────────────────────────────────────────
if analyze_btn:
    if mode == "Single Topic" and topic_input:
        df, src = run_pipeline(topic_input, post_limit)
        if df is not None:
            st.session_state.df            = df
            st.session_state.df2           = None
            st.session_state.current_topic = topic_input
            st.session_state.data_source   = src
            if src == "live":
                st.success(f"✅ Fetched **{len(df)}** real news articles for **{topic_input}**")
            else:
                st.info("Using demo data — paste your NewsAPI key in .env for real articles")

    elif mode == "Compare Two Topics" and topic_input and topic_input2:
        col_a, col_b = st.columns(2)
        with col_a:
            with st.spinner(f"Analyzing {topic_input}..."):
                df1, src1 = run_pipeline(topic_input, post_limit)
        with col_b:
            with st.spinner(f"Analyzing {topic_input2}..."):
                df2, src2 = run_pipeline(topic_input2, post_limit)

        if df1 is not None and df2 is not None:
            st.session_state.df             = df1
            st.session_state.df2            = df2
            st.session_state.current_topic  = topic_input
            st.session_state.current_topic2 = topic_input2
            st.session_state.data_source    = src1
            st.session_state.data_source2   = src2
            st.success(f"✅ Compared **{topic_input}** vs **{topic_input2}**")

if load_btn:
    if os.path.exists(SCORED_DATA_PATH):
        df = pd.read_csv(SCORED_DATA_PATH)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        st.session_state.df          = df
        st.session_state.df2         = None
        st.session_state.data_source = "demo"
        st.success(f"Loaded {len(df)} rows")
    else:
        st.warning("No saved data found. Run an analysis first.")


# ── Main content ───────────────────────────────────────────────────────────────
df  = st.session_state.df
df2 = st.session_state.df2

if df is None or df.empty:
    st.markdown("---")
    st.info("👈 Enter a topic in the sidebar and click **Analyze Now** to get started.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Step 1** — Enter a topic\n\nTesla, Bitcoin, AI, Cricket...")
    with col2:
        st.markdown("**Step 2** — Choose article count\n\nMore = more accurate results")
    with col3:
        st.markdown("**Step 3** — View insights\n\nCharts, word clouds, highlights")

elif df2 is not None and not df2.empty:

    # =========================
    # COMPARE MODE
    # =========================

    t1 = st.session_state.current_topic
    t2 = st.session_state.current_topic2

    src1 = st.session_state.data_source
    src2 = st.session_state.data_source2

    st.markdown(
        f"""
        <p style="color:#cbd5e0;margin-top:0.5rem;">
            Real-time sentiment comparison between
            <b>{t1}</b> and <b>{t2}</b>
        </p>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # MAIN RESPONSIVE COLUMNS
    # =====================================================

    left_col, right_col = st.columns(2)

    # =====================================================
    # LEFT SIDE
    # =====================================================

    with left_col:

        st.markdown(f"# 🔵 {t1}")

        render_source_badge(src1)

        render_metric_cards(df, t1)

        st.markdown("### 📊 Sentiment Distribution")

        st.plotly_chart(
            pie_chart(df),
            use_container_width=True,
            config=CHART_CONFIG
        )

        st.markdown("### 📈 Sentiment Trend")

        st.plotly_chart(
            time_series_chart(df),
            use_container_width=True,
            config=CHART_CONFIG
        )

        st.markdown("### 📉 Sentiment Histogram")

        st.plotly_chart(
            compound_score_histogram(df),
            use_container_width=True,
            config=CHART_CONFIG
        )

        if 'subreddit' in df.columns:

            st.markdown("### 📰 News Sources")

            st.plotly_chart(
                bar_chart_by_subreddit(df),
                use_container_width=True,
                config=CHART_CONFIG
            )

        render_highlighted_posts(df)

    # =====================================================
    # RIGHT SIDE
    # =====================================================

    with right_col:

        st.markdown(f"# 🔴 {t2}")

        render_source_badge(src2)

        render_metric_cards(df2, t2)

        st.markdown("### 📊 Sentiment Distribution")

        st.plotly_chart(
            pie_chart(df2),
            use_container_width=True,
            config=CHART_CONFIG
        )

        st.markdown("### 📈 Sentiment Trend")

        st.plotly_chart(
            time_series_chart(df2),
            use_container_width=True,
            config=CHART_CONFIG
        )

        st.markdown("### 📉 Sentiment Histogram")

        st.plotly_chart(
            compound_score_histogram(df2),
            use_container_width=True,
            config=CHART_CONFIG
        )

        if 'subreddit' in df2.columns:

            st.markdown("### 📰 News Sources")

            st.plotly_chart(
                bar_chart_by_subreddit(df2),
                use_container_width=True,
                config=CHART_CONFIG
            )

        render_highlighted_posts(df2)

    st.markdown("---")

    # =====================================================
    # WORD CLOUDS
    # =====================================================

    st.markdown("""
    <div class="section-header">
        ☁️ Word Cloud Comparison
    </div>
    """, unsafe_allow_html=True)

    wc1, wc2 = st.columns(2)

    with wc1:

        st.markdown(f"### 🔵 {t1}")

        img = generate_wordcloud(df, "all")

        if img:

            st.markdown(
                f"""
                <img src="data:image/png;base64,{img}"
                style="
                    width:100%;
                    border-radius:14px;
                    border:1px solid #e2e8f0;
                ">
                """,
                unsafe_allow_html=True
            )

    with wc2:

        st.markdown(f"### 🔴 {t2}")

        img2 = generate_wordcloud(df2, "all")

        if img2:

            st.markdown(
                f"""
                <img src="data:image/png;base64,{img2}"
                style="
                    width:100%;
                    border-radius:14px;
                    border:1px solid #e2e8f0;
                ">
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # =====================================================
    # DOWNLOAD BUTTONS
    # =====================================================

    d1, d2 = st.columns(2)

    with d1:

        csv1 = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            f"📥 Download {t1} CSV",
            data=csv1,
            file_name=f"{t1}_sentiment.csv",
            mime="text/csv",
            use_container_width=True
        )

    with d2:

        csv2 = df2.to_csv(index=False).encode("utf-8")

        st.download_button(
            f"📥 Download {t2} CSV",
            data=csv2,
            file_name=f"{t2}_sentiment.csv",    
            mime="text/csv",
            use_container_width=True
        )
else:
    # Single topic
    topic_label = st.session_state.current_topic or "loaded data"
    source      = st.session_state.data_source
    st.markdown(f"## Results for: **{topic_label}**")
    render_full_dashboard(df, topic_label, source)