"""
Streamlit app: app_streamlit.py
Run with:
 streamlit run app_streamlit.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set(style="whitegrid")
st.set_page_config(page_title="CORD-19 Data Explorer", layout="wide")

@st.cache_data
def load_data(path: str):
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        st.error(f"Could not read {path}: {e}")
        return pd.DataFrame()
    # minimal cleaning
    df['publish_time'] = pd.to_datetime(df.get('publish_time', pd.NaT), errors='coerce')
    df['year'] = df['publish_time'].dt.year
    df['title'] = df.get('title', '').astype(str)
    df['abstract'] = df.get('abstract', '').astype(str)
    df['journal'] = df.get('journal', '').fillna('Unknown').astype(str)
    df['source'] = df.get('source', 'unknown').fillna('unknown').astype(str)
    df['abstract_word_count'] = df['abstract'].apply(lambda x: 0 if pd.isna(x) or x=='' else len(str(x).split()))
    df['title_word_count'] = df['title'].apply(lambda x: 0 if pd.isna(x) or x=='' else len(str(x).split()))
    return df

st.title("CORD-19 Data Explorer")
st.write("Interactive exploration of the CORD-19 `metadata.csv` file")

# Try to load cleaned file first (if the analysis script was run)
data_path_candidates = ['cord19_cleaned_metadata.csv', 'metadata.csv']
data_path = None
for p in data_path_candidates:
    if Path(p).exists():
        data_path = p
        break

if data_path is None:
    st.error("Couldn't find cord19_cleaned_metadata.csv or metadata.csv in working directory.")
    st.info("Place metadata.csv (downloaded from Kaggle) in this folder or run the analysis script first.")
    st.stop()

df = load_data(data_path)
st.sidebar.write(f"Loaded {data_path} — {df.shape[0]} rows")

# Sidebar controls
min_year = int(df['year'].min()) if not df['year'].dropna().empty else 2019
max_year = int(df['year'].max()) if not df['year'].dropna().empty else 2022
year_range = st.sidebar.slider("Select year range", min_year, max_year, (min_year, max_year))

# Journal select (top N)
top_journals = df['journal'].value_counts().head(50)
journal_options = list(top_journals.index)
selected_journal = st.sidebar.selectbox("Filter by journal (top 50)", options=["All"] + journal_options)

# Source select (top)
top_sources = df['source'].value_counts().head(20)
selected_source = st.sidebar.selectbox("Filter by source (top 20)", options=["All"] + list(top_sources.index))

# Apply filters
df_filtered = df[
    df['year'].between(year_range[0], year_range[1], inclusive='both')
]
if selected_journal != "All":
    df_filtered = df_filtered[df_filtered['journal'] == selected_journal]
if selected_source != "All":
    df_filtered = df_filtered[df_filtered['source'] == selected_source]

st.sidebar.write(f"Filtered rows: {df_filtered.shape[0]}")

# Main layout: top row - metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total papers (filtered)", f"{df_filtered.shape[0]}")
col2.metric("Unique journals", f"{df_filtered['journal'].nunique()}")
col3.metric("Avg. abstract words", f"{int(df_filtered['abstract_word_count'].mean() or 0)}")

# Plot: publications by year
st.subheader("Publications by Year")
year_counts = df_filtered['year'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8,4))
year_counts.plot(kind='bar', ax=ax)
ax.set_xlabel('Year')
ax.set_ylabel('Paper count')
st.pyplot(fig)

# Top journals bar chart (based on filtered dataset)
st.subheader("Top Journals (filtered)")
top_j = df_filtered['journal'].value_counts().head(15)
fig2, ax2 = plt.subplots(figsize=(8,4))
sns.barplot(x=top_j.values, y=top_j.index, ax=ax2)
ax2.set_xlabel('Count')
ax2.set_ylabel('Journal')
st.pyplot(fig2)

# Distribution of abstract word counts
st.subheader("Abstract Word Count Distribution")
fig3, ax3 = plt.subplots(figsize=(8,4))
ax3.hist(df_filtered['abstract_word_count'].clip(upper=2000), bins=40)
ax3.set_xlabel('Abstract word count (clipped at 2000)')
ax3.set_ylabel('Number of papers')
st.pyplot(fig3)

# Scatter: title vs abstract length
st.subheader("Title length vs Abstract length (scatter sample)")
sample = df_filtered.sample(n=min(3000, max(100, df_filtered.shape[0])), random_state=1)
fig4, ax4 = plt.subplots(figsize=(7,5))
ax4.scatter(sample['title_word_count'], sample['abstract_word_count'], alpha=0.3, s=8)
ax4.set_xlabel('Title word count')
ax4.set_ylabel('Abstract word count')
st.pyplot(fig4)

# Show sample rows
st.subheader("Sample of Papers (filtered)")
st.dataframe(df_filtered[['publish_time', 'year', 'title', 'journal', 'source', 'abstract_word_count']].head(200))

st.write("Notes: This Streamlit app is intentionally simple — it demonstrates interactive filtering and key visualizations.")
