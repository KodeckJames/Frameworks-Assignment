"""
cord19_analysis.py
A script to load, clean, analyze, and visualize the CORD-19 metadata.csv file.

Outputs:
 - cleaned_metadata.csv
 - PNG visualizations in the working directory
 - prints summary statistics to console

Usage:
 python cord19_analysis.py path/to/metadata.csv
If no path provided it will attempt to read ./metadata.csv
"""

import sys
import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Optional wordcloud (only used if installed)
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

def safe_read_csv(path):
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"Loaded file: {path} ({df.shape[0]} rows, {df.shape[1]} cols)")
        return df
    except FileNotFoundError:
        raise
    except Exception as e:
        print("Error reading CSV:", e)
        raise

def basic_explore(df):
    print("\n=== Basic Exploration ===")
    print("Shape:", df.shape)
    print("\nColumns and dtypes:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head().T)
    print("\nMissing values (top 20):")
    print(df.isnull().sum().sort_values(ascending=False).head(20))

def clean_prepare(df):
    """Cleans and adds useful columns:
    - publish_time -> datetime, year
    - abstract_word_count
    - title_word_count
    - source (keep existing or fill unknown)
    Returns cleaned df (a copy).
    """
    df = df.copy()
    # Ensure columns exist, otherwise create
    if 'publish_time' not in df.columns:
        df['publish_time'] = pd.NA

    # Convert publish_time to datetime where possible
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

    # Year
    df['year'] = df['publish_time'].dt.year

    # Abstract and title columns: ensure strings and compute word counts
    df['abstract'] = df.get('abstract', pd.NA).astype('string').fillna('')
    df['title'] = df.get('title', pd.NA).astype('string').fillna('')

    df['abstract_word_count'] = df['abstract'].apply(lambda x: 0 if pd.isna(x) or x=='' else len(str(x).split()))
    df['title_word_count'] = df['title'].apply(lambda x: 0 if pd.isna(x) or x=='' else len(str(x).split()))

    # Source column (some metadata variants use 'source_x' or 'source_y' or 'source')
    if 'source' not in df.columns:
        # try common alternatives
        for alt in ['source_x', 'source_y', 'source_x_y', 'source_x.1']:
            if alt in df.columns:
                df['source'] = df[alt]
                break
        else:
            df['source'] = 'unknown'

    # Fill missing journal info
    if 'journal' not in df.columns and 'journal_x' in df.columns:
        df['journal'] = df['journal_x']
    if 'journal' not in df.columns:
        df['journal'] = df.get('journal', pd.NA).astype('string').fillna('Unknown')

    # Drop rows that do not have title and no abstract (likely not useful)
    initial = df.shape[0]
    df = df[~((df['title'].str.strip() == '') & (df['abstract'].str.strip() == ''))]
    dropped = initial - df.shape[0]
    print(f"Dropped {dropped} rows with no title and no abstract")

    return df

def analyze_and_save(df, out_prefix='cord19'):
    """Performs analysis and saves visualizations"""
    # Save cleaned CSV
    cleaned_path = f"{out_prefix}_cleaned_metadata.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned dataset to {cleaned_path}")

    # Basic statistics
    print("\n=== Basic statistics for numeric columns ===")
    print(df[['abstract_word_count', 'title_word_count']].describe())

    # Publications by year
    year_counts = df['year'].value_counts().dropna().sort_index()
    print("\nPublications by year (sample):")
    print(year_counts.head(10))

    # Plot publications over time (year)
    plt.figure(figsize=(8,5))
    year_counts.plot(kind='bar')
    plt.title('Number of Publications by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')
    plt.tight_layout()
    plt.savefig('publications_by_year.png', dpi=150)
    print("Saved publications_by_year.png")

    # Top journals
    top_journals = df['journal'].fillna('Unknown').value_counts().head(20)
    plt.figure(figsize=(10,6))
    sns.barplot(y=top_journals.index, x=top_journals.values)
    plt.title('Top 20 Journals by Number of Papers')
    plt.xlabel('Paper Count')
    plt.ylabel('Journal')
    plt.tight_layout()
    plt.savefig('top_journals.png', dpi=150)
    print("Saved top_journals.png")

    # Source distribution
    top_sources = df['source'].fillna('unknown').value_counts().head(15)
    plt.figure(figsize=(8,5))
    sns.barplot(x=top_sources.index, y=top_sources.values)
    plt.xticks(rotation=45, ha='right')
    plt.title('Top Sources')
    plt.xlabel('Source')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('top_sources.png', dpi=150)
    print("Saved top_sources.png")

    # Histogram of abstract word counts
    plt.figure(figsize=(8,5))
    plt.hist(df['abstract_word_count'].clip(upper=2000), bins=40, edgecolor='k')  # clip to avoid huge tails
    plt.title('Distribution of Abstract Word Counts (clipped at 2000)')
    plt.xlabel('Abstract Word Count')
    plt.ylabel('Number of Papers')
    plt.tight_layout()
    plt.savefig('abstract_wordcount_hist.png', dpi=150)
    print("Saved abstract_wordcount_hist.png")

    # Scatter: title_word_count vs abstract_word_count (sample for speed)
    sample = df.sample(n=min(5000, df.shape[0]), random_state=42)
    plt.figure(figsize=(7,6))
    plt.scatter(sample['title_word_count'], sample['abstract_word_count'], alpha=0.3, s=10)
    plt.title('Title Length vs Abstract Length (sample)')
    plt.xlabel('Title Word Count')
    plt.ylabel('Abstract Word Count')
    plt.tight_layout()
    plt.savefig('title_vs_abstract_scatter.png', dpi=150)
    print("Saved title_vs_abstract_scatter.png")

    # Most frequent words in titles (simple approach)
    # Lowercase, remove very short words, basic tokenization
    def tokenize(text):
        if not text:
            return []
        tokens = str(text).lower().replace('\n',' ').split()
        tokens = [t.strip('.,;:()[]\'"') for t in tokens if len(t.strip('.,;:()[]\'"')) > 2]
        return tokens

    title_tokens = Counter()
    for t in df['title'].dropna().astype(str):
        title_tokens.update(tokenize(t))

    most_common = title_tokens.most_common(30)
    words, counts = zip(*most_common)

    plt.figure(figsize=(10,6))
    sns.barplot(x=list(counts), y=list(words))
    plt.title('Top 30 Most Frequent Words in Titles')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.tight_layout()
    plt.savefig('title_word_freq.png', dpi=150)
    print("Saved title_word_freq.png")

    # Optional: wordcloud of titles if library available
    if WORDCLOUD_AVAILABLE:
        all_titles = " ".join(df['title'].dropna().astype(str))
        wc = WordCloud(width=1200, height=600, background_color='white', max_words=200).generate(all_titles)
        plt.figure(figsize=(12,6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('title_wordcloud.png', dpi=150)
        plt.close()
        print("Saved title_wordcloud.png (wordcloud)")

    # Save some summary CSVs
    top_journals_df = top_journals.reset_index().rename(columns={'index':'journal', 'journal':'count'})
    top_journals_df.to_csv('top_journals.csv', index=False)
    print("Saved top_journals.csv")

    return {
        'year_counts': year_counts,
        'top_journals': top_journals,
        'top_sources': top_sources,
        'most_common_title_words': most_common
    }

def main():
    # Get path from argv or default
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = 'metadata.csv'
    if not os.path.exists(path):
        print(f"File not found: {path}. Please place metadata.csv in this directory or pass path as argument.")
        return

    try:
        df = safe_read_csv(path)
    except FileNotFoundError:
        print("metadata.csv not found.")
        return
    except Exception as e:
        print("Failed to load CSV:", e)
        return

    basic_explore(df)
    df_clean = clean_prepare(df)
    results = analyze_and_save(df_clean)

    print("\n=== Quick Findings (console) ===")
    print("Top journals:")
    print(results['top_journals'].head(10))
    print("\nTop title words:")
    print(results['most_common_title_words'][:20])

if __name__ == '__main__':
    main()
