# Frameworks_Assignment - CORD-19 Metadata Explorer

## Description
Analysis and Streamlit app for the CORD-19 `metadata.csv` file.

## How to run
1. Download `metadata.csv` from Kaggle (CORD-19) and place it in repo root.
2. Install dependencies:
   pip install -r requirements.txt
3. Run the analysis script:
   python cord19_analysis.py metadata.csv
   This will create visualizations and `cord19_cleaned_metadata.csv`.
4. Launch the Streamlit app:
   streamlit run app_streamlit.py

## Notes
- The wordcloud is optional and requires `wordcloud` package.
- If the original `metadata.csv` is too large, consider sampling it first.
