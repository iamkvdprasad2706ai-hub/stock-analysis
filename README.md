# Stock Analysis Workspace

A Python-based stock analysis project for exploring Indian stock price history, moving averages, and return statistics across NSE and BSE.

## Features
- Pull historical market data with Yahoo Finance
- Select NSE or BSE when looking up a stock; explicit Yahoo Finance suffixes (`.NS` and `.BO`) are also supported
- Analyze recommendations with up to 15 years of available price history, including CAGR and drawdown context
- Rank market ideas by sector trend and show three eligible stocks per sector, with India/global market indicators and source-linked headlines
- View stock price and moving average trend lines
- Inspect daily return distribution and summary metrics
- Run as a Streamlit dashboard locally

## Quick start

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal. The market screener and institutional-flow panels use NSE-specific data sources. Market headlines provide context for human review; political and geopolitical risks are not automatically verified or scored.

## Project structure

- `app.py` – Streamlit dashboard
- `src/stock_analysis/data.py` – data loading helpers
- `src/stock_analysis/analysis.py` – calculation utilities
- `requirements.txt` – Python dependencies
