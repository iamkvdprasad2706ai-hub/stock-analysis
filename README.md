# Stock Analysis Workspace

A small Python-based stock analysis project for exploring price history, moving averages, and basic return statistics.

## Features
- Pull historical market data with Yahoo Finance
- View stock price and moving average trend lines
- Inspect daily return distribution and summary metrics
- Run as a Streamlit dashboard locally

## Quick start

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Project structure

- `app.py` – Streamlit dashboard
- `src/stock_analysis/data.py` – data loading helpers
- `src/stock_analysis/analysis.py` – calculation utilities
- `requirements.txt` – Python dependencies
