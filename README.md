# Bürgerräte – Empfehlungen Dashboard

A Streamlit web application for visualizing and exploring recommendations from citizen assemblies (Bürgerräte) in Germany.

## Features

- Interactive data filtering and visualization
- Full-text search capabilities
- AI-powered chat interface for exploring recommendations with **rate limiting**
  - 20 requests per hour per user
  - 3 requests per minute per user
- Detailed views of assembly recommendations
- Export functionality

## Deployment

This app is designed to run on Streamlit Community Cloud.

### Requirements

- Python 3.11 (specified in `.python-version`)
- Dependencies listed in `requirements.txt`

### Configuration

1. **Streamlit Cloud Secrets**: Add your OpenAI API key to the Streamlit Cloud secrets:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key-here"
   OPENAI_MODEL = "gpt-4o-mini"  # optional, defaults to gpt-4o-mini
   ```

2. **Local Development**: Set environment variables:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export OPENAI_MODEL="gpt-4o-mini"
   ```

### Data

The app expects a `data/recommendations.ndjson` file containing the citizen assembly recommendations in JSONL format.

## Recent Fixes

- Updated to Python 3.11 for better package compatibility
- Updated package versions for stable deployment
- Added robust error handling for OpenAI secrets
- Added proper deployment configuration files

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```
