# UChicago MS-ADS Program Q&A

This Streamlit application provides a question-answering interface for the University of Chicago's Master of Science in Applied Data Science program. It uses RAG (Retrieval-Augmented Generation) to provide accurate answers based on the program's official materials.

## Features

- Interactive Q&A interface
- Real-time answers using GPT-3.5
- Source citations for transparency
- FAISS vector store for efficient retrieval
- Cohere reranking for improved relevance

## Requirements

- Python 3.8+
- OpenAI API key
- Cohere API key

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <repo-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and go to the URL shown in the terminal (usually http://localhost:8501)

3. Enter your OpenAI and Cohere API keys in the sidebar

4. Start asking questions about the MS-ADS program!

## Data Sources

The application uses pre-scraped data from the official UChicago MS-ADS program website. The data is stored in a FAISS vector store for efficient retrieval.

## Note

This is an unofficial tool and should not be considered a replacement for official program information. Please refer to the [official program website](https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/) for the most up-to-date information. 