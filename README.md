# Sentimentalyst

Sentimentalyst is a Streamlit-based web application for conducting sentiment analysis on Spanish reviews. It leverages both a baseline Perceptron model and a Deep Learning MLP model to classify reviews and provide actionable business insights.

## Project Structure

- `app.py`: Main Streamlit application.
- `preprocess.py`: Text cleaning and preprocessing logic.
- `features.py`: Feature engineering (CountVectorizer, TF-IDF).
- `models_baseline.py`: Perceptron model definition.
- `models_deep.py`: PyTorch MLP model definition.
- `train_baseline.py`: Script to train the baseline model.
- `train_deep.py`: Script to train the deep learning model.
- `insights.py`: Logic for extracting insights (strengths, weaknesses).
- `data/`: Directory containing CSV datasets.
- `models/`: Directory for saving trained models.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <repo_url>
    cd NLP_EntregaFinal
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    python -m spacy download es_core_news_sm
    ```

## Training Models

Before running the app, you must train the models:

1.  **Train Baseline Model:**
    ```bash
    python train_baseline.py
    ```

2.  **Train Deep Learning Model:**
    ```bash
    python train_deep.py
    ```

## Running the App

```bash
streamlit run app.py
```

## Input Format

The application expects a CSV file with at least the following columns:
- `place_name`: Name of the establishment.
- `review_text`: The text of the review.
