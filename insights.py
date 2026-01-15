import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import re
import spacy

try:
    nlp = spacy.load("es_core_news_sm")
except:
    nlp = None # Handle gracefully in app if not loaded yet

def get_top_ngrams(texts, n=2, top_k=10):
    """
    Returns the top K n-grams from a list of texts.
    """
    if texts is None or (hasattr(texts, 'empty') and texts.empty) or (not hasattr(texts, 'empty') and not texts):
        return []
    
    # Use CountVectorizer for n-gram extraction (Bigrams & Trigrams) to capture context like "comida rica" or "muy caro"
    try:
        # We assume texts are already preprocessed/cleaned
        vec = CountVectorizer(ngram_range=(2, 3), stop_words=None, max_features=1000, min_df=1)
        # Note: If we had a Spanish stopword list here it would be better, 
        # but preprocessing usually handles stopwords.
        
        bag_of_words = vec.fit_transform(texts)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:top_k]
    except ValueError:
        return []

def extract_opportunities(texts):
    """
    Analyzes negative reviews to find opportunities for improvement using regex patterns.
    """
    patterns = [
        r"falta\s+\w+",
        r"deber[íi]an\s+\w+",
        r"podr[íi]an\s+\w+",
        r"mejorar\s+\w+",
        r"tard[óo]\s+\w+",
        r"demora\s+\w+",
        r"malo\s+\w+",
        r"pésimo\s+\w+",
        r"sucio",
        r"frío",
        r"caro"
    ]
    
    opportunities = []
    
    for text in texts:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                opportunities.append({
                    'phrase': match.group(0),
                    'context': text,
                    'keyword': pattern.split(r'\s')[0].replace(r"[íi]", "i")
                })
    
    # Aggregate by phrase or keyword
    return pd.DataFrame(opportunities) if opportunities else pd.DataFrame(columns=['phrase', 'context', 'keyword'])

def calculate_global_score(df, col_name='sentiment_global'):
    """
    Calculates the global score (0-100) based on sentiment ratios.
    Score = 50 + 50 * (pos_ratio - neg_ratio)
    """
    if df.empty:
        return 0, "Sin datos"
    
    # Fallback if col not found but sentiment_label exists
    if col_name not in df.columns and 'sentiment_label' in df.columns:
        col_name = 'sentiment_label'
        
    counts = df[col_name].value_counts(normalize=True)
    pos_ratio = counts.get('POS', 0.0)
    neg_ratio = counts.get('NEG', 0.0)
    
    score = 50 + 50 * (pos_ratio - neg_ratio)
    score = max(0, min(100, score)) # Clamp
    
    if score >= 60:
        rating = "Mayormente positivo"
    elif score >= 40:
        rating = "Mixto"
    else:
        rating = "Mayormente negativo"
        
    return round(score, 1), rating
