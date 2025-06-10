import pandas as pd
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from langdetect import detect, LangDetectException

# Download VADER lexicon if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def detect_language(text):
    """Detect the language of the text."""
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'

def clean_for_sentiment(text):
    """Clean text for sentiment analysis."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters but keep punctuation for sentiment analysis
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text.strip()

def get_vader_sentiment(text):
    """Get sentiment using VADER."""
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    
    if sentiment['compound'] >= 0.05:
        return 'positive', sentiment['compound']
    elif sentiment['compound'] <= -0.05:
        return 'negative', sentiment['compound']
    else:
        return 'neutral', sentiment['compound']

def get_textblob_sentiment(text):
    """Get sentiment using TextBlob."""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        return 'positive', polarity
    elif polarity < -0.1:
        return 'negative', polarity
    else:
        return 'neutral', polarity

def get_transformer_sentiment(text, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """Get sentiment using transformer model."""
    try:
        # Initialize the pipeline
        sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
        
        # Truncate text if too long (most transformer models have a limit)
        if len(text.split()) > 500:
            text = ' '.join(text.split()[:500])
            
        result = sentiment_pipeline(text)[0]
        label = result['label'].lower()
        score = result['score']
        
        # Map LABEL_0/LABEL_1 to negative/positive if needed
        if label == 'label_0':
            return 'negative', score
        elif label == 'label_1':
            return 'positive', score
        else:
            return label, score
    except Exception as e:
        print(f"Error with transformer model: {e}")
        # Fall back to VADER if transformer fails
        return get_vader_sentiment(text)

def analyze_sentiment(df, method='transformer'):
    """
    Analyze sentiment for all reviews in the dataframe.
    
    Args:
        df: DataFrame with reviews
        method: 'transformer', 'vader', 'textblob', or 'ensemble'
        
    Returns:
        DataFrame with sentiment analysis results
    """
    result_df = df.copy()
    
    # Add language detection
    result_df['language'] = result_df['review_text'].apply(
        lambda x: detect_language(str(x)) if pd.notnull(x) else 'unknown'
    )
    
    # Clean text for sentiment analysis
    result_df['cleaned_text'] = result_df['review_text'].apply(clean_for_sentiment)
    
    # Apply sentiment analysis based on method
    if method == 'transformer':
        result_df[['sentiment_label', 'sentiment_score']] = result_df['cleaned_text'].apply(
            lambda x: pd.Series(get_transformer_sentiment(x) if x else ('neutral', 0.0))
        )
    
    elif method == 'vader':
        result_df[['sentiment_label', 'sentiment_score']] = result_df['cleaned_text'].apply(
            lambda x: pd.Series(get_vader_sentiment(x) if x else ('neutral', 0.0))
        )
    
    elif method == 'textblob':
        result_df[['sentiment_label', 'sentiment_score']] = result_df['cleaned_text'].apply(
            lambda x: pd.Series(get_textblob_sentiment(x) if x else ('neutral', 0.0))
        )
    
    elif method == 'ensemble':
        # For non-English text, use VADER (more robust for short texts)
        # For English text, use transformer
        result_df[['sentiment_label', 'sentiment_score']] = result_df.apply(
            lambda row: pd.Series(
                get_transformer_sentiment(row['cleaned_text']) 
                if row['language'] == 'en' and row['cleaned_text']
                else get_vader_sentiment(row['cleaned_text'] if row['cleaned_text'] else '')
            ), 
            axis=1
        )
    
    # Drop temporary columns
    result_df.drop('cleaned_text', axis=1, inplace=True)
    
    return result_df

def aggregate_sentiment_by_bank(df):
    """Aggregate sentiment scores by bank."""
    agg_df = df.groupby('bank').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'sentiment_label': lambda x: x.value_counts().to_dict()
    })
    
    # Flatten the multi-level columns
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    
    return agg_df

def aggregate_sentiment_by_bank_and_rating(df):
    """Aggregate sentiment scores by bank and rating."""
    agg_df = df.groupby(['bank', 'rating']).agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'sentiment_label': lambda x: x.value_counts().to_dict()
    })
    
    # Flatten the multi-level columns
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    
    return agg_df