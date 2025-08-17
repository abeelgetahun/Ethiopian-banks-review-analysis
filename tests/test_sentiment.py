import pandas as pd
from src.sentiment import analyze_sentiment

def test_analyze_sentiment_ensemble():
    df = pd.DataFrame({
        'review_text': ['I love this app', 'This app keeps crashing', 'It is okay'],
        'rating': [5, 1, 3],
        'date': ['2025-06-01', '2025-06-02', '2025-06-03'],
        'bank': ['CBE', 'CBE', 'BOA'],
        'source': ['play', 'play', 'play']
    })

    out = analyze_sentiment(df, method='ensemble')
    assert {'sentiment_label', 'sentiment_score', 'language'}.issubset(out.columns)
    assert len(out) == 3
