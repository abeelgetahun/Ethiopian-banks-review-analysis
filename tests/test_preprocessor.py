import pandas as pd
from src.preprocessor import preprocess_reviews

def test_preprocess_reviews_basic():
    df = pd.DataFrame({
        'review_text': ['Good app!', 'Bad app...', None],
        'rating': [5, 1, 3],
        'date': ['2025-06-01', '2025-06-02', '2025-06-03'],
        'bank': ['CBE', 'CBE', 'BOA'],
        'source': ['play', 'play', 'play']
    })

    processed, report = preprocess_reviews(df)

    assert 'review_id' in processed.columns
    assert processed['review_text'].isna().sum() == 0
    assert report['total_reviews'] == len(processed)
