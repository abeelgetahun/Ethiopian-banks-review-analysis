import pandas as pd
import os
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from src.sentiment import analyze_sentiment, aggregate_sentiment_by_bank, aggregate_sentiment_by_bank_and_rating
from src.thematic import (extract_keywords_tfidf, identify_themes, 
                         topic_modeling, assign_themes_to_reviews)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(input_path, output_dir):
    """
    Run the complete sentiment and thematic analysis pipeline.
    
    Args:
        input_path: Path to the processed reviews CSV
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Sentiment Analysis
    logger.info("Performing sentiment analysis...")
    sentiment_df = analyze_sentiment(df, method='ensemble')
    
    # Save sentiment results
    sentiment_path = os.path.join(output_dir, 'sentiment_results.csv')
    sentiment_df.to_csv(sentiment_path, index=False)
    logger.info(f"Sentiment results saved to {sentiment_path}")
    
    # 2. Sentiment Aggregation
    logger.info("Aggregating sentiment by bank...")
    bank_sentiment = aggregate_sentiment_by_bank(sentiment_df)
    bank_sentiment_path = os.path.join(output_dir, 'bank_sentiment_summary.csv')
    bank_sentiment.to_csv(bank_sentiment_path)
    
    logger.info("Aggregating sentiment by bank and rating...")
    bank_rating_sentiment = aggregate_sentiment_by_bank_and_rating(sentiment_df)
    bank_rating_path = os.path.join(output_dir, 'bank_rating_sentiment_summary.csv')
    bank_rating_sentiment.to_csv(bank_rating_path)
    
    # 3. Keyword Extraction
    logger.info("Extracting keywords using TF-IDF...")
    bank_keywords = extract_keywords_tfidf(df, n_keywords=30, ngram_range=(1, 3))
    
    # Save keywords
    keyword_path = os.path.join(output_dir, 'bank_keywords.csv')
    keyword_rows = []
    for bank, keywords in bank_keywords.items():
        for keyword, score in keywords:
            keyword_rows.append({
                'bank': bank,
                'keyword': keyword,
                'tfidf_score': score
            })
    keyword_df = pd.DataFrame(keyword_rows)
    keyword_df.to_csv(keyword_path, index=False)
    logger.info(f"Keywords saved to {keyword_path}")
    
    # 4. Theme Identification
    logger.info("Identifying themes...")
    bank_themes = identify_themes(bank_keywords, num_themes=5)
    
    # Save themes
    theme_path = os.path.join(output_dir, 'bank_themes.csv')
    theme_rows = []
    for bank, themes in bank_themes.items():
        for theme, score in themes:
            theme_rows.append({
                'bank': bank,
                'theme': theme,
                'theme_score': score
            })
    theme_df = pd.DataFrame(theme_rows)
    theme_df.to_csv(theme_path, index=False)
    logger.info(f"Themes saved to {theme_path}")
    
    # 5. Topic Modeling (optional)
    logger.info("Performing topic modeling...")
    topics = topic_modeling(df, n_topics=5, method='nmf')
    
    # Save topics
    topic_path = os.path.join(output_dir, 'topics.csv')
    topic_rows = []
    for topic, keywords in topics.items():
        for keyword in keywords:
            topic_rows.append({
                'topic': topic,
                'keyword': keyword
            })
    topic_df = pd.DataFrame(topic_rows)
    topic_df.to_csv(topic_path, index=False)
    logger.info(f"Topics saved to {topic_path}")
    
    # 6. Assign themes to reviews
    logger.info("Assigning themes to reviews...")
    final_df = assign_themes_to_reviews(sentiment_df, bank_themes)
    
    # Save final results
    final_path = os.path.join(output_dir, 'final_results.csv')
    final_df.to_csv(final_path, index=False)
    logger.info(f"Final results saved to {final_path}")
    
    return final_df

if __name__ == "__main__":
    # Define paths aligned with preprocessor outputs
    input_path = "data/processed/bank_reviews_processed.csv"
    output_dir = "data/task2"

    # Run pipeline
    run_pipeline(input_path, output_dir)