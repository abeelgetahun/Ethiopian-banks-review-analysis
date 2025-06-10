# src/insights.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_analysis_results(data_dir):
    """
    Load the results from previous analysis steps.
    
    Args:
        data_dir: Path to the data directory
    
    Returns:
        Dictionary containing the loaded DataFrames
    """
    try:
        task2_dir = Path(data_dir) / "task2"
        
        # Load sentiment results
        sentiment_path = task2_dir / "sentiment_results.csv"
        logger.info(f"Loading sentiment results from {sentiment_path}")
        sentiment_df = pd.read_csv(sentiment_path)
        
        # Load theme results
        themes_path = task2_dir / "bank_themes.csv"
        logger.info(f"Loading themes from {themes_path}")
        themes_df = pd.read_csv(themes_path)
        
        # Load keywords
        keywords_path = task2_dir / "bank_keywords.csv"
        logger.info(f"Loading keywords from {keywords_path}")
        keywords_df = pd.read_csv(keywords_path)
        
        # Load final results with themes assigned to reviews
        final_path = task2_dir / "final_results.csv"
        logger.info(f"Loading final results from {final_path}")
        final_df = pd.read_csv(final_path)
        
        return {
            "sentiment": sentiment_df,
            "themes": themes_df,
            "keywords": keywords_df,
            "final": final_df
        }
    except Exception as e:
        logger.error(f"Error loading analysis results: {e}")
        raise

def identify_drivers_and_pain_points(data, top_n=3):
    """
    Identify key drivers of satisfaction and pain points for each bank.
    
    Args:
        data: Dictionary of analysis DataFrames
        top_n: Number of top drivers/pain points to identify
    
    Returns:
        Dictionary with drivers and pain points for each bank
    """
    final_df = data["final"]
    themes_df = data["themes"]
    
    insights = {}
    
    for bank in final_df["bank"].unique():
        bank_df = final_df[final_df["bank"] == bank]
        
        # Get positive reviews (rating >= 4)
        positive_df = bank_df[bank_df["rating"] >= 4]
        
        # Get negative reviews (rating <= 2)
        negative_df = bank_df[bank_df["rating"] <= 2]
        
        # Count themes in positive reviews
        positive_themes = []
        if not positive_df.empty:
            for themes_str in positive_df["identified_themes"].dropna():
                for theme in themes_str.split("; "):
                    if theme and theme != "Other":
                        positive_themes.append(theme)
        
        # Count themes in negative reviews
        negative_themes = []
        if not negative_df.empty:
            for themes_str in negative_df["identified_themes"].dropna():
                for theme in themes_str.split("; "):
                    if theme and theme != "Other":
                        negative_themes.append(theme)
        
        # Get top drivers (most frequent themes in positive reviews)
        pos_theme_counts = pd.Series(positive_themes).value_counts()
        drivers = pos_theme_counts.head(top_n).index.tolist() if not pos_theme_counts.empty else []
        
        # Get top pain points (most frequent themes in negative reviews)
        neg_theme_counts = pd.Series(negative_themes).value_counts()
        pain_points = neg_theme_counts.head(top_n).index.tolist() if not neg_theme_counts.empty else []
        
        # Get example reviews for each driver
        driver_examples = {}
        for driver in drivers:
            # Find reviews mentioning this theme with high ratings
            examples = positive_df[positive_df["identified_themes"].str.contains(driver, na=False)]
            if not examples.empty:
                # Get a representative example
                example = examples.iloc[0]["review_text"]
                driver_examples[driver] = example
        
        # Get example reviews for each pain point
        pain_point_examples = {}
        for pain_point in pain_points:
            # Find reviews mentioning this theme with low ratings
            examples = negative_df[negative_df["identified_themes"].str.contains(pain_point, na=False)]
            if not examples.empty:
                # Get a representative example
                example = examples.iloc[0]["review_text"]
                pain_point_examples[pain_point] = example
        
        insights[bank] = {
            "drivers": drivers,
            "pain_points": pain_points,
            "driver_examples": driver_examples,
            "pain_point_examples": pain_point_examples
        }
    
    return insights

def compare_banks(data):
    """
    Compare different banks based on sentiment scores and key themes.
    
    Args:
        data: Dictionary of analysis DataFrames
    
    Returns:
        Dictionary with comparison metrics
    """
    final_df = data["final"]
    
    comparison = {}
    
    # Calculate average sentiment and rating for each bank
    bank_metrics = final_df.groupby("bank").agg({
        "sentiment_score": ["mean", "count"],
        "rating": "mean"
    })
    
    # Flatten the multi-level columns
    bank_metrics.columns = ['_'.join(col).strip() for col in bank_metrics.columns.values]
    
    # Reset index for easier processing
    bank_metrics = bank_metrics.reset_index()
    
    # Calculate positive sentiment percentage
    sentiment_counts = final_df.groupby("bank")["sentiment_label"].value_counts().unstack().fillna(0)
    
    if "positive" in sentiment_counts.columns:
        bank_metrics["positive_percentage"] = sentiment_counts["positive"] / sentiment_counts.sum(axis=1) * 100
    else:
        bank_metrics["positive_percentage"] = 0
    
    # Get top themes for each bank
    bank_themes = {}
    for bank in final_df["bank"].unique():
        bank_df = final_df[final_df["bank"] == bank]
        themes = []
        for themes_str in bank_df["identified_themes"].dropna():
            for theme in themes_str.split("; "):
                if theme and theme != "Other":
                    themes.append(theme)
        
        # Get theme counts
        theme_counts = pd.Series(themes).value_counts()
        
        # Save top 5 themes
        bank_themes[bank] = theme_counts.head(5).to_dict()
    
    comparison["metrics"] = bank_metrics.to_dict(orient="records")
    comparison["themes"] = bank_themes
    
    return comparison

def generate_recommendations(insights):
    """
    Generate app improvement recommendations based on identified pain points.
    
    Args:
        insights: Dictionary with insights for each bank
    
    Returns:
        Dictionary with recommendations for each bank
    """
    recommendations = {}
    
    # Common recommendations based on typical banking app pain points
    common_recs = {
        "Account Access Issues": "Improve the login process with biometric authentication options and streamline the password reset process.",
        "Transaction Performance": "Optimize transaction processing speed and reduce failures by implementing better error handling and retry mechanisms.",
        "App Performance & Reliability": "Enhance app stability through rigorous testing and optimize performance for older devices and slower networks.",
        "User Interface & Experience": "Redesign the interface with a focus on usability, implementing a more intuitive navigation system and cleaner layout.",
        "Customer Support": "Integrate an in-app chat support feature and improve the responsiveness of customer service channels.",
        "Network & Connectivity": "Implement offline mode for basic features and optimize the app to work better in areas with poor connectivity.",
        "Features & Functionality": "Add budgeting tools and expense tracking features to provide more value to users.",
        "Security & Trust": "Enhance security with additional measures like transaction notifications and implement clear privacy policies."
    }
    
    for bank, bank_insights in insights.items():
        bank_recs = []
        
        # Generate specific recommendations based on pain points
        for pain_point in bank_insights["pain_points"]:
            # Use common recommendation if available
            for theme, recommendation in common_recs.items():
                if theme in pain_point:
                    bank_recs.append({
                        "area": pain_point,
                        "recommendation": recommendation,
                        "priority": "High"
                    })
                    break
            else:
                # If no specific recommendation is available, add a generic one
                bank_recs.append({
                    "area": pain_point,
                    "recommendation": f"Address issues related to {pain_point.lower()} based on user feedback.",
                    "priority": "Medium"
                })
        
        # Add some general recommendations if we don't have enough specific ones
        if len(bank_recs) < 2:
            bank_recs.append({
                "area": "User Experience",
                "recommendation": "Conduct usability testing to identify and address pain points in the user journey.",
                "priority": "Medium"
            })
            
            bank_recs.append({
                "area": "Feature Enhancement",
                "recommendation": "Add budgeting tools and expense categorization to provide more value to users.",
                "priority": "Medium"
            })
        
        recommendations[bank] = bank_recs
    
    return recommendations

def identify_biases(data):
    """
    Identify potential biases in the review data.
    
    Args:
        data: Dictionary of analysis DataFrames
    
    Returns:
        List of identified biases
    """
    final_df = data["final"]
    
    biases = []
    
    # Check rating distribution for negativity bias
    rating_counts = final_df["rating"].value_counts(normalize=True) * 100
    
    if rating_counts.get(1, 0) + rating_counts.get(2, 0) > 50:
        biases.append({
            "type": "Negativity Bias",
            "description": "More than 50% of reviews are negative (1-2 stars), which may not represent the average user experience. Users with negative experiences are more likely to leave reviews."
        })
    
    if rating_counts.get(5, 0) > 60:
        biases.append({
            "type": "Positivity Bias",
            "description": "More than 60% of reviews are 5-star ratings, which may indicate selection bias where only very satisfied users leave reviews."
        })
    
    # Check for recency bias
    final_df["date"] = pd.to_datetime(final_df["date"])
    recent_count = final_df[final_df["date"] > (final_df["date"].max() - pd.Timedelta(days=90))].shape[0]
    recent_percentage = (recent_count / final_df.shape[0]) * 100
    
    if recent_percentage > 70:
        biases.append({
            "type": "Recency Bias",
            "description": f"About {recent_percentage:.1f}% of reviews are from the last 3 months, which may overrepresent recent app changes and underrepresent long-term issues or benefits."
        })
    
    # Add general biases that are common in app reviews
    biases.extend([
        {
            "type": "Self-Selection Bias",
            "description": "Users who leave reviews typically have strong opinions (very positive or very negative), which may not represent the average user experience."
        },
        {
            "type": "Language Bias",
            "description": "Our analysis focuses primarily on reviews in English, which may not represent the full user base in Ethiopia where Amharic and other local languages are commonly used."
        },
        {
            "type": "Platform Bias",
            "description": "Reviews are collected only from Google Play Store, missing user opinions from iOS users or other platforms."
        }
    ])
    
    return biases

def generate_insights_report(data_dir="data"):
    """
    Generate comprehensive insights report.
    
    Args:
        data_dir: Path to the data directory
    
    Returns:
        Dictionary with all insights
    """
    # Load analysis results
    data = load_analysis_results(data_dir)
    
    # Extract insights
    drivers_and_pain_points = identify_drivers_and_pain_points(data)
    bank_comparison = compare_banks(data)
    recommendations = generate_recommendations(drivers_and_pain_points)
    biases = identify_biases(data)
    
    # Compile all insights
    insights_report = {
        "drivers_and_pain_points": drivers_and_pain_points,
        "bank_comparison": bank_comparison,
        "recommendations": recommendations,
        "biases": biases
    }
    
    return insights_report