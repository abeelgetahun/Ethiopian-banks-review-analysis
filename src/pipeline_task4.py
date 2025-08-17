# src/pipeline_task4.py

import os
import sys
from pathlib import Path
import logging
import json
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from src.insights import generate_insights_report
from src.visualization import create_visualizations

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

def generate_markdown_report(insights, output_path):
    """
    Generate a final report in markdown format.
    
    Args:
        insights: Dictionary with insights
        output_path: Path to save the report
    """
    logger.info(f"Generating markdown report at {output_path}")
    
    # Create the report content
    report = [
        "# Ethiopian Mobile Banking Apps: User Review Analysis",
        "\n## Executive Summary",
        "\nThis report presents an analysis of user reviews for major Ethiopian mobile banking applications.",
        "We analyzed sentiment, identified key themes, and extracted insights to understand customer satisfaction drivers and pain points.",
        "Based on our findings, we provide recommendations for application improvements.",
        "\n## 1. Methodology",
        "\nWe collected and analyzed user reviews from Google Play Store for the following banks:",
        "- Commercial Bank of Ethiopia",
        "- Bank of Abyssinia",
        "- Dashen Bank",
        "\nThe analysis pipeline included:",
        "1. Data collection and preprocessing",
        "2. Sentiment analysis using ensemble methods",
        "3. Thematic analysis to identify key topics and issues",
        "4. Insights extraction and visualization",
        "\n## 2. Key Findings",
        "\n### 2.1 Sentiment Analysis",
        "\n![Sentiment Distribution](figures/sentiment_distribution.png)",
        "\nThe chart above shows the distribution of sentiment across banks.",
        "\n### 2.2 Rating Distribution",
        "\n![Rating Distribution](figures/rating_distribution.png)",
        "\nThis visualization shows how ratings are distributed across different banks.",
        "\n### 2.3 Key Themes",
        "\n![Themes Heatmap](figures/themes_heatmap.png)",
        "\nThe heatmap shows the prominence of various themes in user reviews for each bank.",
        "\n### 2.4 Sentiment by Theme",
        "\n![Sentiment by Theme](figures/sentiment_by_theme.png)",
        "\nThis chart shows how sentiment varies across different themes for each bank.",
        "\n## 3. Bank-Specific Insights",
    ]
    
    # Add bank-specific insights
    for bank, bank_insights in insights["drivers_and_pain_points"].items():
        bank_section = [
            f"\n### 3.{list(insights['drivers_and_pain_points'].keys()).index(bank) + 1} {bank}",
            "\n#### Satisfaction Drivers:"
        ]
        
        # Add drivers
        for i, driver in enumerate(bank_insights["drivers"]):
            if driver in bank_insights["driver_examples"]:
                bank_section.append(f"\n- **{driver}**: Example: \"{bank_insights['driver_examples'][driver]}\"")
            else:
                bank_section.append(f"\n- **{driver}**")
        
        bank_section.append("\n#### Pain Points:")
        
        # Add pain points
        for i, pain_point in enumerate(bank_insights["pain_points"]):
            if pain_point in bank_insights["pain_point_examples"]:
                bank_section.append(f"\n- **{pain_point}**: Example: \"{bank_insights['pain_point_examples'][pain_point]}\"")
            else:
                bank_section.append(f"\n- **{pain_point}**")
        
        # Add word cloud for the bank
        bank_name = bank.replace(" ", "_").lower()
        bank_section.append(f"\n![{bank} Word Cloud](figures/{bank_name}_wordcloud.png)")
        
        report.extend(bank_section)
    
    # Add bank comparison section
    report.extend([
        "\n## 4. Bank Comparison",
        "\nThe table below summarizes key metrics for each bank:"
    ])
    
    # Create comparison table
    report.append("\n| Bank | Avg. Rating | Avg. Sentiment | Positive % | Review Count |")
    report.append("| ---- | ----------- | -------------- | ---------- | ------------ |")
    
    for metric in insights["bank_comparison"]["metrics"]:
        report.append(f"| {metric['bank']} | {metric['rating_mean']:.2f} | {metric['sentiment_score_mean']:.2f} | {metric.get('positive_percentage', 0):.1f}% | {metric['sentiment_score_count']} |")
    
    # Add recommendations section
    report.extend([
        "\n## 5. Recommendations",
        "\nBased on our analysis, we recommend the following improvements:"
    ])
    
    for bank, bank_recs in insights["recommendations"].items():
        report.append(f"\n### 5.{list(insights['recommendations'].keys()).index(bank) + 1} {bank}")
        
        for rec in bank_recs:
            report.append(f"\n- **{rec['area']}**: {rec['recommendation']} (Priority: {rec['priority']})")
    
    # Add limitations and biases section
    report.extend([
        "\n## 6. Limitations and Potential Biases",
        "\nIt's important to acknowledge potential biases in our analysis:"
    ])
    
    for bias in insights["biases"]:
        report.append(f"\n- **{bias['type']}**: {bias['description']}")
    
    # Add conclusion
    report.extend([
        "\n## 7. Conclusion",
        "\nOur analysis provides valuable insights into user experiences with Ethiopian mobile banking applications.",
        "By addressing the identified pain points and building on existing strengths,",
        "banks can improve user satisfaction and drive adoption of their mobile applications.",
        "\nPriority areas for improvement across all banks include:",
        "\n1. Enhancing app stability and reliability",
        "\n2. Improving transaction performance and success rates",
        "\n3. Streamlining the login and authentication process",
        "\n4. Providing better customer support through multiple channels",
        "\n5. Optimizing the user interface for better usability"
    ])
    
    # Write the report to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(report))
    
    logger.info(f"Report generated successfully at {output_path}")

def run_pipeline(data_dir, output_dir):
    """
    Run the complete insights and visualization pipeline.
    
    Args:
        data_dir: Path to the data directory
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    # Load analysis results
    logger.info("Loading analysis results")
    data = load_analysis_results(data_dir)
    
    # Generate insights
    logger.info("Generating insights")
    insights = generate_insights_report(data_dir)
    
    # Create visualizations
    logger.info("Creating visualizations")
    create_visualizations(data, os.path.join(output_dir, "figures"))
    
    # Generate final report
    logger.info("Generating final report")
    generate_markdown_report(insights, os.path.join(output_dir, "final_report.md"))
    
    # Save insights as JSON for potential future use
    with open(os.path.join(output_dir, "insights.json"), "w") as f:
        json.dump(insights, f, indent=4)
    
    logger.info(f"Pipeline completed successfully. Results saved to {output_dir}")
    
    return insights

if __name__ == "__main__":
    # Define paths
    data_dir = "data"
    output_dir = "reports"
    
    # Run pipeline
    run_pipeline(data_dir, output_dir)