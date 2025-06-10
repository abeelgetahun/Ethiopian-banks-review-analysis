# src/visualization.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from wordcloud import WordCloud
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_plotting_style():
    """Set consistent plotting style for all visualizations."""
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["figure.titlesize"] = 20
    
    # Use a color palette suitable for bank data visualization
    colors = ["#2C3E50", "#E74C3C", "#3498DB", "#F39C12", "#1ABC9C"]
    sns.set_palette(sns.color_palette(colors))

def plot_sentiment_distribution(data, output_dir):
    """
    Create a bar plot showing sentiment distribution across banks.
    
    Args:
        data: Dictionary containing analysis results
        output_dir: Directory to save the plot
    """
    logger.info("Creating sentiment distribution plot")
    set_plotting_style()
    
    final_df = data["final"]
    
    # Calculate sentiment counts by bank
    sentiment_counts = final_df.groupby(["bank", "sentiment_label"]).size().unstack(fill_value=0)
    
    # Calculate percentages
    sentiment_percentages = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sentiment_percentages.plot(kind="bar", stacked=True, ax=ax, 
                              color=["#E74C3C", "#F39C12", "#2ECC71"])
    
    # Add labels and title
    plt.title("Sentiment Distribution by Bank", fontweight="bold")
    plt.xlabel("Bank")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45)
    plt.legend(title="Sentiment")
    
    # Add percentages on bars
    for i, bank in enumerate(sentiment_percentages.index):
        cumulative = 0
        for sentiment in sentiment_percentages.columns:
            value = sentiment_percentages.loc[bank, sentiment]
            if value > 5:  # Only show percentage if it's significant
                plt.text(i, cumulative + value/2, f"{value:.1f}%", 
                       ha="center", va="center", fontweight="bold", color="white")
            cumulative += value
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, "sentiment_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Sentiment distribution plot saved to {output_path}")

def plot_rating_distribution(data, output_dir):
    """
    Create a grouped bar plot showing rating distribution across banks.
    
    Args:
        data: Dictionary containing analysis results
        output_dir: Directory to save the plot
    """
    logger.info("Creating rating distribution plot")
    set_plotting_style()
    
    final_df = data["final"]
    
    # Calculate rating counts by bank
    rating_counts = pd.crosstab(final_df["bank"], final_df["rating"])
    
    # Calculate percentages
    rating_percentages = rating_counts.div(rating_counts.sum(axis=1), axis=0) * 100
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    rating_percentages.plot(kind="bar", ax=ax)
    
    # Add labels and title
    plt.title("Rating Distribution by Bank", fontweight="bold")
    plt.xlabel("Bank")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45)
    plt.legend(title="Rating (stars)")
    
    # Add grid for better readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, "rating_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Rating distribution plot saved to {output_path}")

def plot_themes_heatmap(data, output_dir):
    """
    Create a heatmap showing the prominence of themes across banks.
    
    Args:
        data: Dictionary containing analysis results
        output_dir: Directory to save the plot
    """
    logger.info("Creating themes heatmap")
    set_plotting_style()
    
    themes_df = data["themes"]
    
    # Pivot the data to create a bank x theme matrix
    pivot_df = themes_df.pivot(index="bank", columns="theme", values="theme_score")
    
    # Fill NaN with 0
    pivot_df = pivot_df.fillna(0)
    
    # Select top themes for better visualization
    top_themes = pivot_df.sum().nlargest(8).index
    pivot_df = pivot_df[top_themes]
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5, ax=ax)
    
    # Add labels and title
    plt.title("Theme Prominence Across Banks", fontweight="bold")
    plt.ylabel("Bank")
    plt.xlabel("Theme")
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, "themes_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Themes heatmap saved to {output_path}")

def create_word_clouds(data, output_dir):
    """
    Create word clouds for each bank based on keywords.
    
    Args:
        data: Dictionary containing analysis results
        output_dir: Directory to save the plots
    """
    logger.info("Creating word clouds")
    
    keywords_df = data["keywords"]
    
    # Generate word cloud for each bank
    for bank in keywords_df["bank"].unique():
        logger.info(f"Creating word cloud for {bank}")
        
        # Extract keywords and scores for this bank
        bank_keywords = keywords_df[keywords_df["bank"] == bank]
        
        # Create a dictionary of word:score for the word cloud
        word_scores = dict(zip(bank_keywords["keyword"], bank_keywords["tfidf_score"]))
        
        # Generate the word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color="white", 
            max_words=100,
            colormap="viridis",
            contour_width=1, 
            contour_color="steelblue"
        ).generate_from_frequencies(word_scores)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Key Terms in {bank} Reviews", fontsize=18, fontweight="bold")
        plt.tight_layout(pad=0)
        
        # Save the plot
        bank_name = bank.replace(" ", "_").lower()
        output_path = os.path.join(output_dir, f"{bank_name}_wordcloud.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Word cloud for {bank} saved to {output_path}")

def plot_sentiment_by_theme(data, output_dir):
    """
    Create a plot showing average sentiment score by theme for each bank.
    
    Args:
        data: Dictionary containing analysis results
        output_dir: Directory to save the plot
    """
    logger.info("Creating sentiment by theme plot")
    set_plotting_style()
    
    final_df = data["final"]
    
    # Expand the themes column into separate rows
    theme_rows = []
    
    for _, row in final_df.iterrows():
        if pd.isnull(row["identified_themes"]) or row["identified_themes"] == "Other":
            continue
        
        themes = row["identified_themes"].split("; ")
        for theme in themes:
            if theme and theme != "Other":
                theme_row = row.copy()
                theme_row["theme"] = theme
                theme_rows.append(theme_row)
    
    if not theme_rows:
        logger.warning("No theme data available for sentiment by theme plot")
        return
    
    theme_df = pd.DataFrame(theme_rows)
    
    # Calculate average sentiment score by bank and theme
    sentiment_by_theme = theme_df.groupby(["bank", "theme"])["sentiment_score"].mean().reset_index()
    
    # Only keep themes with significant data across banks
    theme_counts = theme_df["theme"].value_counts()
    significant_themes = theme_counts[theme_counts > 10].index
    sentiment_by_theme = sentiment_by_theme[sentiment_by_theme["theme"].isin(significant_themes)]
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    # Use different markers for different banks
    markers = ["o", "s", "D", "^", "*"]
    banks = sentiment_by_theme["bank"].unique()
    
    for i, bank in enumerate(banks):
        bank_data = sentiment_by_theme[sentiment_by_theme["bank"] == bank]
        plt.scatter(bank_data["theme"], bank_data["sentiment_score"], 
                   label=bank, marker=markers[i % len(markers)], s=100, alpha=0.7)
    
    # Add labels and title
    plt.title("Average Sentiment Score by Theme Across Banks", fontweight="bold")
    plt.xlabel("Theme")
    plt.ylabel("Average Sentiment Score")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Add a horizontal line at 0 (neutral sentiment)
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    
    # Add legend
    plt.legend(title="Bank")
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, "sentiment_by_theme.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Sentiment by theme plot saved to {output_path}")

def create_visualizations(data, output_dir="reports/figures"):
    """
    Create all visualizations for the final report.
    
    Args:
        data: Dictionary containing analysis results
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    plot_sentiment_distribution(data, output_dir)
    plot_rating_distribution(data, output_dir)
    plot_themes_heatmap(data, output_dir)
    create_word_clouds(data, output_dir)
    plot_sentiment_by_theme(data, output_dir)
    
    logger.info(f"All visualizations saved to {output_dir}")