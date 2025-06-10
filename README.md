# 🏦 Ethiopian Banks Mobile App Review Analysis

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A professional data science project for collecting, processing, and analyzing user reviews of major Ethiopian banks' mobile applications from the Google Play Store. This repository provides insights into user satisfaction, highlights areas for app improvement, and presents thematic/sentiment analysis for actionable recommendations.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Data Workflow](#data-workflow)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Key Insights](#key-insights)
- [Contributing](#contributing)
- [License](#license)

---

## 🚀 Project Overview

The **Ethiopian Banks Mobile App Review Analysis** project automates the collection and analysis of user reviews for leading Ethiopian banks' mobile apps. It leverages web scraping, thorough preprocessing, exploratory data analysis (EDA), and advanced thematic/sentiment analytics to help banks and fintech experts understand customer feedback and improve digital banking experiences.

---

## ✨ Features

- **Automated Scraping:** Efficiently collects reviews for multiple banks from the Google Play Store.
- **Data Preprocessing:** Cleans and structures raw reviews for reliability and consistency.
- **Exploratory Data Analysis:** Visualizes review statistics, ratings, and common issues.
- **Thematic & Sentiment Analysis:** Detects key complaint/reward themes and user sentiments.
- **Actionable Reporting:** Generates quality and thematic reports to guide bank app improvements.

---

## 🔄 Data Workflow

1. **Scraping Reviews**  
   (`src/scraper.py`)  
   Gathers the latest reviews for supported banks (e.g., Commercial Bank of Ethiopia, Bank of Abyssinia, Dashen Bank).

2. **Data Preprocessing**  
   (`src/preprocessor.py`)  
   Cleans text, handles missing values, and standardizes fields.

3. **Exploratory Data Analysis**  
   (`notebooks/main.ipynb`)  
   - Loads, inspects, and summarizes the dataset.
   - Visualizes rating distributions, frequent terms, and review patterns.

4. **Thematic & Sentiment Analysis**  
   (`notebooks/thematic_analysis.ipynb`)  
   - Identifies common pain points (e.g., login issues, transaction failures).
   - Highlights app strengths (e.g., user interface, features).

5. **Reporting**  
   - Processed data and quality reports saved in `/data/processed/`.
   - Results support bank product teams and researchers.

---

## 📂 Repository Structure

```
├── data/
│   ├── raw/             # Raw scraped reviews
│   └── processed/       # Cleaned datasets, quality reports
├── notebooks/
│   ├── main.ipynb       # EDA workflow
│   └── thematic_analysis.ipynb  # Thematic/sentiment insights
├── src/
│   ├── scraper.py       # Google Play review scraper
│   └── preprocessor.py  # Data cleaning pipeline
├── LICENSE
└── README.md
```

---

## ⚡ Getting Started

### Prerequisites

- Python 3.8+
- Recommended: Create a virtual environment

### Installation

```bash
git clone https://github.com/abeelgetahun/Ethiopian-banks-review-analysis.git
cd Ethiopian-banks-review-analysis
pip install -r requirements.txt
```

### Usage

1. **Scrape Reviews:**
   - Configure target banks in `src/scraper.py`.
   - Run the scraper to fetch the latest reviews.

2. **Process Data:**
   - Execute `src/preprocessor.py` or run the relevant cells in `notebooks/main.ipynb`.

3. **Explore and Analyze:**
   - Open `notebooks/main.ipynb` for EDA.
   - Use `notebooks/thematic_analysis.ipynb` for deeper insights.

---

## 📊 Key Insights

- **Low Ratings (1-2 stars):**
  - App crashes, login/authentication issues, failed transactions.

- **High Ratings (4-5 stars):**
  - Positive user interface and experience, reliable transactions, useful features.

- **Actionable Recommendations:**
  - Banks should prioritize fixing login reliability and transaction bugs, while enhancing user experience and feature offerings.

---

## 🤝 Contributing

Contributions, suggestions, and feature requests are welcome! Please open an issue or submit a pull request.

---

## 📝 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

> **Developed by [abeelgetahun](https://github.com/abeelgetahun)**
