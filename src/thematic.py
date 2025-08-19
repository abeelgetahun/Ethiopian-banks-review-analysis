import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from collections import Counter
try:
    import spacy
except Exception:  # optional dependency
    spacy = None
import string
from langdetect import detect, LangDetectException

# Avoid downloads at import; try to ensure resources on first use instead

def _get_spacy_model():
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        try:
            # Attempt to download quietly
            spacy.cli.download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception:
            return None

# Custom stopwords list with banking terms we want to keep
ADDITIONAL_STOPWORDS = ['app', 'bank', 'use', 'using', 'used', 'good', 'great', 'best', 'bad',
                       'worst', 'better', 'worse', 'nice', 'excellent', 'terrible', 'awesome']

# Banking terms to preserve (not treated as stopwords)
BANKING_TERMS = ['transfer', 'transaction', 'account', 'balance', 'payment', 'money', 'deposit', 
                'withdraw', 'login', 'atm', 'card', 'credit', 'debit', 'mobile', 'password', 
                'authentication', 'otp', 'security', 'interface', 'ui', 'ux', 'feature', 'service',
                'customer', 'support', 'fee', 'charge', 'error', 'bug', 'crash', 'slow', 'fast',
                'easy', 'difficult', 'simple', 'complex', 'verify', 'verification', 'pin', 'block']

def preprocess_text(text, remove_banking_terms=False):
    """Preprocess text for thematic analysis."""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs, emails, mentions
    text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+|\S+\.\S+|@\S+', '', text)
    
    # Remove numbers and special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Ensure NLTK resources
    # Try NLTK tokenization; if unavailable, fall back to regex tokens
    tokens = []
    try:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
                # Newer NLTK versions also need punkt_tab
                nltk.download('punkt_tab', quiet=True)
            except Exception:
                pass
        tokens = word_tokenize(text)
    except Exception:
        # Very simple fallback tokenizer
        tokens = re.findall(r"\b\w+\b", text)
    
    # Remove stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except Exception:
            pass
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        # Minimal fallback set
        stop_words = {
            'the','a','an','and','or','in','on','at','to','for','of','is','are','was','were','be','this','that','it','with','as','by','from'
        }
    
    # Add custom stopwords but keep banking terms
    final_stopwords = stop_words.union(set(ADDITIONAL_STOPWORDS))
    if not remove_banking_terms:
        final_stopwords = final_stopwords - set(BANKING_TERMS)
    
    tokens = [word for word in tokens if word not in final_stopwords and len(word) > 2]
    
    # Lemmatize
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        try:
            nltk.download('wordnet', quiet=True)
        except Exception:
            pass
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    except Exception:
        # If lemmatizer unavailable, keep tokens as-is
        tokens = list(tokens)
    
    return " ".join(tokens)

def detect_language(text):
    """Detect the language of the text."""
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'

def extract_keywords_tfidf(df, n_keywords=20, ngram_range=(1, 3)):
    """Extract keywords using TF-IDF."""
    # Preprocess reviews
    df['processed_text'] = df['review_text'].apply(preprocess_text)
    
    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=1000, 
                           min_df=2, 
                           max_df=0.8, 
                           ngram_range=ngram_range)
    
    # Fit and transform
    tfidf_matrix = tfidf.fit_transform(df['processed_text'])
    
    # Get feature names
    feature_names = tfidf.get_feature_names_out()
    
    # Extract top keywords for each bank
    bank_keywords = {}
    for bank in df['bank'].unique():
        bank_df = df[df['bank'] == bank]
        
        # Skip if no reviews for this bank
        if len(bank_df) == 0:
            continue
            
        bank_tfidf = tfidf.transform(bank_df['processed_text'])
        
        # Sum TF-IDF scores across all documents for this bank
        bank_scores = bank_tfidf.sum(axis=0).A1
        
        # Get top keywords
        top_indices = bank_scores.argsort()[-n_keywords:][::-1]
        bank_keywords[bank] = [(feature_names[i], bank_scores[i]) for i in top_indices]
    
    return bank_keywords

def extract_keywords_spacy(df, n_keywords=20):
    """Extract keywords using spaCy."""
    # Add language detection
    df['language'] = df['review_text'].apply(
        lambda x: detect_language(str(x)) if pd.notnull(x) else 'unknown'
    )

    nlp = _get_spacy_model()
    if nlp is None:
        # Fallback: use TFIDF keywords if spaCy model unavailable
        tfidf_keywords = extract_keywords_tfidf(df)
        # Convert to expected shape
        keywords_rows = []
        for bank, pairs in tfidf_keywords.items():
            for kw, score in pairs:
                keywords_rows.append({"bank": bank, "keyword": kw, "score": score})
        return keywords_rows

    # Only process English reviews with spaCy
    english_df = df[df['language'] == 'en'].copy()
    english_df['processed_text'] = english_df['review_text'].apply(
        lambda x: ' '.join([token.lemma_ for token in nlp(str(x))
                           if not token.is_stop and not token.is_punct and len(token.text) > 2])
    )
    
    bank_keywords = {}
    for bank in english_df['bank'].unique():
        bank_df = english_df[english_df['bank'] == bank]
        
        # Skip if no reviews for this bank
        if len(bank_df) == 0:
            continue
            
        # Process all reviews together for this bank
        all_reviews = ' '.join(bank_df['processed_text'].tolist())
        doc = nlp(all_reviews)
        
        # Extract noun phrases and named entities
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        entities = [ent.text for ent in doc.ents]
        
        # Count frequencies
        phrase_counter = Counter(noun_phrases)
        entity_counter = Counter(entities)
        
        # Combine and get top keywords
        combined_counter = phrase_counter + entity_counter
        bank_keywords[bank] = combined_counter.most_common(n_keywords)
    
    return bank_keywords

def identify_themes(bank_keywords, num_themes=5):
    """
    Manually group keywords into themes.
    This is a rule-based approach that maps keywords to predefined themes.
    """
    # Define theme categories and associated keywords
    theme_mapping = {
        'Account Access Issues': ['login', 'password', 'access', 'authentication', 'otp', 'verification', 
                               'register', 'sign', 'account', 'pin', 'fingerprint', 'face', 'biometric', 'block'],
        
        'Transaction Performance': ['transfer', 'transaction', 'payment', 'deposit', 'withdraw', 'money', 
                                 'send', 'receive', 'balance', 'check', 'speed', 'quick', 'slow', 'delay', 
                                 'pending', 'failed', 'success', 'fee', 'charge', 'cost', 'amount'],
        
        'User Interface & Experience': ['ui', 'ux', 'interface', 'design', 'layout', 'screen', 'menu', 'navigation', 
                                     'easy', 'difficult', 'simple', 'complex', 'intuitive', 'confusing', 'user', 
                                     'friendly', 'experience', 'dark', 'mode', 'theme', 'color'],
        
        'App Performance & Reliability': ['crash', 'bug', 'error', 'glitch', 'freeze', 'hang', 'slow', 'fast', 
                                       'responsive', 'load', 'performance', 'stable', 'unstable', 'reliable', 
                                       'unreliable', 'server', 'maintenance', 'update', 'version'],
        
        'Customer Support': ['support', 'service', 'help', 'contact', 'call', 'center', 'customer', 'agent', 
                          'response', 'responsive', 'unresponsive', 'solve', 'solution', 'resolve', 'issue', 
                          'problem', 'complaint', 'feedback'],
        
        'Features & Functionality': ['feature', 'function', 'option', 'tool', 'capability', 'service', 'offer',
                                  'provide', 'add', 'missing', 'need', 'want', 'lack', 'request', 'suggest',
                                  'bill', 'airtime', 'statement', 'history', 'report', 'utility'],
        
        'Security & Trust': ['security', 'secure', 'trust', 'safe', 'privacy', 'protect', 'data', 'information',
                          'fraud', 'scam', 'hack', 'breach', 'concern', 'worry', 'risk', 'danger'],
                          
        'Network & Connectivity': ['network', 'internet', 'connection', 'connect', 'disconnect', 'offline', 
                               'online', 'data', 'wifi', 'mobile', 'signal', 'server', 'downtime']
    }
    
    # Map keywords to themes for each bank
    bank_themes = {}
    
    for bank, keywords in bank_keywords.items():
        theme_scores = {theme: 0 for theme in theme_mapping}
        
        # Process keywords and calculate theme scores
        for keyword_info in keywords:
            if isinstance(keyword_info, tuple):
                keyword, score = keyword_info
            else:
                keyword, score = keyword_info, 1
                
            # Clean the keyword
            keyword = keyword.lower()
            
            # Match keyword to themes
            for theme, theme_keywords in theme_mapping.items():
                for theme_keyword in theme_keywords:
                    if theme_keyword in keyword:
                        theme_scores[theme] += score
                        break
        
        # Sort themes by score and select top ones
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        bank_themes[bank] = sorted_themes[:num_themes]
    
    return bank_themes

def topic_modeling(df, n_topics=5, method='lda'):
    """
    Perform topic modeling using either LDA or NMF.
    
    Args:
        df: DataFrame with reviews
        n_topics: Number of topics to extract
        method: 'lda' or 'nmf'
        
    Returns:
        topics: Dictionary with topic words
    """
    # Preprocess text
    df['processed_text'] = df['review_text'].apply(preprocess_text)
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(max_features=1000, 
                           min_df=2, 
                           max_df=0.8, 
                           ngram_range=(1, 2))
    
    X = tfidf.fit_transform(df['processed_text'])
    feature_names = tfidf.get_feature_names_out()
    
    # Apply topic modeling
    if method == 'lda':
        model = LatentDirichletAllocation(n_components=n_topics, 
                                         random_state=42,
                                         max_iter=10)
    else:  # NMF
        model = NMF(n_components=n_topics, 
                   random_state=42,
                   max_iter=1000)
    
    model.fit(X)
    
    # Extract topics
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-11:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics[f'Topic {topic_idx+1}'] = top_features
    
    return topics

def assign_themes_to_reviews(df, bank_themes):
    """Assign identified themes to each review."""
    # Create a flattened dictionary mapping keywords to themes
    keyword_to_theme = {}
    for theme, keywords in {
        'Account Access Issues': ['login', 'password', 'access', 'authentication', 'otp', 'verification', 
                               'register', 'sign', 'account', 'pin', 'fingerprint', 'face', 'biometric', 'block'],
        
        'Transaction Performance': ['transfer', 'transaction', 'payment', 'deposit', 'withdraw', 'money', 
                                 'send', 'receive', 'balance', 'check', 'speed', 'quick', 'slow', 'delay', 
                                 'pending', 'failed', 'success', 'fee', 'charge', 'cost', 'amount'],
        
        'User Interface & Experience': ['ui', 'ux', 'interface', 'design', 'layout', 'screen', 'menu', 'navigation', 
                                     'easy', 'difficult', 'simple', 'complex', 'intuitive', 'confusing', 'user', 
                                     'friendly', 'experience', 'dark', 'mode', 'theme', 'color'],
        
        'App Performance & Reliability': ['crash', 'bug', 'error', 'glitch', 'freeze', 'hang', 'slow', 'fast', 
                                       'responsive', 'load', 'performance', 'stable', 'unstable', 'reliable', 
                                       'unreliable', 'server', 'maintenance', 'update', 'version'],
        
        'Customer Support': ['support', 'service', 'help', 'contact', 'call', 'center', 'customer', 'agent', 
                          'response', 'responsive', 'unresponsive', 'solve', 'solution', 'resolve', 'issue', 
                          'problem', 'complaint', 'feedback'],
        
        'Features & Functionality': ['feature', 'function', 'option', 'tool', 'capability', 'service', 'offer',
                                  'provide', 'add', 'missing', 'need', 'want', 'lack', 'request', 'suggest',
                                  'bill', 'airtime', 'statement', 'history', 'report', 'utility'],
        
        'Security & Trust': ['security', 'secure', 'trust', 'safe', 'privacy', 'protect', 'data', 'information',
                          'fraud', 'scam', 'hack', 'breach', 'concern', 'worry', 'risk', 'danger'],
                          
        'Network & Connectivity': ['network', 'internet', 'connection', 'connect', 'disconnect', 'offline', 
                               'online', 'data', 'wifi', 'mobile', 'signal', 'server', 'downtime']
    }.items():
        for keyword in keywords:
            keyword_to_theme[keyword] = theme
    
    # Function to assign themes to a review
    def assign_themes(row):
        review_text = str(row['review_text']).lower()
        bank = row['bank']
        
        # Get available themes for this bank
        available_themes = [theme for theme, _ in bank_themes.get(bank, [])]
        
        # Match keywords in the review to themes
        matched_themes = set()
        for keyword, theme in keyword_to_theme.items():
            if keyword in review_text and theme in available_themes:
                matched_themes.add(theme)
        
        # If no themes matched, assign the top theme for this bank
        if not matched_themes and bank in bank_themes and bank_themes[bank]:
            matched_themes.add(bank_themes[bank][0][0])
        
        return '; '.join(matched_themes) if matched_themes else 'Other'
    
    # Apply theme assignment
    df['identified_themes'] = df.apply(assign_themes, axis=1)
    
    return df