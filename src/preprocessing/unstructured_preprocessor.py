"""
Unstructured Data Preprocessor for Claims Risk Classification Pipeline

This module handles preprocessing of unstructured data including:
- Text cleaning and normalization
- Text vectorization (TF-IDF, Word2Vec, BERT embeddings)
- Document classification and sentiment analysis
- Named entity recognition for claims
- Text feature extraction and engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import re
import string
from collections import Counter
import joblib

# Text processing imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Vectorization imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import gensim
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# Sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# Advanced NLP (optional)
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnstructuredDataPreprocessor:
    """
    Comprehensive preprocessor for unstructured text data in claims
    
    Features:
    - Text cleaning and normalization
    - Multiple vectorization methods (TF-IDF, Word2Vec, Doc2Vec)
    - Sentiment analysis and emotion detection
    - Named entity recognition for insurance context
    - Topic modeling and document clustering
    - Text quality assessment
    - Domain-specific feature extraction
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize unstructured data preprocessor
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or self._get_default_config()
        self.is_fitted = False
        self.vectorizers = {}
        self.models = {}
        
        # Initialize text processing components
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize sentiment analyzer
        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Load spaCy model if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found. Some features will be disabled.")
                self.nlp = None
        else:
            self.nlp = None
        
        # Insurance-specific keywords and entities
        self._initialize_domain_knowledge()
        
    def _get_default_config(self) -> Dict:
        """Get default preprocessing configuration"""
        return {
            # Text cleaning
            'lowercase': True,
            'remove_punctuation': True,
            'remove_numbers': False,
            'remove_stopwords': True,
            'min_word_length': 2,
            'max_word_length': 20,
            
            # Text normalization
            'use_stemming': False,
            'use_lemmatization': True,
            'expand_contractions': True,
            
            # Vectorization
            'vectorization_methods': ['tfidf', 'word2vec'],  # 'tfidf', 'count', 'word2vec', 'doc2vec'
            'tfidf_max_features': 5000,
            'tfidf_min_df': 2,
            'tfidf_max_df': 0.95,
            'tfidf_ngram_range': (1, 2),
            
            # Word2Vec parameters
            'w2v_vector_size': 100,
            'w2v_window': 5,
            'w2v_min_count': 2,
            'w2v_workers': 4,
            
            # Feature engineering
            'extract_text_features': True,
            'perform_sentiment_analysis': True,
            'extract_entities': True,
            'perform_topic_modeling': True,
            'num_topics': 10,
            
            # Text filtering
            'min_text_length': 10,
            'max_text_length': 5000,
            'filter_languages': ['en'],
        }
    
    def _initialize_domain_knowledge(self):
        """Initialize insurance and claims domain knowledge"""
        self.insurance_keywords = {
            'damage_types': ['collision', 'comprehensive', 'liability', 'theft', 'vandalism', 
                           'fire', 'flood', 'hail', 'windstorm', 'earthquake'],
            'body_parts': ['head', 'neck', 'back', 'spine', 'arm', 'leg', 'hand', 'foot', 
                          'shoulder', 'knee', 'ankle', 'wrist'],
            'severity_indicators': ['severe', 'minor', 'major', 'critical', 'serious', 
                                  'significant', 'extensive', 'minimal', 'moderate'],
            'medical_terms': ['injury', 'fracture', 'sprain', 'strain', 'laceration', 
                            'contusion', 'concussion', 'whiplash', 'surgery', 'hospital'],
            'vehicle_parts': ['bumper', 'fender', 'door', 'window', 'windshield', 'tire', 
                            'engine', 'transmission', 'airbag', 'headlight'],
            'fraud_indicators': ['staged', 'suspicious', 'inconsistent', 'exaggerated', 
                               'fraudulent', 'false', 'misleading', 'fabricated']
        }
        
        # Compile patterns for entity extraction
        self.entity_patterns = {
            'monetary_amount': r'\$?[\d,]+\.?\d*',
            'date': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            'time': r'\b\d{1,2}:\d{2}(?:\s?[AaPp][Mm])?\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'license_plate': r'\b[A-Z]{2,3}[-\s]?\d{3,4}\b'
        }
    
    def fit(self, texts: Union[List[str], pd.Series], labels: Optional[Union[List, pd.Series]] = None) -> 'UnstructuredDataPreprocessor':
        """
        Fit the preprocessor to training texts
        
        Args:
            texts: Training text data
            labels: Optional labels for supervised feature engineering
            
        Returns:
            Self
        """
        logger.info("Fitting unstructured data preprocessor...")
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Clean and preprocess texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        cleaned_texts = [text for text in cleaned_texts if len(text) >= self.config['min_text_length']]
        
        # Fit vectorizers
        self._fit_vectorizers(cleaned_texts)
        
        # Fit models for advanced features
        if self.config['perform_topic_modeling']:
            self._fit_topic_model(cleaned_texts)
        
        if 'word2vec' in self.config['vectorization_methods']:
            self._fit_word2vec(cleaned_texts)
        
        if 'doc2vec' in self.config['vectorization_methods']:
            self._fit_doc2vec(cleaned_texts, labels)
        
        self.is_fitted = True
        logger.info("Unstructured data preprocessor fitted successfully")
        return self
    
    def transform(self, texts: Union[List[str], pd.Series]) -> pd.DataFrame:
        """
        Transform texts using fitted preprocessor
        
        Args:
            texts: Texts to transform
            
        Returns:
            DataFrame with extracted features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        logger.info("Transforming unstructured data...")
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        # Initialize feature DataFrame
        features_df = pd.DataFrame()
        
        # Basic text features
        if self.config['extract_text_features']:
            text_features = self._extract_text_features(texts, cleaned_texts)
            features_df = pd.concat([features_df, text_features], axis=1)
        
        # Vectorization features
        if 'tfidf' in self.config['vectorization_methods']:
            tfidf_features = self._transform_tfidf(cleaned_texts)
            features_df = pd.concat([features_df, tfidf_features], axis=1)
        
        if 'word2vec' in self.config['vectorization_methods'] and 'word2vec' in self.models:
            w2v_features = self._transform_word2vec(cleaned_texts)
            features_df = pd.concat([features_df, w2v_features], axis=1)
        
        # Sentiment analysis
        if self.config['perform_sentiment_analysis']:
            sentiment_features = self._extract_sentiment_features(texts)
            features_df = pd.concat([features_df, sentiment_features], axis=1)
        
        # Entity extraction
        if self.config['extract_entities']:
            entity_features = self._extract_entity_features(texts)
            features_df = pd.concat([features_df, entity_features], axis=1)
        
        # Topic modeling features
        if self.config['perform_topic_modeling'] and 'lda' in self.models:
            topic_features = self._transform_topics(cleaned_texts)
            features_df = pd.concat([features_df, topic_features], axis=1)
        
        logger.info(f"Text data transformed. Features shape: {features_df.shape}")
        return features_df
    
    def fit_transform(self, texts: Union[List[str], pd.Series], labels: Optional[Union[List, pd.Series]] = None) -> pd.DataFrame:
        """
        Fit preprocessor and transform texts in one step
        
        Args:
            texts: Training text data
            labels: Optional labels
            
        Returns:
            DataFrame with extracted features
        """
        return self.fit(texts, labels).transform(texts)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize individual text"""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Expand contractions
        if self.config['expand_contractions']:
            text = self._expand_contractions(text)
        
        # Convert to lowercase
        if self.config['lowercase']:
            text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very long or short texts
        if len(text) < self.config['min_text_length'] or len(text) > self.config['max_text_length']:
            return ""
        
        # Remove numbers (optional)
        if self.config['remove_numbers']:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation (optional, but keep for some features)
        if self.config['remove_punctuation']:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenization and further processing
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.config['remove_stopwords']:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token.lower() not in stop_words]
        
        # Filter by word length
        tokens = [token for token in tokens 
                 if self.config['min_word_length'] <= len(token) <= self.config['max_word_length']]
        
        # Stemming or Lemmatization
        if self.config['use_stemming']:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif self.config['use_lemmatization']:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def _expand_contractions(self, text: str) -> str:
        """Expand common English contractions"""
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _fit_vectorizers(self, texts: List[str]):
        """Fit text vectorizers"""
        # TF-IDF Vectorizer
        if 'tfidf' in self.config['vectorization_methods']:
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=self.config['tfidf_max_features'],
                min_df=self.config['tfidf_min_df'],
                max_df=self.config['tfidf_max_df'],
                ngram_range=self.config['tfidf_ngram_range'],
                stop_words='english'
            )
            self.vectorizers['tfidf'].fit(texts)
        
        # Count Vectorizer
        if 'count' in self.config['vectorization_methods']:
            self.vectorizers['count'] = CountVectorizer(
                max_features=self.config['tfidf_max_features'],
                min_df=self.config['tfidf_min_df'],
                max_df=self.config['tfidf_max_df'],
                ngram_range=self.config['tfidf_ngram_range'],
                stop_words='english'
            )
            self.vectorizers['count'].fit(texts)
    
    def _fit_topic_model(self, texts: List[str]):
        """Fit topic model using LDA"""
        try:
            # Use count vectorizer for LDA
            count_vectorizer = CountVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )
            count_matrix = count_vectorizer.fit_transform(texts)
            
            # Fit LDA model
            self.models['lda'] = LatentDirichletAllocation(
                n_components=self.config['num_topics'],
                random_state=42,
                max_iter=100
            )
            self.models['lda'].fit(count_matrix)
            self.models['lda_vectorizer'] = count_vectorizer
            
            logger.info(f"Topic model fitted with {self.config['num_topics']} topics")
        except Exception as e:
            logger.warning(f"Could not fit topic model: {e}")
    
    def _fit_word2vec(self, texts: List[str]):
        """Fit Word2Vec model"""
        try:
            # Tokenize texts for Word2Vec
            tokenized_texts = [text.split() for text in texts]
            
            self.models['word2vec'] = Word2Vec(
                tokenized_texts,
                vector_size=self.config['w2v_vector_size'],
                window=self.config['w2v_window'],
                min_count=self.config['w2v_min_count'],
                workers=self.config['w2v_workers'],
                seed=42
            )
            
            logger.info(f"Word2Vec model fitted with {self.config['w2v_vector_size']} dimensions")
        except Exception as e:
            logger.warning(f"Could not fit Word2Vec model: {e}")
    
    def _fit_doc2vec(self, texts: List[str], labels: Optional[List] = None):
        """Fit Doc2Vec model"""
        try:
            # Prepare tagged documents
            tagged_docs = [TaggedDocument(words=text.split(), tags=[str(i)]) 
                          for i, text in enumerate(texts)]
            
            self.models['doc2vec'] = Doc2Vec(
                tagged_docs,
                vector_size=self.config['w2v_vector_size'],
                window=self.config['w2v_window'],
                min_count=self.config['w2v_min_count'],
                workers=self.config['w2v_workers'],
                epochs=40,
                seed=42
            )
            
            logger.info(f"Doc2Vec model fitted")
        except Exception as e:
            logger.warning(f"Could not fit Doc2Vec model: {e}")
    
    def _extract_text_features(self, original_texts: List[str], cleaned_texts: List[str]) -> pd.DataFrame:
        """Extract basic text features"""
        features = {
            'text_length': [len(text) for text in original_texts],
            'word_count': [len(text.split()) for text in original_texts],
            'sentence_count': [len(sent_tokenize(text)) for text in original_texts],
            'avg_word_length': [np.mean([len(word) for word in text.split()]) if text.split() else 0 
                               for text in cleaned_texts],
            'punctuation_count': [sum(1 for char in text if char in string.punctuation) 
                                 for text in original_texts],
            'capital_letters_count': [sum(1 for char in text if char.isupper()) 
                                    for text in original_texts],
            'digit_count': [sum(1 for char in text if char.isdigit()) 
                           for text in original_texts]
        }
        
        # Domain-specific features
        for category, keywords in self.insurance_keywords.items():
            feature_name = f'{category}_mentions'
            features[feature_name] = [
                sum(1 for keyword in keywords if keyword in text.lower())
                for text in original_texts
            ]
        
        return pd.DataFrame(features)
    
    def _transform_tfidf(self, texts: List[str]) -> pd.DataFrame:
        """Transform texts using fitted TF-IDF vectorizer"""
        if 'tfidf' not in self.vectorizers:
            return pd.DataFrame()
        
        tfidf_matrix = self.vectorizers['tfidf'].transform(texts)
        feature_names = self.vectorizers['tfidf'].get_feature_names_out()
        
        # Convert to DataFrame with proper feature names
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{name}' for name in feature_names]
        )
        
        return tfidf_df
    
    def _transform_word2vec(self, texts: List[str]) -> pd.DataFrame:
        """Transform texts using Word2Vec embeddings"""
        if 'word2vec' not in self.models:
            return pd.DataFrame()
        
        w2v_model = self.models['word2vec']
        features = []
        
        for text in texts:
            words = text.split()
            word_vectors = []
            
            for word in words:
                if word in w2v_model.wv:
                    word_vectors.append(w2v_model.wv[word])
            
            if word_vectors:
                # Average word vectors to get document vector
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                # Zero vector if no words found
                doc_vector = np.zeros(self.config['w2v_vector_size'])
            
            features.append(doc_vector)
        
        # Create DataFrame
        w2v_df = pd.DataFrame(
            features,
            columns=[f'w2v_dim_{i}' for i in range(self.config['w2v_vector_size'])]
        )
        
        return w2v_df
    
    def _extract_sentiment_features(self, texts: List[str]) -> pd.DataFrame:
        """Extract sentiment analysis features"""
        features = {
            'sentiment_compound': [],
            'sentiment_positive': [],
            'sentiment_negative': [],
            'sentiment_neutral': []
        }
        
        if VADER_AVAILABLE and self.sentiment_analyzer:
            for text in texts:
                scores = self.sentiment_analyzer.polarity_scores(text)
                features['sentiment_compound'].append(scores['compound'])
                features['sentiment_positive'].append(scores['pos'])
                features['sentiment_negative'].append(scores['neg'])
                features['sentiment_neutral'].append(scores['neu'])
        else:
            # Simple sentiment analysis fallback
            positive_words = ['good', 'excellent', 'great', 'satisfied', 'happy', 'pleased']
            negative_words = ['bad', 'terrible', 'awful', 'dissatisfied', 'unhappy', 'angry']
            
            for text in texts:
                text_lower = text.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                
                total = pos_count + neg_count
                features['sentiment_compound'].append((pos_count - neg_count) / max(total, 1))
                features['sentiment_positive'].append(pos_count / max(total, 1))
                features['sentiment_negative'].append(neg_count / max(total, 1))
                features['sentiment_neutral'].append(1 - ((pos_count + neg_count) / max(len(text.split()), 1)))
        
        return pd.DataFrame(features)
    
    def _extract_entity_features(self, texts: List[str]) -> pd.DataFrame:
        """Extract named entities and patterns"""
        features = {
            'monetary_amounts': [],
            'dates_mentioned': [],
            'times_mentioned': [],
            'phone_numbers': [],
            'license_plates': []
        }
        
        for text in texts:
            # Extract using regex patterns
            for entity_type, pattern in self.entity_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                
                if entity_type == 'monetary_amount':
                    features['monetary_amounts'].append(len(matches))
                elif entity_type == 'date':
                    features['dates_mentioned'].append(len(matches))
                elif entity_type == 'time':
                    features['times_mentioned'].append(len(matches))
                elif entity_type == 'phone':
                    features['phone_numbers'].append(len(matches))
                elif entity_type == 'license_plate':
                    features['license_plates'].append(len(matches))
        
        # Named entity recognition using spaCy (if available)
        if self.nlp:
            person_counts = []
            org_counts = []
            location_counts = []
            
            for text in texts:
                doc = self.nlp(text)
                person_count = len([ent for ent in doc.ents if ent.label_ == 'PERSON'])
                org_count = len([ent for ent in doc.ents if ent.label_ == 'ORG'])
                location_count = len([ent for ent in doc.ents if ent.label_ in ['GPE', 'LOC']])
                
                person_counts.append(person_count)
                org_counts.append(org_count)
                location_counts.append(location_count)
            
            features['person_mentions'] = person_counts
            features['organization_mentions'] = org_counts
            features['location_mentions'] = location_counts
        
        return pd.DataFrame(features)
    
    def _transform_topics(self, texts: List[str]) -> pd.DataFrame:
        """Transform texts using topic model"""
        if 'lda' not in self.models or 'lda_vectorizer' not in self.models:
            return pd.DataFrame()
        
        try:
            # Transform texts
            count_matrix = self.models['lda_vectorizer'].transform(texts)
            topic_probs = self.models['lda'].transform(count_matrix)
            
            # Create DataFrame
            topic_df = pd.DataFrame(
                topic_probs,
                columns=[f'topic_{i}' for i in range(self.config['num_topics'])]
            )
            
            return topic_df
        except Exception as e:
            logger.warning(f"Could not transform topics: {e}")
            return pd.DataFrame()
    
    def save_preprocessor(self, file_path: str):
        """Save fitted preprocessor to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        # Prepare data for saving (excluding non-serializable objects)
        save_data = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'vectorizers': self.vectorizers,
            'insurance_keywords': self.insurance_keywords,
            'entity_patterns': self.entity_patterns
        }
        
        # Save models separately if they exist
        if 'word2vec' in self.models:
            self.models['word2vec'].save(f"{file_path}_word2vec.model")
            save_data['has_word2vec'] = True
        
        if 'doc2vec' in self.models:
            self.models['doc2vec'].save(f"{file_path}_doc2vec.model")
            save_data['has_doc2vec'] = True
        
        if 'lda' in self.models:
            save_data['lda_model'] = self.models['lda']
            save_data['lda_vectorizer'] = self.models['lda_vectorizer']
        
        joblib.dump(save_data, file_path)
        logger.info(f"Unstructured preprocessor saved to {file_path}")
    
    @classmethod
    def load_preprocessor(cls, file_path: str) -> 'UnstructuredDataPreprocessor':
        """Load fitted preprocessor from disk"""
        save_data = joblib.load(file_path)
        
        preprocessor = cls(save_data['config'])
        preprocessor.is_fitted = save_data['is_fitted']
        preprocessor.vectorizers = save_data['vectorizers']
        preprocessor.insurance_keywords = save_data['insurance_keywords']
        preprocessor.entity_patterns = save_data['entity_patterns']
        
        # Load models
        if save_data.get('has_word2vec', False):
            preprocessor.models['word2vec'] = Word2Vec.load(f"{file_path}_word2vec.model")
        
        if save_data.get('has_doc2vec', False):
            preprocessor.models['doc2vec'] = Doc2Vec.load(f"{file_path}_doc2vec.model")
        
        if 'lda_model' in save_data:
            preprocessor.models['lda'] = save_data['lda_model']
            preprocessor.models['lda_vectorizer'] = save_data['lda_vectorizer']
        
        logger.info(f"Unstructured preprocessor loaded from {file_path}")
        return preprocessor
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature importance for TF-IDF features"""
        if 'tfidf' not in self.vectorizers:
            return pd.DataFrame()
        
        # Get TF-IDF feature names and scores
        tfidf_features = self.vectorizers['tfidf'].get_feature_names_out()
        
        # For demonstration, use IDF scores as importance
        idf_scores = self.vectorizers['tfidf'].idf_
        
        importance_df = pd.DataFrame({
            'feature': [f'tfidf_{name}' for name in tfidf_features],
            'importance': idf_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing configuration and fitted models"""
        summary = {
            'is_fitted': self.is_fitted,
            'vectorization_methods': self.config['vectorization_methods'],
            'text_cleaning_enabled': self.config['lowercase'] and self.config['remove_stopwords'],
            'sentiment_analysis': self.config['perform_sentiment_analysis'],
            'entity_extraction': self.config['extract_entities'],
            'topic_modeling': self.config['perform_topic_modeling']
        }
        
        if self.is_fitted:
            summary['fitted_components'] = list(self.vectorizers.keys()) + list(self.models.keys())
            
            if 'tfidf' in self.vectorizers:
                summary['tfidf_features'] = len(self.vectorizers['tfidf'].get_feature_names_out())
                
            if 'word2vec' in self.models:
                summary['word2vec_vocab_size'] = len(self.models['word2vec'].wv.key_to_index)
        
        return summary


# Example usage
if __name__ == "__main__":
    # Sample claim descriptions
    sample_texts = [
        "I was involved in a car accident on Highway 101. The other driver ran a red light and hit my vehicle on the passenger side. There was significant damage to my car and I suffered minor injuries including whiplash and bruising.",
        "My home was damaged during the recent storm. A large tree fell on my roof causing water damage to the living room and kitchen. The repair costs are estimated at $15,000.",
        "I fell down the stairs at work and injured my back. I was taken to the hospital where X-rays showed a herniated disc. I will need physical therapy and possibly surgery.",
        "Someone broke into my car last night and stole my laptop, GPS, and some cash. The window was smashed and the door lock was damaged. Police report #12345 was filed.",
        "I was rear-ended at a traffic light by a distracted driver. My neck hurts and I have a headache. The damage to my car is minor but I want to get checked by a doctor."
    ]
    
    # Initialize preprocessor
    preprocessor = UnstructuredDataPreprocessor()
    
    # Fit and transform
    features_df = preprocessor.fit_transform(sample_texts)
    
    print("Preprocessing Summary:")
    print(preprocessor.get_preprocessing_summary())
    print(f"\nExtracted features shape: {features_df.shape}")
    print(f"Feature categories: {features_df.columns.tolist()[:20]}...")  # Show first 20 features
    
    # Save preprocessor
    preprocessor.save_preprocessor("unstructured_preprocessor.pkl")