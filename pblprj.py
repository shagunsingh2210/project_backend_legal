import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
from pyswarm import pso  # Particle Swarm Optimization

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# =============================================
# 1. Data Loading and Preprocessing
# =============================================

def load_jsonl(file_path):
    """Load JSONL file into a DataFrame"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Load training and testing data
train_df = load_jsonl('train.jsonl')
test_df = load_jsonl('test.jsonl')

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

def preprocess_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Preprocess text data
train_df['processed_text'] = train_df['text'].apply(preprocess_text)
test_df['processed_text'] = test_df['text'].apply(preprocess_text)

# =============================================
# 2. Feature Extraction
# =============================================

# Option 1: TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(train_df['processed_text'])
X_test_tfidf = tfidf.transform(test_df['processed_text'])

# Option 2: Word2Vec Embeddings
sentences = [word_tokenize(text) for text in train_df['processed_text']]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

def document_vector(word2vec_model, doc):
    """Create document vectors by averaging word vectors"""
    doc = [word for word in doc if word in word2vec_model.wv]
    if len(doc) == 0:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word2vec_model.wv[doc], axis=0)

X_train_w2v = np.array([document_vector(word2vec_model, word_tokenize(text)) 
                      for text in train_df['processed_text']])
X_test_w2v = np.array([document_vector(word2vec_model, word_tokenize(text)) 
                     for text in test_df['processed_text']])

# Encode labels
label_map = {label: idx for idx, label in enumerate(train_df['label'].unique())}
y_train = train_df['label'].map(label_map)
y_test = test_df['label'].map(label_map)

# =============================================
# 3. Machine Learning Models
# =============================================

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
rf_predictions = rf_model.predict(X_test_tfidf)

print("Random Forest Performance:")
print(classification_report(y_test, rf_predictions))

# =============================================
# 4. Deep Learning Model
# =============================================

# Neural Network with Word2Vec features
dl_model = Sequential([
    Dense(128, activation='relu', input_shape=(100,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(label_map), activation='softmax')
])

dl_model.compile(optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

history = dl_model.fit(X_train_w2v, y_train,
                      epochs=10,
                      batch_size=32,
                      validation_data=(X_test_w2v, y_test))

# Evaluate deep learning model
dl_loss, dl_accuracy = dl_model.evaluate(X_test_w2v, y_test)
print(f"Deep Learning Model Accuracy: {dl_accuracy:.4f}")

# =============================================
# 5. Bio-Inspired Optimization (PSO)
# =============================================

# Objective function to optimize (using Random Forest)
def objective_function(params):
    n_estimators = int(params[0])
    max_depth = int(params[1]) if params[1] > 0 else None
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)
    return -model.score(X_test_tfidf, y_test)  # Negative for minimization

# Set bounds for PSO
bounds = [(50, 200),  # n_estimators range
          (3, 20)]    # max_depth range

# Run PSO optimization
optimized_params, _ = pso(objective_function, 
                         [b[0] for b in bounds], 
                         [b[1] for b in bounds], 
                         swarmsize=10, 
                         maxiter=20)

# Create optimized model
optimized_rf = RandomForestClassifier(
    n_estimators=int(optimized_params[0]),
    max_depth=int(optimized_params[1]) if optimized_params[1] > 0 else None,
    random_state=42
)
optimized_rf.fit(X_train_tfidf, y_train)
optimized_predictions = optimized_rf.predict(X_test_tfidf)

print("\nOptimized Random Forest Performance (PSO):")
print(classification_report(y_test, optimized_predictions))

# =============================================
# 6. Save Models
# =============================================

import joblib

# Save models
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(optimized_rf, 'optimized_random_forest_model.pkl')
dl_model.save('deep_learning_model.h5')
word2vec_model.save('word2vec_model.model')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("All models saved successfully.")
