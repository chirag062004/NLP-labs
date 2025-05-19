# Import gensim's model downloader utility which provides access to pre-trained word embedding models
import gensim.downloader

# Suppress warnings to keep the output clean (not recommended for development/debugging)
import warnings
warnings.filterwarnings('ignore')

"""
Gensim Model Downloader Explanation:
- gensim.downloader provides access to various pre-trained word embedding models
- These models are trained on massive corpora and capture semantic relationships between words
- Available models include Word2Vec (Google News), GloVe (Wikipedia), FastText, etc.
"""
# List all available pre-trained models in gensim's repository
print("Available pre-trained models in gensim:")
print(list(gensim.downloader.info()['models'].keys()))  # Prints all available pre-trained models

"""
Word2Vec Model Loading:
- 'word2vec-google-news-300' is a Word2Vec model trained on Google News corpus
- Contains 300-dimensional vectors for 3 million words/phrases
- Takes significant memory to load (~1.5GB)
- The vectors capture semantic and syntactic word relationships
"""
print("\nLoading word2vec-google-news-300 model (this may take several minutes)...")
word2vec = gensim.downloader.load('word2vec-google-news-300')  # Load pre-trained Word2Vec model

"""
Most Similar Words Demonstration:
- The most_similar() method finds words with vectors most similar to the target word
- Uses cosine similarity between word vectors
- Shows how the model captures semantic relationships
"""
print("\nWords most similar to 'technology':")
print(word2vec.most_similar('technology'))  # Display similar words to 'technology'

"""
Expected Output Insight:
- Returns related terms like 'technologies', 'science', 'engineering'
- Shows the model understands technology-related concepts
"""

print("\nWords most similar to 'Science':")
print(word2vec.most_similar('Science'))  # Display similar words to 'Science'

"""
Capitalization Note:
- The model is case-insensitive for English words
- 'Science' and 'science' will yield similar results
"""

print("\nWords most similar to 'arts':")
print(word2vec.most_similar('arts'))  # Display similar words to 'arts'

"""
Semantic Field Example:
- Should return related terms from arts/culture domain
- Demonstrates the model's understanding of different knowledge domains
"""

"""
Word Similarity Measurement:
- similarity() calculates cosine similarity between two word vectors
- Returns value between -1 (opposite) and 1 (identical)
- For related words, typically between 0.5-0.8
"""
print("\nSimilarity between 'hot' and 'cold':")
print(word2vec.similarity('hot', 'cold'))  # Cosine similarity score between 'hot' and 'cold'

"""
Interesting Observation:
- While antonyms, they have high similarity because:
  1. Both are temperature-related terms
  2. Often appear in similar contexts
  3. Share many syntactic relationships
- Shows word vectors capture linguistic relationships beyond simple synonymy
"""

"""
Additional Key Concepts:

1. Word Embedding Properties:
- Similar words cluster together in vector space
- Vector arithmetic captures relationships (king - man + woman ≈ queen)
- The 300 dimensions encode various linguistic features

2. Model Limitations:
- Fixed vocabulary (out-of-vocabulary words won't work)
- Context-insensitive (same vector for all word senses)
- Trained on news data - biases present in source material

3. Practical Applications:
- Semantic search
- Document clustering
- As features for machine learning models
- Analogical reasoning

4. Alternative Methods:
- Contextual embeddings (BERT, ELMo) for polysemous words
- Domain-specific embeddings for specialized vocabularies
"""
