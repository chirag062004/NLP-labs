
# Import necessary libraries
import pandas as pd  # For data manipulation
import re  # For regular expressions used in text cleaning
import gensim  # For topic modeling using LDA
import nltk  # Natural Language Toolkit for text processing
from nltk.corpus import stopwords  # Import English stopwords
from nltk.stem import WordNetLemmatizer  # For lemmatizing words to their base form
from string import punctuation  # To get standard punctuation characters
from gensim.corpora import Dictionary  # To create a dictionary of words for LDA
from nltk.tokenize import word_tokenize  # Tokenize text into words
from gensim.models.ldamodel import LdaModel, CoherenceModel  # LDA model and coherence metrics
import pyLDAvis  # For LDA topic visualization
import pyLDAvis.gensim  # Integration with gensim
import matplotlib.pyplot as plt  # For plotting (optional)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')  # Load dataset in JSON format
print("Dataset Preview:")
print(df.head())  # Display the first 5 rows of the dataset

# Function to remove email addresses using regex
def removing_email(text):
    text = re.sub(r'\S*@\S*\s?', ' ', text)  # Remove patterns like user@domain.com
    return text

# Function to remove non-alphabet characters
def only_words(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Replace special characters and numbers with space
    return text

# Combine stopwords, punctuation, and custom noise
stop_words = (
    list(set(stopwords.words('english'))) +  # English stopwords
    list(punctuation) +  # Punctuation characters
    ['\n', '----', '---\n\n\n\n\n']  # Additional custom noise
)

# Initialize lemmatizer
lem = WordNetLemmatizer()

# Complete cleaning function: lowercase, tokenize, remove stopwords, short words, lemmatize
def cleaning(text):
    text = text.lower()  # Convert to lowercase
    words = word_tokenize(text)  # Tokenize the text
    words = [w for w in words if w not in stop_words]  # Remove stopwords
    words = [w for w in words if len(w) >= 3]  # Remove short words
    lemma = [lem.lemmatize(w, 'v') for w in words]  # Lemmatize to base form
    return lemma

# Apply text preprocessing functions
df['without email'] = df['content'].apply(removing_email)  # Remove emails
df['only words'] = df['without email'].apply(only_words)  # Remove special characters
df['clean content'] = df['only words'].apply(cleaning)  # Final cleaning steps

print("\nProcessed Data Preview:")
print(df.head())  # Show cleaned content

# Create a list of tokenized documents
clean_doc = list(df['clean content'].values)

# Create a dictionary: maps each word to a unique id
dictionary = Dictionary(clean_doc)
# Optional: dictionary.filter_extremes(no_below=5, no_above=0.5)  # Remove too rare or common words

# Convert each document into Bag-of-Words format
corpus = [dictionary.doc2bow(doc) for doc in clean_doc]  # List of (word_id, frequency) tuples

# Train LDA model with specified parameters
ldamodel = LdaModel(
    corpus=corpus,  # Bag-of-Words representation
    id2word=dictionary,  # Word-id mapping
    num_topics=5,  # Number of topics
    random_state=42,  # For reproducibility
    update_every=1,  # Update model every 1 chunk
    passes=50,  # Number of full passes through the corpus
    chunksize=100,  # Documents processed per chunk
    alpha='auto',  # Automatic learning of document-topic distribution
    eta='auto'  # Automatic learning of word-topic distribution
)

# Display the top words in each topic
print("\nDiscovered Topics:")
print(ldamodel.print_topics())

# Evaluate model using log perplexity (lower is better)
print("\nLog Perplexity:", ldamodel.log_perplexity(corpus))

# Evaluate model using Coherence Score (c_v and u_mass)
coherence_cv = CoherenceModel(
    model=ldamodel,
    texts=clean_doc,
    dictionary=dictionary,
    coherence='c_v'
)
print("\nCoherence (c_v):", coherence_cv.get_coherence())

coherence_umass = CoherenceModel(
    model=ldamodel,
    texts=clean_doc,
    dictionary=dictionary,
    coherence='u_mass'
)
print("Coherence (u_mass):", coherence_umass.get_coherence())

# Visualize topics interactively with pyLDAvis
pyLDAvis.enable_notebook()  # Enable notebook mode (can skip in .py)
vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)  # Prepare visualization
pyLDAvis.show(vis)  # Display in browser
