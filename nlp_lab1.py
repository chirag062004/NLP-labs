# Import necessary libraries
import pandas as pd  # For data manipulation
from tensorflow.keras.models import Sequential  # Sequential model architecture
from tensorflow.keras.layers import LSTM, Dense, Embedding  # LSTM and dense layers for the model
from tensorflow.keras.preprocessing.text import Tokenizer  # Convert text to sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Make sequences equal length
from nltk.tokenize import word_tokenize  # Tokenize text into words
from nltk.stem import WordNetLemmatizer  # Reduce words to their root form
from nltk.corpus import stopwords  # Common words to ignore
from string import punctuation  # List of punctuation symbols
import numpy as np  # For numerical operations
from tqdm import tqdm  # Progress bar for loops
import nltk  # Natural Language Toolkit

# Enable progress bar for pandas apply
tqdm.pandas()

# Load the dataset (update the path to your local CSV if needed)
df = pd.read_csv('Quora Text Classification Data.csv')
print(df.head())  # Display first few rows

# Download required resources for text preprocessing
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Combine English stopwords and punctuation for removal
stop_words = stopwords.words('english') + list(punctuation)
lem = WordNetLemmatizer()  # Initialize the lemmatizer

# Define the cleaning function
def cleaning(text):
    text = text.lower()  # Convert to lowercase
    words = word_tokenize(text)  # Tokenize the text
    words = [w for w in words if w not in stop_words]  # Remove stopwords and punctuation
    words = [lem.lemmatize(w) for w in words]  # Lemmatize each word
    return ' '.join(words)  # Join words back into a cleaned string

# Apply text cleaning to the question_text column
df['Clean Text'] = df['question_text'].progress_apply(cleaning)

# Load pre-trained GloVe word vectors (adjust path if necessary)
embedding_values = {}
with open('glove.42B.300d.txt', encoding='utf8') as f:
    for line in tqdm(f):
        value = line.split(' ')
        word = value[0]  # First item is the word
        coef = np.array(value[1:], dtype='float32')  # Rest is the vector
        embedding_values[word] = coef  # Add word and its vector

# Prepare tokenizer and sequences
tokenizer = Tokenizer()
x = df['Clean Text']  # Input text
y = df['target']  # Output labels

tokenizer.fit_on_texts(x)  # Build vocabulary
seq = tokenizer.texts_to_sequences(x)  # Convert text to sequences of integers
pad_seq = pad_sequences(seq, maxlen=300)  # Pad sequences to 300 words

# Get vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")

# Create embedding matrix
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tqdm(tokenizer.word_index.items()):
    value = embedding_values.get(word)
    if value is not None:
        embedding_matrix[i] = value

# Build the model
model = Sequential()
model.add(Embedding(vocab_size, 300, input_length=300,
                    weights=[embedding_matrix], trainable=False))  # Pre-trained embedding
model.add(LSTM(50, return_sequences=False))  # LSTM layer with 50 units
model.add(Dense(128, activation='relu'))  # Fully connected hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(pad_seq, y, validation_split=0.2, epochs=5)  # Train for 5 epochs
