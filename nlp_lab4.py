# Import required libraries
import math  # Mathematical operations
import numpy as np  # Numerical computing library
import os  # Operating system interfaces
import random  # Generate random numbers
import tensorflow as tf  # Machine learning framework
from matplotlib import pylab  # Plotting library (not used in this code)
from collections import Counter  # Count hashable objects
import csv  # CSV file handling

# Set TensorFlow 1.x compatibility
# Note: This is a comment indicating TF 1.x should be used, but no actual code enforces this

# Parameters for the model and data processing
vocab_size = 50000  # Size of vocabulary to use
num_units = 128  # Number of units in RNN cells
input_size = 128  # Size of input embeddings
batch_size = 16  # Number of samples per training batch
source_sequence_length = 40  # Maximum length for source sequences
target_sequence_length = 60  # Maximum length for target sequences
decoder_type = 'basic'  # Type of decoder ('basic' or 'attention')
sentences_to_read = 50000  # Number of sentences to read from files

# Load source vocabulary (German in this case)
src_dictionary = {}  # Initialize empty dictionary for source vocabulary
with open('vocab.50K.de.txt', encoding='utf-8') as f:  # Open source vocab file
    for line in f:  # Read each line
        src_dictionary[line.strip()] = len(src_dictionary)  # Add word to dict with index as value

# Create reverse dictionary mapping indices to words
src_reverse_dictionary = dict(zip(src_dictionary.values(), src_dictionary.keys()))

# Display information about source vocabulary
print('Source')
print('\t', list(src_dictionary.items())[:10])  # First 10 items in dictionary
print('\t', list(src_reverse_dictionary.items())[:10])  # First 10 items in reverse dict
print('\t', 'Vocabulary size: ', len(src_dictionary))  # Total vocabulary size

# Load target vocabulary (English in this case)
tgt_dictionary = {}  # Initialize empty dictionary for target vocabulary
with open('vocab.50K.en.txt', encoding='utf-8') as f:  # Open target vocab file
    for line in f:  # Read each line
        tgt_dictionary[line.strip()] = len(tgt_dictionary)  # Add word to dict with index as value

# Create reverse dictionary mapping indices to words
tgt_reverse_dictionary = dict(zip(tgt_dictionary.values(), tgt_dictionary.keys()))

# Display information about target vocabulary
print('Target')
print('\t', list(tgt_dictionary.items())[:10])  # First 10 items in dictionary
print('\t', list(tgt_reverse_dictionary.items())[:10])  # First 10 items in reverse dict
print('\t', 'Vocabulary size: ', len(tgt_dictionary))  # Total vocabulary size

# Load training sentences (source language - German)
source_sent = []  # List to store source sentences
with open('train.de', encoding='utf-8') as f:  # Open source sentences file
    for l_i, line in enumerate(f):  # Read each line with index
        if l_i < 50:  # Skip first 50 lines (noisy data)
            continue
        source_sent.append(line)  # Add sentence to list
        if len(source_sent) >= sentences_to_read:  # Stop if we've read enough
            break

# Load training sentences (target language - English)
target_sent = []  # List to store target sentences
with open('train.en', encoding='utf-8') as f:  # Open target sentences file
    for l_i, line in enumerate(f):  # Read each line with index
        if l_i < 50:  # Skip first 50 lines (noisy data)
            continue
        target_sent.append(line)  # Add sentence to list
        if len(target_sent) >= sentences_to_read:  # Stop if we've read enough
            break

# Verify we have equal number of source and target sentences
assert len(source_sent) == len(target_sent), 'Mismatch between source and target sentence counts.'

# Function to split sentences into tokens and handle unknown words
def split_to_tokens(sent, is_source):
    # Add spaces around punctuation for proper tokenization
    sent = sent.replace(',', ' ,').replace('.', ' .').replace('\n', ' ')
    # Split sentence into tokens
    sent_toks = sent.split(' ')
    # Replace unknown tokens with <unk> token
    for t_i, tok in enumerate(sent_toks):
        if is_source:  # Check against source vocabulary
            if tok not in src_dictionary:
                sent_toks[t_i] = '<unk>'
        else:  # Check against target vocabulary
            if tok not in tgt_dictionary:
                sent_toks[t_i] = '<unk>'
    return sent_toks

# Calculate and print statistics about source sentence lengths
source_len = [len(split_to_tokens(sent, True)) for sent in source_sent]
print('(Source) Sentence mean length: ', np.mean(source_len))
print('(Source) Sentence stddev length: ', np.std(source_len))

# Calculate and print statistics about target sentence lengths
target_len = [len(split_to_tokens(sent, False)) for sent in target_sent]
print('(Target) Sentence mean length: ', np.mean(target_len))
print('(Target) Sentence stddev length: ', np.std(target_len))

# Prepare input and output data for training
train_inputs, train_outputs = [], []  # Lists to store numerical representations
train_inp_lengths, train_out_lengths = [], []  # Lists to store sequence lengths
src_max_sent_length = 41  # Maximum source sentence length (including special tokens)
tgt_max_sent_length = 61  # Maximum target sentence length (including special tokens)

# Process each sentence pair
for src_sent, tgt_sent in zip(source_sent, target_sent):
    # Tokenize source sentence
    src_sent_tokens = split_to_tokens(src_sent, True)
    # Tokenize target sentence
    tgt_sent_tokens = split_to_tokens(tgt_sent, False)

    # Convert source tokens to indices and reverse (common practice for NMT)
    num_src_sent = [src_dictionary[tok] for tok in src_sent_tokens[::-1]]  # reverse source
    # Add start-of-sentence token
    num_src_sent.insert(0, src_dictionary['<s>'])
    # Record actual length (before padding)
    train_inp_lengths.append(min(len(num_src_sent) + 1, src_max_sent_length))
    # Pad with end-of-sentence tokens and truncate if needed
    num_src_sent = (num_src_sent + [src_dictionary['</s>']] * src_max_sent_length)[:src_max_sent_length]
    train_inputs.append(num_src_sent)

    # Convert target tokens to indices with end-of-sentence token at start
    num_tgt_sent = [tgt_dictionary['</s>']] + [tgt_dictionary[tok] for tok in tgt_sent_tokens]
    # Record actual length (before padding)
    train_out_lengths.append(min(len(num_tgt_sent) + 1, tgt_max_sent_length))
    # Pad with end-of-sentence tokens and truncate if needed
    num_tgt_sent = (num_tgt_sent + [tgt_dictionary['</s>']] * tgt_max_sent_length)[:tgt_max_sent_length]
    train_outputs.append(num_tgt_sent)

# Convert lists to numpy arrays for efficient processing
train_inputs = np.array(train_inputs, dtype=np.int32)
train_outputs = np.array(train_outputs, dtype=np.int32)
train_inp_lengths = np.array(train_inp_lengths, dtype=np.int32)
train_out_lengths = np.array(train_out_lengths, dtype=np.int32)

# Print sample input/output pair for verification
print('Samples from bin')
# Convert indices back to words for the first sample
print('\t', [src_reverse_dictionary[w] for w in train_inputs[0]])
print('\t', [tgt_reverse_dictionary[w] for w in train_outputs[0]])