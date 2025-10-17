import json
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
from scipy.spatial import distance
import time
import re
from token_calculator import enforce_token_limit

def split_and_lower(text):
    words = re.findall(r'\b\w+\b', text)
    lower_words = [word.lower() for word in words]
    return lower_words

# Start measuring time for loading model
start_time = time.time()

# Load the model
print("Loading model...")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

# Print model loading time
model_load_time = time.time() - start_time
print(f"Model loaded in {model_load_time} seconds")

# File path where the JSON file is located.
# Replace this with your JSON file path. 
file_path = 'articles-filtered-and-truncated.json'

# Open the JSON file and load the data.
with open(file_path, 'r', encoding='utf-8') as file:
    messages_dict = json.load(file)

# Start measuring time for embedding the messages
start_time = time.time()

message_lines_embeddings = {}
newline = '\n'

for message in messages_dict:
    # lines = message.split(newline)
    # lines = [line.lower().strip() for line in lines if line.strip()]

    words = split_and_lower(message)
    words = [
        w for w 
        in words 
        if 
            w != 'деяние'
        and w != 'деяния'
        and w != 'преступление'
        and w != 'преступления'
        and w != 'преступлении'
        and w != 'осужденного'
    ]

    pairs = []

    if len(words) > 1:
        i = 1
        while i < len(words):
            pair = f"{words[i-1]} {words[i]}"
            print(pair)
            pairs.append(pair)
            i += 1
    elif len(words) == 1:
        pairs.append(words[0])

    # Enforce 512 token limit on pairs
    pairs = enforce_token_limit(pairs, max_tokens=512, warn=False)

    lines_embeddings = embed(pairs)
    message_lines_embeddings[message] = (lines_embeddings.numpy(), pairs)

# Print message embedding time
message_embedding_time = time.time() - start_time
print(f"Messages embedded in {message_embedding_time} seconds")

# Function to calculate cosine similarity
def rank_messages(query):
    # Enforce 512 token limit on query
    query = enforce_token_limit(query, max_tokens=512, warn=True)

    query_embedding = embed([query]).numpy()
    ranked_messages_scores = {}

    for message, (lines_embeddings, words) in message_lines_embeddings.items():
        similarity_scores = 1 - distance.cdist(query_embedding, lines_embeddings, "cosine")[0]
        total_similarity_score = np.max(similarity_scores)
        
        word_index = [i for i, x in enumerate(similarity_scores) if x == total_similarity_score]
        print(f'{total_similarity_score}: ', words[word_index[0]])

        ranked_messages_scores[message] = total_similarity_score

    ranked_messages = sorted(list(ranked_messages_scores.items()), key = lambda x: -x[1])

    return ranked_messages

# Starts the loop to get queries
while True:
    query = input("Enter the query (or press enter to exit): ").lower()

    # breaks loop if enter is pressed
    if query == '':
        break

    # start measuring query execution time
    start_query_time = time.time()

    print('\n')

    # Ranks messages based on query
    ranked_messages = rank_messages(query)
    for message, score in ranked_messages[:5]:
        print(f'{score}: {message}\n')

    # Print query execution time
    query_time = time.time() - start_query_time
    print(f"Query executed in {query_time} seconds")