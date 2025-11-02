import json
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
from scipy.spatial import distance
import time
import re
import math
from collections import defaultdict
from token_calculator import enforce_token_limit

def make_doc_words_count_dict(doc_words):
  """ 
  Make a dictionary that contains count of each word in a document.
  doc_words - the list of words of the text document.
  """

  doc_word_count_dict = defaultdict(int)
  for word in doc_words:
    doc_word_count_dict[word] += 1
  return doc_word_count_dict

def make_word_docs_count_dict(docs_words):
  """ 
  Make a dictionary that contains count of docs for each word in these docs.
  docs_words - a list of documents (each document represented as a list of words).
  """

  word_docs_count_dict = defaultdict(int)
  for doc in docs_words:
    for word in set(doc):
      word_docs_count_dict[word] += 1
  return word_docs_count_dict

def compute_tf(doc_word_count_dict, doc_words):
  """ 
  Compute term frequency.
  doc_word_count_dict - a dictionary with words and their counts.
  doc_words - the document text.
  """

  tf_dict = {}
  total_words = len(doc_words)
  for word, count in doc_word_count_dict.items():
    tf_dict[word] = count / float(total_words)
  return tf_dict

def compute_idf(docs_words):
  """
  Compute inverse document frequency.
  docs_words - a list of documents (each document represented as a list of words).
  """

  idf_dict = {}
  total_docs = len(docs_words)
  
  word_docs_count_dict = make_word_docs_count_dict(docs_words)

  # Compute IDF for each word
  for word, count in word_docs_count_dict.items():
    idf_dict[word] = math.log(total_docs / float(count))
    
  return idf_dict

def compute_tfidf(tf, idf):
  """
  Compute TF*IDF.
  tf - term frequency of words in a document.
  idf - inverse document frequency of words.
  """
  
  tfidf = {}
  for word, tf_val in tf.items():
    tfidf[word] = tf_val * idf[word]
  return tfidf

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

docs_words = [split_and_lower(message) for message in messages_dict]
idf = compute_idf(docs_words)
# print('idf[кража]', idf['кража'])
# print('idf[грабеж]', idf['грабеж'])

i = 0
for words in docs_words:
    # lines = message.split(newline)
    # lines = [line.lower().strip() for line in lines if line.strip()]

    # words = split_and_lower(message)
    # words = [
    #     w for w 
    #     in words 
    #     if 
    #         w != 'деяние'
    #     and w != 'деяния'
    #     and w != 'преступление'
    #     and w != 'преступления'
    #     and w != 'преступлении'
    #     and w != 'осужденного'
    # ]

    doc_word_count_dict = make_doc_words_count_dict(words)
    term_frequency = compute_tf(doc_word_count_dict, words)
    tfidf = compute_tfidf(term_frequency, idf)
    # print('tfidf[кража]', tfidf.get('кража', None))
    # print('tfidf[грабеж]', tfidf.get('грабеж', None))
    # print('tfidf', tfidf)

    # Enforce 512 token limit on words
    words = enforce_token_limit(words, max_tokens=512, warn=False)

    lines_embeddings = embed(words)
    message_lines_embeddings[messages_dict[i]] = (lines_embeddings.numpy(), words, tfidf)
    i += 1

# Print message embedding time
message_embedding_time = time.time() - start_time
print(f"Messages embedded in {message_embedding_time} seconds")

# Function to calculate cosine similarity
def rank_messages(query):
    # Enforce 512 token limit on query
    query = enforce_token_limit(query, max_tokens=512, warn=True)

    query_embedding = embed([query]).numpy()
    ranked_messages_scores = {}

    for message, (lines_embeddings, words, tfidf) in message_lines_embeddings.items():
        similarity_scores = 1 - distance.cdist(query_embedding, lines_embeddings, "cosine")[0]
        # total_similarity_score = np.max(similarity_scores)

        i = 0
        for word in words:
            importance = idf[word]
            # print('word', word)
            # print('importance', importance)
            # print('similarity_score', similarity_scores[i])
            similarity_scores[i] *= importance
            # print('updated similarity_score', similarity_scores[i])
            i += 1
        total_similarity_score = np.max(similarity_scores)

        # word_index = [i for i, x in enumerate(similarity_scores) if x == total_similarity_score]
        # print(f'{total_similarity_score}: ', words[word_index[0]])

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

    # print('\n')

    # Ranks messages based on query
    ranked_messages = rank_messages(query)
    for message, score in ranked_messages[:5]:
        # print(f'{score}: {message}\n')
        print(f'{score}: {message.strip().partition(newline)[0]}')

    # Print query execution time
    query_time = time.time() - start_query_time
    print(f"Query executed in {query_time} seconds")