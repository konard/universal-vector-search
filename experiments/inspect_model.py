"""
Inspect the Universal Sentence Encoder Multilingual model structure.
"""

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

print("Loading model...")
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

print("\nModel type:", type(model))
print("\nAvailable signatures:")
for sig_name in model.signatures.keys():
    print(f"  - {sig_name}")

print("\nModel attributes:")
for attr in dir(model):
    if not attr.startswith('_'):
        print(f"  - {attr}")

# Try to inspect the default signature
print("\nDefault signature:")
try:
    default_sig = model.signatures['serving_default']
    print("Input names:", list(default_sig.structured_input_signature[1].keys()))
    print("Output names:", list(default_sig.structured_outputs.keys()))
except Exception as e:
    print(f"Error inspecting default signature: {e}")

# Test basic embedding
print("\nTesting basic embedding:")
test_text = "Hello, world!"
try:
    embedding = model([test_text])
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding type: {type(embedding)}")
except Exception as e:
    print(f"Error: {e}")
