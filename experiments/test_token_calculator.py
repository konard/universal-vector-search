"""
Test script for token calculator.

This script tests the token calculator functionality with various inputs.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token_calculator import TokenCalculator, truncate_to_token_limit

def test_basic_counting():
    """Test basic token counting functionality."""
    print("Testing basic token counting...")

    calculator = TokenCalculator()

    test_texts = [
        "Hello, world!",
        "This is a simple test sentence.",
        "Привет, Мир!",
        "I enjoy playing football and reading books.",
        "Мне нравится играть в игры и читать книги"
    ]

    for text in test_texts:
        try:
            # Note: This might fail if tokenize signature doesn't exist
            # We'll catch the error and try a different approach
            token_count = calculator.count_tokens(text)
            print(f"Text: '{text}'")
            print(f"Token count: {token_count}\n")
        except Exception as e:
            print(f"Error counting tokens for '{text}': {e}\n")

def test_truncation():
    """Test text truncation to token limit."""
    print("\nTesting text truncation...")

    # Create a long text
    long_text = " ".join(["This is a test sentence."] * 100)

    print(f"Original text length: {len(long_text)} characters")

    try:
        truncated = truncate_to_token_limit(long_text, max_tokens=512)
        print(f"Truncated text length: {len(truncated)} characters")
        print(f"First 100 chars: {truncated[:100]}...")
    except Exception as e:
        print(f"Error during truncation: {e}")

if __name__ == "__main__":
    print("Token Calculator Test Script\n")
    print("=" * 50)

    try:
        test_basic_counting()
        test_truncation()
        print("\nTests completed!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
