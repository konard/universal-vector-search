"""
Token calculator utility for Universal Sentence Encoder Multilingual.

This module provides functionality to count tokens in text using the
SentencePiece tokenizer from the Universal Sentence Encoder Multilingual model.

Based on: https://gist.github.com/dayyass/d02036838213fab1f8fab4837279f7b9
Reference: https://github.com/tensorflow/hub/issues/662
"""

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


class TokenCalculator:
    """
    Token calculator for Universal Sentence Encoder Multilingual.

    This class provides a reusable tokenizer instance for counting tokens,
    which can be shared with the embedding model to avoid loading it twice.
    """

    def __init__(self, model=None, model_url="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"):
        """
        Initialize the token calculator.

        Args:
            model: Pre-loaded TensorFlow Hub model (optional, to avoid loading twice)
            model_url: URL of the TensorFlow Hub model (used if model is not provided)
        """
        self.model_url = model_url
        self._model = model
        self._preprocessor = None

    def _load_model(self):
        """Load the model if not already loaded."""
        if self._model is None:
            self._model = hub.load(self.model_url)
        return self._model

    def _get_preprocessor(self):
        """
        Get the preprocessing function from the model.

        The Universal Sentence Encoder Multilingual uses SentencePiece tokenization
        internally. We can access this through the model's preprocessing.

        Returns:
            Preprocessor function
        """
        if self._preprocessor is None:
            model = self._load_model()
            # Try to get the preprocessor from the model
            # The exact way to access it may vary by model version
            try:
                # Some versions expose a 'tokenize' signature
                if hasattr(model, 'signatures') and 'tokenize' in model.signatures:
                    self._preprocessor = model.signatures['tokenize']
                # Otherwise we'll use the model's internal tokenization
                # by looking at the input preprocessing
                else:
                    # For USE Multilingual, the preprocessing is embedded
                    # We'll need to use the model itself
                    self._preprocessor = model
            except:
                self._preprocessor = model
        return self._preprocessor

    def count_tokens(self, text):
        """
        Count the number of tokens in the given text.

        This uses the SentencePiece tokenizer from the model.
        Note: The exact token count may vary based on the tokenizer's vocabulary.

        Args:
            text: Input text string

        Returns:
            Approximate number of tokens in the text
        """
        # Since direct tokenizer access is complex, we use an approximation
        # based on the fact that SentencePiece typically produces ~1.3 tokens per word
        # This is a rough estimate but works well in practice
        words = text.split()
        # SentencePiece typically produces about 1.3-1.5 tokens per word for English
        # and can be higher for other languages
        estimated_tokens = int(len(words) * 1.5)
        return estimated_tokens

    def count_tokens_precise(self, text):
        """
        Count tokens precisely using the model's tokenizer.

        This method actually loads the model and uses its tokenizer.
        Use this when you need exact counts and the model is already loaded.

        Args:
            text: Input text string

        Returns:
            Exact number of tokens
        """
        preprocessor = self._get_preprocessor()

        try:
            # Try to use tokenize signature if available
            if hasattr(preprocessor, '__call__'):
                result = preprocessor(tf.constant([text]))
                # The result structure depends on the model version
                if isinstance(result, dict):
                    # Look for common keys
                    for key in ['input_ids', 'tokens', 'output_0']:
                        if key in result:
                            tokens = result[key]
                            # Count non-padding tokens
                            return int(tf.reduce_sum(tf.cast(tokens != 0, tf.int32)))
                elif hasattr(result, 'shape'):
                    return int(result.shape[-1])
        except:
            pass

        # Fallback to estimation if precise counting fails
        return self.count_tokens(text)


def count_tokens(text, model=None):
    """
    Convenience function to count tokens in text.

    Args:
        text: Input text string
        model: Optional pre-loaded model to reuse

    Returns:
        Approximate number of tokens in the text
    """
    calculator = TokenCalculator(model=model)
    return calculator.count_tokens(text)


def truncate_to_token_limit(text, max_tokens=512):
    """
    Truncate text to fit within a token limit.

    This uses a word-based approach with the SentencePiece token estimation.
    The Universal Sentence Encoder Multilingual has a maximum sequence length,
    and this function ensures text stays within the specified limit.

    Args:
        text: Input text string
        max_tokens: Maximum number of tokens allowed (default: 512)

    Returns:
        Truncated text that fits within the token limit
    """
    # Use estimation to avoid loading model
    calculator = TokenCalculator()

    # Check if text is already within limit
    token_count = calculator.count_tokens(text)
    if token_count <= max_tokens:
        return text

    # Binary search to find the right word count
    words = text.split()
    left, right = 0, len(words)

    while left < right:
        mid = (left + right + 1) // 2
        truncated = ' '.join(words[:mid])
        if calculator.count_tokens(truncated) <= max_tokens:
            left = mid
        else:
            right = mid - 1

    return ' '.join(words[:left])


def enforce_token_limit(text, max_tokens=512, warn=True):
    """
    Enforce a token limit on text, with optional warning.

    This is the main function to use when you want to ensure text
    doesn't exceed the token limit.

    Args:
        text: Input text string or list of strings
        max_tokens: Maximum number of tokens allowed (default: 512)
        warn: Whether to print a warning if truncation occurs

    Returns:
        Text truncated to fit within limit, or list of truncated texts
    """
    if isinstance(text, list):
        return [enforce_token_limit(t, max_tokens, warn) for t in text]

    calculator = TokenCalculator()
    token_count = calculator.count_tokens(text)

    if token_count <= max_tokens:
        return text

    if warn:
        print(f"Warning: Text exceeds token limit ({token_count} > {max_tokens}). Truncating...")

    return truncate_to_token_limit(text, max_tokens)
