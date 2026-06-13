import json
import os
from typing import List, Union
import torch

class CharTokenizer:
    """
    A simple character-level tokenizer that maps characters to integer IDs.
    Supports serialization to and from JSON.
    """
    def __init__(self, chars: List[str] = None):
        self.chars = chars if chars is not None else []
        self.vocab_size = len(self.chars)
        self._update_mappings()

    def _update_mappings(self):
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def build_from_text(self, text: str):
        """Build the vocabulary from the unique characters present in a text sample."""
        self.chars = sorted(list(set(text)))
        self._update_mappings()
        print(f"Tokenizer vocabulary built. Size: {self.vocab_size} characters.")

    def encode(self, text: str, return_tensors: str = None) -> Union[List[int], torch.Tensor]:
        """Convert a string of text into integer token IDs."""
        ids = []
        for ch in text:
            # Fallback to a space or unknown character if not in vocabulary
            if ch in self.char2idx:
                ids.append(self.char2idx[ch])
            else:
                # Handle unknown characters gracefully
                if ' ' in self.char2idx:
                    ids.append(self.char2idx[' '])
                elif len(self.char2idx) > 0:
                    ids.append(next(iter(self.char2idx.values())))
                else:
                    raise ValueError("Tokenizer has not been built or loaded with any vocabulary.")
                    
        if return_tensors == "pt":
            return torch.tensor(ids, dtype=torch.long)
        return ids

    def decode(self, ids: Union[List[int], torch.Tensor]) -> str:
        """Convert a list or tensor of token IDs back into a string of text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join([self.idx2char.get(i, "") for i in ids])

    def save(self, filepath: str):
        """Save the tokenizer vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.chars, f, ensure_ascii=False)
        print(f"Tokenizer saved successfully to {filepath}")

    def load(self, filepath: str):
        """Load the tokenizer vocabulary from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.chars = json.load(f)
        self._update_mappings()
        print(f"Tokenizer loaded successfully from {filepath}. Vocabulary size: {self.vocab_size}")
