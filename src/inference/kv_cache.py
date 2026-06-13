import torch
from typing import Dict, Tuple, Optional

class KVCache:
    """
    Stores key and value attention history for each transformer layer.
    Allows for efficient O(1) attention steps during token-by-token generation.
    """
    def __init__(self):
        # Maps layer index -> (key_tensor, value_tensor)
        # key_tensor: (Batch, Head, Seq_len, Head_size)
        # value_tensor: (Batch, Head, Seq_len, Head_size)
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Retrieve key-value pair for a given layer index."""
        if layer_idx in self.cache:
            return self.cache[layer_idx]
        return None, None

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Store or update the key-value pair for a given layer index."""
        self.cache[layer_idx] = (k, v)

    def get_seq_len(self, layer_idx: int) -> int:
        """Return the sequence length currently cached for a given layer."""
        if layer_idx in self.cache:
            # key tensor shape is (B, nh, T, hs)
            return self.cache[layer_idx][0].size(-2)
        return 0

    def reset(self):
        """Clear all stored key-value caches."""
        self.cache.clear()
