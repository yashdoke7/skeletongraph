"""
Bloom filter for O(1) function existence checks.

Before traversing the dependency graph, check if a function name from the
user's prompt actually exists in the codebase. Avoids unnecessary graph lookups.

Uses MurmurHash3 via mmh3 for fast, well-distributed hashing.
For 10,000 functions: ~12KB memory, <1% false positive rate.
"""

from __future__ import annotations

import math
from typing import Iterable


try:
    import mmh3
    _HAS_MMH3 = True
except ImportError:
    import hashlib
    _HAS_MMH3 = False


class BloomFilter:
    """Space-efficient probabilistic set membership test.

    Guarantees:
        - No false negatives: if contains() returns False, the item is definitely absent
        - Possible false positives: if contains() returns True, the item MIGHT be present
          (with probability < fp_rate)
    """

    def __init__(self, expected_items: int = 10000, fp_rate: float = 0.01) -> None:
        """Initialize bloom filter.

        Args:
            expected_items: Expected number of items to insert.
            fp_rate: Desired false positive rate (0.01 = 1%).
        """
        # Calculate optimal bit array size and number of hash functions
        # m = -(n * ln(p)) / (ln(2)^2)
        self._size = max(
            64,
            int(-expected_items * math.log(fp_rate) / (math.log(2) ** 2))
        )
        # k = (m/n) * ln(2)
        self._num_hashes = max(
            1,
            int((self._size / max(expected_items, 1)) * math.log(2))
        )

        self._bits = bytearray(math.ceil(self._size / 8))
        self._count = 0

    def add(self, item: str) -> None:
        """Add an item to the filter."""
        for i in range(self._num_hashes):
            pos = self._hash(item, i) % self._size
            self._bits[pos // 8] |= 1 << (pos % 8)
        self._count += 1

    def add_all(self, items: Iterable[str]) -> None:
        """Batch add items."""
        for item in items:
            self.add(item)

    def contains(self, item: str) -> bool:
        """Check if an item MIGHT be in the filter.

        Returns False → definitely not present.
        Returns True → possibly present (check the actual data structure).
        """
        for i in range(self._num_hashes):
            pos = self._hash(item, i) % self._size
            if not (self._bits[pos // 8] & (1 << (pos % 8))):
                return False
        return True

    def __contains__(self, item: str) -> bool:
        return self.contains(item)

    def __len__(self) -> int:
        return self._count

    @property
    def size_bytes(self) -> int:
        """Memory usage of the bit array."""
        return len(self._bits)

    @property
    def estimated_fp_rate(self) -> float:
        """Estimated current false positive rate based on items inserted."""
        if self._count == 0:
            return 0.0
        # (1 - e^(-kn/m))^k
        exponent = -self._num_hashes * self._count / self._size
        return (1 - math.exp(exponent)) ** self._num_hashes

    def _hash(self, item: str, seed: int) -> int:
        """Generate a hash for the item with the given seed."""
        if _HAS_MMH3:
            return mmh3.hash(item, seed) & 0x7FFFFFFF
        else:
            # Fallback: SHA256-based hashing (slower but no dependency)
            h = hashlib.sha256(f"{seed}:{item}".encode()).hexdigest()
            return int(h[:8], 16)

    def to_bytes(self) -> bytes:
        """Serialize to bytes for disk storage."""
        import struct
        header = struct.pack(">III", self._size, self._num_hashes, self._count)
        return header + bytes(self._bits)

    @classmethod
    def from_bytes(cls, data: bytes) -> BloomFilter:
        """Deserialize from bytes."""
        import struct
        size, num_hashes, count = struct.unpack(">III", data[:12])
        bf = cls.__new__(cls)
        bf._size = size
        bf._num_hashes = num_hashes
        bf._count = count
        bf._bits = bytearray(data[12:])
        return bf
