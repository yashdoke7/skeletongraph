"""
Pure-Python Okapi BM25 Implementation.

This provides lightweight pseudo-semantic search over LLM function summaries
WITHOUT needing heavyweight vector embedding models locally. 

If a query matches terms tightly correlated to a summary's intent, 
it scores high, resolving ambiguous natural language concepts to hard FQNs.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .inverted_index import tokenize_text


@dataclass
class BM25Model:
    k1: float = 1.5
    b: float = 0.75
    doc_len_avg: float = 0.0
    idf: Dict[str, float] = None
    doc_freqs: List[Dict[str, int]] = None
    doc_lens: List[int] = None
    doc_names: List[str] = None # Stores the FQNs

    def fit(self, corpus: Dict[str, str]) -> None:
        """
        Fit the BM25 model onto a corpus map of {FQN: Summary_Text}.
        """
        self.doc_names = list(corpus.keys())
        self.doc_freqs = []
        self.doc_lens = []
        
        nd = {} # Document frequency of terms
        total_len = 0
        
        for fqn, text in corpus.items():
            tokens = tokenize_text(text)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)
            
            freqs = dict(Counter(tokens))
            self.doc_freqs.append(freqs)
            
            for word in freqs.keys():
                nd[word] = nd.get(word, 0) + 1
                
        num_docs = len(corpus)
        self.doc_len_avg = total_len / num_docs if num_docs > 0 else 0
        
        # Precompute IDF using standard BM25 formula
        self.idf = {}
        for word, freq in nd.items():
            # Standard IDF: ln( (N - n + 0.5) / (n + 0.5) + 1 )
            idf_val = math.log(((num_docs - freq + 0.5) / (freq + 0.5)) + 1)
            self.idf[word] = idf_val

    def get_scores(self, query: str) -> List[Tuple[str, float]]:
        """
        Return unsorted (FQN, score) list for a query.
        """
        if not self.doc_names:
            return []
            
        q_tokens = tokenize_text(query)
        scores = []
        
        for idx in range(len(self.doc_names)):
            score = 0.0
            doc_len = self.doc_lens[idx]
            freqs = self.doc_freqs[idx]
            
            for token in q_tokens:
                if token not in freqs:
                    continue
                tf = freqs[token]
                idf = self.idf.get(token, 0)
                
                # TF normalization
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.doc_len_avg))
                
                score += idf * (numerator / denominator)
                
            scores.append((self.doc_names[idx], score))
            
        return scores

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Return the top_k FQNs by BM25 score.
        """
        scores = self.get_scores(query)
        # Filter 0-scores and sort descending
        ranked = sorted([s for s in scores if s[1] > 0], key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
