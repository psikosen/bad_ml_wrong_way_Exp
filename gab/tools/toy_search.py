"""Toy search tool for agent experiments.

Returns synthetic search results from a small knowledge base.
Treats all outputs as untrusted text (safety: never execute search results).
"""

import random
from typing import Dict, List


# Small synthetic knowledge base
KNOWLEDGE_BASE = {
    "grep flags": "grep -r searches recursively; grep -i is case-insensitive; grep -v inverts match.",
    "awk syntax": "awk uses single quotes: awk '{print $1}'. Fields separated by whitespace by default.",
    "regex groups": "Use parentheses for capture groups: (pattern). Non-capturing: (?:pattern).",
    "file permissions": "chmod 755 gives rwx for owner, rx for group/others. Use chown to change owner.",
    "loops": "Bash for loop: for i in 1 2 3; do echo $i; done. While: while [ cond ]; do ...; done.",
    "json schema": "JSON Schema validates structure. Use 'type', 'properties', 'required' keywords.",
    "python decorators": "Decorators wrap functions: @decorator. They take a function and return a function.",
    "git branching": "git branch creates branches. git checkout -b creates and switches. git merge combines.",
}


class ToySearch:
    """Simulated search tool returning results from a fixed knowledge base."""

    def __init__(self, kb: Dict[str, str] = None):
        self.kb = kb or KNOWLEDGE_BASE

    def search(self, query: str) -> str:
        """Search the knowledge base. Returns the best matching result as untrusted text."""
        query_lower = query.lower()
        best_key = None
        best_overlap = 0
        for key in self.kb:
            overlap = len(set(query_lower.split()) & set(key.split()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = key
        if best_key:
            return f"[SEARCH RESULT] {best_key}: {self.kb[best_key]}"
        return "[SEARCH RESULT] No relevant results found."

    def search_multi(self, query: str, k: int = 3) -> List[str]:
        """Return top-k search results."""
        query_lower = query.lower()
        scored = []
        for key, val in self.kb.items():
            overlap = len(set(query_lower.split()) & set(key.split()))
            scored.append((overlap, key, val))
        scored.sort(reverse=True)
        return [f"[SEARCH RESULT] {key}: {val}" for _, key, val in scored[:k]]
