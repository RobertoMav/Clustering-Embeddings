#!/usr/bin/env python3
"""
Corpus Analysis Script for NER Assignment
Analyzes the legal documents corpus and extracts basic statistics.
"""

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd


def read_corpus_files(data_dir: str = "data") -> List[str]:
    """Read all text files from the corpus directory."""
    texts = []
    data_path = Path(data_dir)

    for file_path in sorted(data_path.glob("texto*.txt")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return texts


def tokenize_text(text: str) -> List[str]:
    """Simple tokenization by splitting on whitespace and punctuation."""
    # Basic tokenization - split on whitespace and punctuation
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


def count_sentences(text: str) -> int:
    """Count sentences in text using basic sentence splitting."""
    # Simple sentence counting based on sentence-ending punctuation
    sentences = re.split(r"[.!?]+", text)
    # Remove empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)


def analyze_corpus(texts: List[str]) -> Dict:
    """Analyze the corpus and return statistics."""
    total_texts = len(texts)
    total_sentences = 0
    all_tokens = []
    all_types = set()

    for text in texts:
        sentences = count_sentences(text)
        total_sentences += sentences

        tokens = tokenize_text(text)
        all_tokens.extend(tokens)
        all_types.update(tokens)

    total_tokens = len(all_tokens)
    total_types = len(all_types)

    avg_sentences_per_text = total_sentences / total_texts if total_texts > 0 else 0
    avg_tokens_per_text = total_tokens / total_texts if total_texts > 0 else 0

    # Get most common tokens
    token_counter = Counter(all_tokens)
    most_common_tokens = token_counter.most_common(20)

    stats = {
        "total_texts": total_texts,
        "total_sentences": total_sentences,
        "total_tokens": total_tokens,
        "total_types": total_types,
        "avg_sentences_per_text": round(avg_sentences_per_text, 2),
        "avg_tokens_per_text": round(avg_tokens_per_text, 2),
        "type_token_ratio": round(total_types / total_tokens, 4) if total_tokens > 0 else 0,
        "most_common_tokens": most_common_tokens,
    }

    return stats


def print_corpus_statistics(stats: Dict):
    """Print corpus statistics in a formatted way."""
    print("=" * 60)
    print("CORPUS ANALYSIS REPORT")
    print("=" * 60)
    print(f"Total number of texts: {stats['total_texts']}")
    print(f"Total number of sentences: {stats['total_sentences']}")
    print(f"Total number of tokens: {stats['total_tokens']}")
    print(f"Total number of types (unique tokens): {stats['total_types']}")
    print(f"Average sentences per text: {stats['avg_sentences_per_text']}")
    print(f"Average tokens per text: {stats['avg_tokens_per_text']}")
    print(f"Type-Token Ratio: {stats['type_token_ratio']}")
    print()
    print("Most common tokens:")
    for token, count in stats["most_common_tokens"]:
        print(f"  {token}: {count}")
    print("=" * 60)


def main():
    """Main function to run corpus analysis."""
    print("Starting corpus analysis...")

    # Read corpus
    texts = read_corpus_files()
    print(f"Successfully read {len(texts)} text files.")

    # Analyze corpus
    stats = analyze_corpus(texts)

    # Print results
    print_corpus_statistics(stats)

    # Save results to CSV for later use
    stats_df = pd.DataFrame(
        [
            {"metric": k, "value": v if not isinstance(v, list) else str(v)}
            for k, v in stats.items()
        ]
    )

    stats_df.to_csv("corpus_statistics.csv", index=False)
    print("Corpus statistics saved to corpus_statistics.csv")


if __name__ == "__main__":
    main()
