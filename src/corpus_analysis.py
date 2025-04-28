"""
This script analyzes a corpus of text documents and calculates basic statistics.
"""

import math  # Added for disk size formatting
import os
from collections import Counter

import pandas as pd
import spacy

nlp = spacy.load("en_core_web_md")
DATA_DIR = "src/data"


def load_documents(data_dir):
    """Loads text documents from a directory."""
    documents = []
    filenames = []
    abs_data_dir = os.path.abspath(data_dir)
    print(f"Looking for files in: {abs_data_dir}")
    if not os.path.isdir(abs_data_dir):
        print(f"Error: Directory not found at {abs_data_dir}")
        return documents, filenames, 0  # Return 0 for disk size if dir not found

    total_disk_size = 0
    for filename in os.listdir(abs_data_dir):
        # Check if the file is a text file (e.g., ends with .txt)
        if filename.lower().endswith(".txt") and not filename.startswith(
            "."
        ):  # Ignore hidden files
            filepath = os.path.join(abs_data_dir, filename)
            # Simplified: Assume UTF-8 encoding works
            with open(filepath, "r", encoding="utf-8") as file:
                documents.append(file.read())
            filenames.append(filename)
            total_disk_size += os.path.getsize(filepath)  # Add file size

    print(f"Found {len(documents)} text files.")
    print(f"Total disk size of .txt files: {format_size(total_disk_size)}")  # Print formatted size
    return documents, filenames, total_disk_size  # Return size


def analyze_corpus(documents):
    """
    Analyzes the corpus to calculate basic statistics, including min/max tokens.
    Tokens exclude punctuation and whitespace.
    """
    num_documents = len(documents)
    if num_documents == 0:
        print("No documents loaded, cannot analyze.")
        return None

    total_tokens = 0
    all_tokens = []
    doc_token_lengths = []  # Store length of each doc

    # Process documents using SpaCy for tokenization
    print("Tokenizing documents with SpaCy...")
    docs = list(nlp.pipe(documents))
    print("Tokenization complete.")

    print("Calculating statistics...")
    for doc in docs:
        # Filter out punctuation and whitespace tokens
        tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
        doc_len = len(tokens)
        doc_token_lengths.append(doc_len)  # Add length to list
        total_tokens += doc_len
        all_tokens.extend(tokens)

    # Calculate types (unique tokens) - case-insensitive
    all_tokens_lower = [t.lower() for t in all_tokens]
    word_counts = Counter(all_tokens_lower)
    num_types = len(word_counts)

    # Calculate average, min, max tokens per document
    avg_tokens_per_doc = total_tokens / num_documents if num_documents > 0 else 0
    min_tokens_per_doc = min(doc_token_lengths) if doc_token_lengths else 0
    max_tokens_per_doc = max(doc_token_lengths) if doc_token_lengths else 0

    stats = {
        "Number of Documents": num_documents,
        "Total Tokens": total_tokens,
        "Total Types": num_types,
        "Average Tokens per Document": avg_tokens_per_doc,
        "Min Tokens per Document": min_tokens_per_doc,
        "Max Tokens per Document": max_tokens_per_doc,
        # Disk size will be added separately from load_documents result
    }
    print("Statistics calculation complete.")
    return stats


def format_size(size_bytes):
    """Formats size in bytes to human-readable string."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    # Load documents and get disk size
    texts, text_filenames, total_disk_size_bytes = load_documents(DATA_DIR)

    if texts:
        # Analyze corpus
        corpus_stats = analyze_corpus(texts)

        if corpus_stats:
            # Add disk size to stats
            corpus_stats["Total Disk Size"] = total_disk_size_bytes

            # Print statistics
            print("\n--- Corpus Statistics (English) ---")
            for key, value in corpus_stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.2f}")
                elif key == "Total Disk Size":
                    print(f"{key}: {format_size(value)}")
                else:
                    print(f"{key}: {value}")

            # Save English stats to CSV
            stats_df = pd.DataFrame([corpus_stats])
            # Use a temporary dict to format size for CSV if needed, or save bytes
            stats_df_save = stats_df.copy()
            stats_df_save["Total Disk Size Formatted"] = format_size(total_disk_size_bytes)
            stats_df_save.to_csv("corpus_stats.csv", index=False)
            print("\nStatistics saved to corpus_stats.csv")
