"""
Handles text preprocessing and generation of text representations (TF-IDF and Embeddings).
"""

import os
import sys

import joblib  # type: ignore
import numpy as np  # type: ignore
import scipy.sparse  # type: ignore
import spacy  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

# --- Configuration ---
SPACY_MODEL = "en_core_web_md"
DATA_DIR = "src/data"
OUTPUT_DIR = "src/vectorizer_output"

# Define the embedding models to generate
EMBEDDING_MODELS_CONFIG = [
    "en_core_web_md",
    "pt_core_news_md",
    "paraphrase-multilingual-mpnet-base-v2",
    "all-MiniLM-L6-v2",
]

# --- Create Output Directory ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# --- Load SpaCy Model ---
nlp = spacy.load(SPACY_MODEL)


# --- Data Loading (copied from corpus_analysis.py for standalone use) ---
def load_documents(data_dir):
    """Loads text documents from a directory."""
    documents = []
    loaded_filenames = []
    abs_data_dir = os.path.abspath(data_dir)
    print(f"Looking for files in: {abs_data_dir}")
    if not os.path.isdir(abs_data_dir):
        print(f"Error: Directory not found at {abs_data_dir}")
        return documents, loaded_filenames

    for filename in os.listdir(abs_data_dir):
        if filename.lower().endswith(".txt"):
            filepath = os.path.join(abs_data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                documents.append(f.read())
            loaded_filenames.append(filename)

    print(f"Found {len(documents)} text files.")
    return documents, loaded_filenames


# --- Text Preprocessing (Task 2.1) ---
def preprocess_text(text):
    """
    Applies preprocessing: lowercasing, lemmatization, removes stopwords,
    punctuation, and non-alphabetic tokens.
    Returns a string of processed tokens.
    """
    # Process text with SpaCy
    doc = nlp(text.lower())

    # Lemmatize and filter tokens
    processed_tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop  # Remove stopwords
        and not token.is_punct  # Remove punctuation
        and not token.is_space  # Remove whitespace tokens
        and token.is_alpha  # Keep only alphabetic tokens
    ]

    # Join tokens back into a single string for TF-IDF
    return " ".join(processed_tokens)


def preprocess_documents(documents):
    """Applies preprocessing to a list of documents."""
    print(f"Preprocessing {len(documents)} documents...")
    preprocessed_docs = [preprocess_text(doc) for doc in documents]
    print("Preprocessing complete.")
    return preprocessed_docs


# --- TF-IDF Representation (Task 2.2) ---
def get_tfidf_representation(processed_docs):
    """
    Generates TF-IDF representation from preprocessed documents.
    Uses default TfidfVectorizer settings initially.
    """
    print("Generating TF-IDF representation...")
    vectorizer = TfidfVectorizer()  # Using default settings
    loaded_tfidf_matrix = vectorizer.fit_transform(processed_docs)
    print(f"TF-IDF matrix shape: {loaded_tfidf_matrix.shape}")
    print("TF-IDF representation complete.")
    # tfidf_matrix is a sparse matrix
    return vectorizer, loaded_tfidf_matrix


# --- Embedding Representation (Task 2.3) ---
def generate_embeddings(raw_documents, model_name):
    """
    Generates document embedding representation using SpaCy's doc.vector.
    Uses the mean of token vectors (default SpaCy behavior).
    Processes raw documents for potentially better context for doc vectors.
    """
    print(f"\n--- Generating embeddings using model: {model_name} ---")
    local_embedding_matrix = None

    # Infer model type and generate embeddings
    if model_name.startswith("en_") or model_name.startswith("pt_"):
        # Assume SpaCy model
        print("  Assuming SpaCy model.")
        nlp_embed = spacy.load(model_name)  # Might raise OSError if not installed
        print(f"  Loaded SpaCy model: {model_name}")
        docs = list(nlp_embed.pipe(raw_documents))
        local_embedding_matrix = np.array([doc.vector for doc in docs])
        if not nlp_embed.has_pipe("tok2vec") and not nlp_embed.has_pipe("transformer"):
            print(
                f"  Warning: SpaCy model '{model_name}' might lack static vectors. "
                "Embeddings might be zero vectors or less meaningful."
            )

    else:
        # Assume Sentence Transformer model
        print("  Assuming Sentence Transformer model.")
        model = SentenceTransformer(model_name)  # Might raise OSError or others
        print(f"  Loaded Sentence Transformer model: {model_name}")
        local_embedding_matrix = model.encode(raw_documents)

    if local_embedding_matrix is None:
        print(f"  Embedding generation failed for model {model_name}.")
        return None

    # --- Validation ---
    zero_vector_indices = [i for i, vec in enumerate(local_embedding_matrix) if np.all(vec == 0)]
    if zero_vector_indices:
        print(f"Warning: Found {len(zero_vector_indices)} docs with zero vectors for {model_name}")

    print(f"  Embedding matrix shape: {local_embedding_matrix.shape}")
    print(f"  Embedding generation complete for model {model_name}.")
    return local_embedding_matrix


# --- Main Execution Example ---
if __name__ == "__main__":
    print("--- Starting Text Representation Generation ---")

    # 1. Load Data
    raw_texts, filenames = load_documents(DATA_DIR)

    if not raw_texts:
        print(f"\nNo documents loaded from '{DATA_DIR}'. Cannot generate representations.")
        sys.exit(1)

    # Store filenames along with representations for later analysis
    filenames_path = os.path.join(OUTPUT_DIR, "filenames.joblib")
    joblib.dump(filenames, filenames_path)
    print(f"Saved filenames list to {filenames_path}")

    # 2. Preprocess Texts (for TF-IDF)
    preprocessed_texts = preprocess_documents(raw_texts)

    # 3. Generate TF-IDF Representation
    tfidf_vectorizer, tfidf_matrix = get_tfidf_representation(preprocessed_texts)

    # 4. Generate Embedding Representations (using raw texts for all models)
    embedding_results = {}  # Store matrices keyed by model name
    print("\n--- Generating Embeddings ---")
    for model_name in EMBEDDING_MODELS_CONFIG:
        embeddings = generate_embeddings(raw_texts, model_name)
        if embeddings is not None:
            embedding_results[model_name] = embeddings  # Use original name as key
            # Save each embedding matrix immediately using the sanitized name
            embeddings_path = os.path.join(OUTPUT_DIR, f"embeddings_{model_name}.npy")
            np.save(embeddings_path, embeddings)
            print(f"  Saved embeddings for '{model_name}' to {embeddings_path}")
        else:
            print(f"  Skipping model '{model_name}' due to generation errors.")

    # --- Output Verification & Saving TF-IDF ---
    print("\n--- Representation Summary & Saving ---")
    print(f"Number of documents processed: {len(raw_texts)}")
    # TF-IDF Saving
    if tfidf_matrix is not None and tfidf_vectorizer is not None:
        print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
        print(f"TF-IDF Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
        tfidf_matrix_path = os.path.join(OUTPUT_DIR, "tfidf_matrix.npz")
        vectorizer_path = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib")
        scipy.sparse.save_npz(tfidf_matrix_path, tfidf_matrix)
        joblib.dump(tfidf_vectorizer, vectorizer_path)
        print(f"  - TF-IDF matrix saved to: {tfidf_matrix_path}")
        print(f"  - TF-IDF vectorizer saved to: {vectorizer_path}")
    else:
        print("TF-IDF generation failed or skipped, not saving.")

    # Embedding Summary (already saved in the loop)
    print("\n--- Embedding Generation Summary ---")
    if not embedding_results:
        print("No embedding representations were successfully generated.")
    else:
        for model_name, matrix in embedding_results.items():
            print(f"Model Name: {model_name}")
            print(f"  - Embedding Matrix shape: {matrix.shape}")
            if matrix.shape[0] > 0:
                print(f"  - Embedding dimension: {matrix.shape[1]}")
            print(f"  - Saved to: embeddings_{model_name}.npy")

    print("\n--- Text Representation Script Finished ---")
