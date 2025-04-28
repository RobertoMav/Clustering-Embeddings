import json
import os
import re
from typing import Any

import joblib  # type: ignore
import numpy as np  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

# --- Configuration ---
CLUSTERING_OUTPUT_DIR = "src/clustering_output"
VECTORIZER_OUTPUT_DIR = "src/vectorizer_output"
DATA_DIR = "src/data"
CENTRAL_DOC_SNIPPET_LINES = 5  # Number of lines to show from central doc
TOP_N_WORDS = 10  # Number of top words to find per cluster
TOP_N_DOCS = 5  # Number of top documents to find per cluster


# Updated chosen clustering results
CHOSEN_RESULTS = [
    {"key": "kmeans_paraphrase-multilingual-mpnet-base-v2", "type": "kmeans_embedding"},
    # Note: type changed to reflect embedding usage for HDBSCAN as well
    {"key": "hdbscan_paraphrase-multilingual-mpnet-base-v2", "type": "hdbscan_embedding"},
]

# --- Helper Functions ---


def load_data(key, result_type):
    """Loads necessary data for a given clustering result."""
    print(f"--- Loading data for {key} ({result_type}) ---")
    labels_path = os.path.join(CLUSTERING_OUTPUT_DIR, f"{key}_labels.joblib")
    filenames_path = os.path.join(VECTORIZER_OUTPUT_DIR, "filenames.joblib")

    # Load labels and filenames
    labels = joblib.load(labels_path)
    filenames = joblib.load(filenames_path)
    print(f"Loaded labels ({len(labels)}) and filenames ({len(filenames)}).")

    # Load embeddings
    model_name = key.split("_", 1)[1]
    embeddings_path = os.path.join(VECTORIZER_OUTPUT_DIR, f"embeddings_{model_name}.npy")
    data_matrix = np.load(embeddings_path)
    print(f"Loaded embeddings matrix: {embeddings_path} ({data_matrix.shape})")

    # Load the sentence transformer model
    print(f"Loading Sentence Transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    print("Model loaded.")

    # Vectorizer is no longer loaded
    vectorizer = None

    # Load raw documents
    raw_docs = {}
    abs_data_dir = os.path.abspath(DATA_DIR)
    print(f"Loading raw documents from: {abs_data_dir}")
    for filename in filenames:
        filepath = os.path.join(abs_data_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_docs[filename] = f.read()
        except Exception as e:
            print(f"Warning: Could not read file {filepath}: {e}")
            raw_docs[filename] = ""  # Assign empty string if read fails
    print(f"Loaded content for {len(raw_docs)} documents.")

    print("Data loading complete.")
    return labels, filenames, data_matrix, vectorizer, raw_docs, model  # Return model


def get_cluster_centroid(cluster_embeddings):
    """Calculates the mean embedding vector (centroid) for a cluster."""
    if cluster_embeddings.shape[0] == 0:
        return None  # Handle empty clusters
    return cluster_embeddings.mean(axis=0)


def get_top_documents(embedding_matrix, labels, filenames, cluster_id, centroid, n_docs):
    """Finds the top N documents closest to the cluster centroid."""
    cluster_indices = np.where(labels == cluster_id)[0]
    if len(cluster_indices) == 0:
        return [], []  # Handle empty clusters

    cluster_embeddings = embedding_matrix[cluster_indices]

    # Calculate cosine similarity (higher is better, hence closer)
    similarities = cosine_similarity(cluster_embeddings, centroid.reshape(1, -1)).flatten()

    # Get indices sorted by similarity (descending)
    closest_indices_in_cluster = np.argsort(similarities)[::-1]

    # Get the top N original indices and filenames
    top_original_indices = [cluster_indices[i] for i in closest_indices_in_cluster[:n_docs]]
    top_filenames = [str(filenames[idx]) for idx in top_original_indices]

    return top_filenames, top_original_indices


def get_top_words_for_cluster(cluster_docs_text, centroid, model, n_words):
    """
    Finds the top N words whose embeddings are closest to the cluster centroid.
    This encodes individual words using the sentence transformer.
    """
    if not cluster_docs_text or centroid is None:
        return []

    # Simple word extraction (lowercase, alphanumeric) - might need refinement
    words = re.findall(r"\b\w+\b", cluster_docs_text.lower())
    if not words:
        return []

    # Get unique words
    unique_words = list(set(words))

    # Encode unique words - batch encoding is more efficient
    # Filter out very short words if desired (e.g., len > 2)
    unique_words_filtered = [w for w in unique_words if len(w) > 2]
    if not unique_words_filtered:
        return []

    print(f"    Encoding {len(unique_words_filtered)} unique words for similarity check...")
    try:
        word_embeddings = model.encode(unique_words_filtered, show_progress_bar=False)
    except Exception as e:
        print(f"    Error encoding words: {e}")
        return []  # Return empty if encoding fails

    # Calculate cosine similarity between word embeddings and centroid
    similarities = cosine_similarity(word_embeddings, centroid.reshape(1, -1)).flatten()

    # Get indices of top N words
    closest_word_indices = np.argsort(similarities)[::-1][:n_words]

    # Get the actual top words
    top_words = [unique_words_filtered[i] for i in closest_word_indices]

    return top_words


def get_document_snippet(filename, raw_docs, num_lines):
    """Gets the first few lines of a document's text (simplified)."""
    # Assume filename exists in raw_docs
    doc_content = raw_docs.get(filename, "")  # Use .get for safety
    lines = doc_content.splitlines()
    # Corrected join string to use escaped newline
    return "\n".join(lines[:num_lines])


# --- Main Analysis Logic ---

if __name__ == "__main__":
    print("--- Starting Cluster Analysis and Labeling ---")

    analysis_results = {}

    for result_info in CHOSEN_RESULTS:
        key = result_info["key"]
        result_type = result_info["type"]  # kmeans_embedding or hdbscan_embedding

        # Load data (now includes model)
        labels, filenames, data_matrix, _, raw_docs, model = load_data(key, result_type)

        unique_labels = sorted(list(set(labels)))
        cluster_summaries = {}

        print(f"\n--- Analyzing Clusters for: {key} ---")

        for cluster_id in unique_labels:
            # Define expected structure for summary_data with default types
            summary_data: dict[str, Any] = {
                "count": 0,
                "is_noise": False,
                "central_doc_filename": None,
                "central_doc_snippet": None,
                "top_words_cluster": [],
                "top_words_central_doc": [],
            }

            # Handle potential noise cluster in HDBSCAN (-1)
            if cluster_id == -1 and result_type == "hdbscan_embedding":
                cluster_label = "Noise"
                count = np.count_nonzero(labels == -1)
                print(f"\nCluster {cluster_label} ({count} documents)")
                summary_data["count"] = count
                summary_data["is_noise"] = True
            else:
                cluster_label = f"Cluster {cluster_id}"
                cluster_indices = np.where(labels == cluster_id)[0]
                count = len(cluster_indices)
                print(f"\n{cluster_label} ({count} documents):")

                summary_data["count"] = count
                summary_data["is_noise"] = False

                if count > 0:
                    cluster_embeddings = data_matrix[cluster_indices]
                    centroid = get_cluster_centroid(cluster_embeddings)

                    if centroid is not None:
                        # Find top documents (needed to identify the single most central one)
                        top_docs, top_indices = get_top_documents(
                            data_matrix,
                            labels,
                            filenames,
                            cluster_id,
                            centroid,
                            1,  # Only need the top 1
                        )
                        # summary_data["top_documents"] = top_docs # REMOVED

                        # Get info for the most central document
                        if top_docs:
                            central_filename = top_docs[0]
                            snippet = get_document_snippet(
                                central_filename, raw_docs, CENTRAL_DOC_SNIPPET_LINES
                            )
                            summary_data["central_doc_filename"] = central_filename
                            summary_data["central_doc_snippet"] = snippet

                            # Find top words for the ENTIRE cluster
                            cluster_text = " ".join(
                                [raw_docs.get(filenames[i], "") for i in cluster_indices]
                            )
                            top_words_cluster = get_top_words_for_cluster(
                                cluster_text, centroid, model, TOP_N_WORDS
                            )
                            summary_data["top_words_cluster"] = top_words_cluster
                            print(
                                f"  Top {len(top_words_cluster)} Words (Cluster): {', '.join(top_words_cluster)}"
                            )

                            # Find top words for the CENTRAL document
                            central_doc_text = raw_docs.get(central_filename, "")
                            top_words_central_doc = get_top_words_for_cluster(
                                central_doc_text, centroid, model, TOP_N_WORDS
                            )
                            summary_data["top_words_central_doc"] = top_words_central_doc
                            print(
                                f"  Top {len(top_words_central_doc)} Words (Central Doc): {', '.join(top_words_central_doc)}"
                            )

                        else:
                            print("  Could not find a central document for this cluster.")
                            # summary_data already has defaults [] for word lists

                    else:
                        print("  Could not calculate centroid (empty or invalid cluster).")
                        # summary_data already has defaults [] for word lists

            # Convert cluster_id (potentially numpy int) to standard Python int for JSON key
            summary_key = int(cluster_id)
            cluster_summaries[summary_key] = summary_data
        analysis_results[key] = cluster_summaries

    # Save the analysis results
    # import json # Already imported earlier

    analysis_output_path = os.path.join(CLUSTERING_OUTPUT_DIR, "cluster_analysis_summary.json")
    # Ensure the output directory exists
    os.makedirs(CLUSTERING_OUTPUT_DIR, exist_ok=True)
    with open(analysis_output_path, "w", encoding="utf-8") as f:
        # Use a custom serializer for numpy types if they sneak in, although casting should prevent it
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        json.dump(analysis_results, f, indent=4, ensure_ascii=False, cls=NpEncoder)
    print(f"\nSaved detailed analysis summary to {analysis_output_path}")

    print("\n--- Cluster Analysis Finished ---")
