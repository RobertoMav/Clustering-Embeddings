# Text Representation and Clustering (NLP Assignment)

## a. Introduction
This project clusters 30 legal procedural documents written in Portuguese. We compare two representation methods (TF-IDF and word embeddings) and two clustering algorithms (K-Means and HDBSCAN) to identify meaningful groups without labeled data.

## b. Corpus and Preprocessing
- **Data:** 30 `.txt` files (117.75 KB total).
- **Statistics:**
  - Total Tokens: 17 560
  - Unique Types: 2 482
  - Avg. Tokens/Doc: 585.33
  - Min/Max Tokens: 384 / 864
- **Preprocessing Steps (in `text_representation.py`):**
  1. Lowercase
  2. SpaCy tokenization & lemmatization (`en_core_web_md`)
  3. Remove stopwords, punctuation, non-alphabetic tokens
- **Note:** Documents are Portuguese; using the English model may affect quality.

## c. Representation by BOW (TF-IDF)
- **Input:** Preprocessed texts (joined lemmas).
- **Method:** Scikit-learn `TfidfVectorizer` (default params).
- **Output:** Sparse matrix (30 × 2 468), saved to `vectorizer_output/tfidf_matrix.npz` and `.joblib`.

## d. Representation by Word Embeddings
- **Models Used:**
  - SpaCy: `en_core_web_md`, `pt_core_news_md` (doc.vector)
  - SentenceTransformer: `paraphrase-multilingual-mpnet-base-v2`, `all-MiniLM-L6-v2`
- **Chosen Model:** `paraphrase-multilingual-mpnet-base-v2` (768 d).
- **Aggregation:** Default average of token embeddings (`model.encode(raw_documents)`).
- **Output:** Dense matrices saved under `vectorizer_output/embeddings_{model}.npy`.

## e. Clustering
- **Algorithms:** K-Means and HDBSCAN (`min_cluster_size=2`).
- **Parameter Selection:**
  - *K-Means:* Elbow plot for `paraphrase-multilingual-mpnet-base-v2` suggested K≈6 (see `elbow_embeddings_paraphrase-multilingual-mpnet-base-v2.png`); silhouette analysis returned K=10 (score 0.4320).  
  - *HDBSCAN:* DBCV-optimal clusters for `paraphrase-multilingual-mpnet-base-v2`: 6 clusters, 1 noise (DBCV 0.3846).
- **Data:** TF-IDF and all four embedding matrices.

## f. Comparison and Analysis of Results
**Summary (from `clustering_output/clustering_summary.csv`):**

| Algorithm | Representation                                    | #Clusters | Noise | Silhouette | DBCV   |
|-----------|---------------------------------------------------|-----------|-------|------------|--------|
| K-Means   | TF-IDF                                            | 15        | 0     | 0.2627     | –      |
| HDBSCAN   | TF-IDF                                            | 6         | 4     | –          | 0.1554 |
| K-Means   | Embeddings (`paraphrase-multilingual-mpnet-base-v2`) | 10        | 0     | 0.4320     | –      |
| HDBSCAN   | Embeddings (`paraphrase-multilingual-mpnet-base-v2`) | 6         | 1     | –          | 0.3846 |

- **Visualization:** 3-PC PCA explained ~61% variance; clusters are moderately separated but overlap in some dimensions.
- **Subjective Labels:** Six semantic clusters defined via top-10 words from `cluster_analysis_summary.json` (e.g., Cluster 0: [word1,…,word10], …).
- **Best Configuration:** HDBSCAN on `paraphrase-multilingual-mpnet-base-v2`, yielding compact, interpretable clusters (6 groups, 1 noise).

## g. Conclusion
HDBSCAN with the multilingual MPNet embeddings provided the most cohesive clusters (highest DBCV, minimal noise). The main challenges were evaluating unlabeled data and using an English model on Portuguese texts. Future work: retrain preprocessing with a Portuguese SpaCy model and fine-tune clustering parameters for deeper insights.
