# Report: Text Representation and Clustering of Legal Documents

## a. Introduction
This project clusters 30 legal procedural documents (Portuguese `.txt` files) using two text representations (TF-IDF and word embeddings) and two clustering algorithms (K-Means and HDBSCAN) to discover thematic groups without supervision.

## b. Corpus and Preprocessing
- **Data Characteristics**:
  - 30 documents, total disk size 117.75 KB
  - Total tokens (excl. punctuation/whitespace): 17 560
  - Unique types (case-insensitive): 2 482
  - Tokens per document: avg 585.33 (min 384, max 864)

- **Preprocessing Steps** (in `text_representation.py`):
  1. Lowercasing to unify token forms.
  2. SpaCy tokenization & lemmatization (`en_core_web_md`) to reduce inflectional variants.
  3. Removal of stopwords to eliminate high-frequency low-value words.
  4. Removal of punctuation and whitespace tokens to focus on content words.
  5. Filtering non-alphabetic tokens to exclude numbers/symbols.

- **Decisions & Justification**:
  - Lemmatization and stopword removal reduce vocabulary size and noise.
  - Non-alphabetic filtering targets legal jargon (primarily words).
  - Used English model for convenience; ideally a Portuguese model (`pt_core_news_md`) better fits the corpus.

## c. Representation by BOW/TF-IDF
- **Construction**:
  - Input: preprocessed texts joined as lemma strings.
  - Tool: `sklearn.feature_extraction.text.TfidfVectorizer` with default settings (L2 norm, smooth_idf).
  - Output: sparse TF-IDF matrix (30×2 468).

- **Tests & Limitations**:
  - No parameter grid search or document-frequency thresholds were tested; defaults were assumed sufficient for baseline.
  - A next step would tune `min_df`, `max_df`, or `ngram_range` to improve feature selection.

## d. Representation by Word Embeddings
- **Models Evaluated**:
  - SpaCy: `en_core_web_md` (96 d), `pt_core_news_md` (300 d)
  - SentenceTransformer: `paraphrase-multilingual-mpnet-base-v2` (768 d), `all-MiniLM-L6-v2` (384 d)
- **Aggregation**: mean pooling of token embeddings (`model.encode(raw_documents)` or `doc.vector`).
  - Chosen for its simplicity, computational efficiency and consistent performance on small datasets.
  - Alternative strategies that could be explored include:
    - Max pooling of token vectors to capture the most salient features.
    - Concatenation of mean and max pooled embeddings for richer representations.
    - TF-IDF weighted averaging to emphasize more informative tokens.
    - Advanced techniques such as Smooth Inverse Frequency (SIF) or attention-based pooling.
- **Output**: four dense matrices saved under `src/vectorizer_output/embeddings_{model}.npy`.

- **Tests & Selection**:
  - No hyperparameter tuning per model.
  - Models were compared via clustering performance (Silhouette for K-Means, DBCV for HDBSCAN).
  - Chosen: `paraphrase-multilingual-mpnet-base-v2` yielded the highest silhouette (0.4320) and DBCV (0.3846).

## e. Clustering (2.0 pts)
- **Algorithms**:
  1. **K-Means** (requires K): inertia minimized, silhouette scored.
  2. **HDBSCAN** (`min_cluster_size=2`): density-based, yields noise and DBCV score.

- **Parameter Selection**:
  - *K-Means*: Elbow plot (`elbow_embeddings_paraphrase-multilingual-mpnet-base-v2.png`) suggested K≈6; silhouette analysis selected K=10 (score 0.4320).
  - *HDBSCAN*: ran with `min_cluster_size=2`; identified 6 clusters + 1 noise (DBCV 0.3846).



## f. Comparison and Analysis of Results (3.0 pts)
### Objective Metrics
| Algorithm | Representation                                  | #Clusters | Noise | Silhouette | DBCV   |
|-----------|-------------------------------------------------|-----------|-------|------------|--------|
| K-Means   | TF-IDF                                          | 15        | 0     | 0.2627     | –      |
| HDBSCAN   | TF-IDF                                          | 6         | 4     | –          | 0.1554 |
| K-Means   | Embeddings (`paraphrase-multilingual-mpnet-base-v2`) | 10        | 0     | 0.4320     | –      |
| HDBSCAN   | Embeddings (`paraphrase-multilingual-mpnet-base-v2`) | 6         | 1     | –          | 0.3846 |

### Subjective Analysis & Class Labels
Using the HDBSCAN result on `paraphrase-multilingual-mpnet-base-v2`, we define **6 classes** (plus 1 noise document):

- **Noise** (1 doc): outlier not assigned to any cluster.

- **Cluster 0** (3 docs): _Top words_: paulista, jurídicos, jurídica, brasileiro, legal, ilegal, brasileira, ambiental, ambientais, fiscalizem

- **Cluster 1** (3 docs): _Top words_: brasileiro, judiciária, judicial, jurídica, advogado, jurídicas, acusado, tribunal, oficiada, ricardo

- **Cluster 2** (4 docs): _Top words_: brasileiro, imputado, ricardo, julgado, acusado, prisional, caso, penal, fiscais, comarca

- **Cluster 3** (11 docs): _Top words_: brasileiro, brasileira, brasil, comarca, decretação, judiciais, juízo, procuradores, 3344ribeiro, luísa

- **Cluster 4** (3 docs): _Top words_: paulista, brasileiro, brasil, autoriza, juros, copacabana, requerente, juízo, comarca, caso

- **Cluster 5** (5 docs): _Top words_: paulista, brasileiro, brasileira, ricardo, laboradas, funcionário, comarca, juízo, reclamante, juros

*Interpretation*: clusters naturally group by document type and jurisdiction (e.g., criminal vs. civil vs. family vs. labor).

- **Best Representation**: HDBSCAN on MPNet embeddings, due to highest DBCV and clear thematic cohesion.

## g. Conclusion (0.5 pts)
HDBSCAN with `paraphrase-multilingual-mpnet-base-v2` embeddings produced the most coherent clusters (6 classes, DBCV 0.3846). Key challenges were choosing and justifying preprocessing for Portuguese text and the lack of labeled ground truth. Future work includes using a Portuguese SpaCy model for preprocessing and tuning TF-IDF parameters for improved BOW representation.


| Algorithm | Representation | #Clusters | Noise | Silhouette | DBCV |
|-----------|------------------------------------------------------|-----------|-------|------------|--------|
| K-Means | TF-IDF | 15 | 0 | 0.2627 | – |
| HDBSCAN | TF-IDF | 6 | 4 | – | 0.1554 |
| K-Means | Embeddings (en_core_web_md) | 13 | 0 | 0.2193 | – |
| HDBSCAN | Embeddings (en_core_web_md) | 2 | 4 | – | 0.1191 |
| K-Means | Embeddings (paraphrase-multilingual-mpnet-base-v2) | 10 | 0 | 0.4320 | – |
| HDBSCAN | Embeddings (paraphrase-multilingual-mpnet-base-v2) | 6 | 1 | – | 0.3846 |
| K-Means | Embeddings (pt_core_news_md) | 2 | 0 | 0.3211 | – |
| HDBSCAN | Embeddings (pt_core_news_md) | 4 | 2 | – | 0.2702 |
| K-Means | Embeddings (all-MiniLM-L6-v2) | 11 | 0 | 0.3162 | – |
| HDBSCAN | Embeddings (all-MiniLM-L6-v2) | 5 | 4 | – | 0.1865 |