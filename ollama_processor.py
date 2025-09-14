"""
Ollama-based semantic text processing and clustering module
Uses Ollama server for embeddings instead of Llama Stack
"""

import json
import re
import time
import warnings
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
import requests
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from config import *

warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
except:
    print("Warning: Could not download NLTK data. Some features may not work.")


class OllamaEmbeddingClient:
    """Client for Ollama embedding API"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "nomic-embed-text",
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.session = requests.Session()
        self.embedding_dimension = None

        # Test connection and model availability
        self._test_connection()
        self._check_model()

    def _test_connection(self):
        """Test connection to Ollama server"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                print(f"✅ Connected to Ollama server at {self.base_url}")
                return True
            else:
                print(f"⚠️  Ollama server responded with status {response.status_code}")
                return False
        except Exception as e:
            print(f"⚠️  Could not connect to Ollama server: {e}")
            print("Make sure Ollama is running: ollama serve")
            return False

    def _check_model(self):
        """Check if the embedding model is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                available_models = [model["name"] for model in models.get("models", [])]

                if self.model_name in available_models:
                    print(f"✅ Model '{self.model_name}' is available")
                    return True
                else:
                    print(f"⚠️  Model '{self.model_name}' not found")
                    print(f"Available models: {available_models}")
                    print(f"To install the model, run: ollama pull {self.model_name}")

                    # Try to pull the model automatically if it's a known embedding model
                    if self.model_name in [
                        "nomic-embed-text",
                        "all-minilm",
                        "mxbai-embed-large",
                    ]:
                        print(f"Attempting to pull {self.model_name} model...")
                        self._pull_model()

                    return False
        except Exception as e:
            print(f"⚠️  Could not check available models: {e}")
            return False

    def _pull_model(self):
        """Attempt to pull the embedding model"""
        try:
            print(f"Pulling model {self.model_name}... This may take a few minutes.")

            # Use streaming API to pull model
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                stream=True,
                timeout=300,  # 5 minutes timeout for model download
            )

            if response.status_code == 200:
                for chunk in response.iter_lines():
                    if chunk:
                        try:
                            data = json.loads(chunk.decode())
                            if "status" in data:
                                print(f"📥 {data['status']}")
                            if data.get("status") == "success":
                                print(f"✅ Successfully pulled {self.model_name}")
                                return True
                        except json.JSONDecodeError:
                            continue
            else:
                print(f"❌ Failed to pull model: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Error pulling model: {e}")
            return False

    def get_embeddings(self, texts: List[str], batch_size: int = 1) -> np.ndarray:
        """Get embeddings from Ollama server"""
        if not texts:
            return np.array([])

        all_embeddings = []

        # Ollama typically processes one text at a time for embeddings
        for i, text in enumerate(tqdm(texts, desc="Getting embeddings from Ollama")):
            try:
                # Prepare request data for Ollama embeddings API
                request_data = {"model": self.model_name, "prompt": text}

                response = self.session.post(
                    f"{self.base_url}/api/embeddings", json=request_data, timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    if "embedding" in result:
                        embedding = result["embedding"]
                        all_embeddings.append(embedding)

                        # Set embedding dimension from first embedding
                        if self.embedding_dimension is None:
                            self.embedding_dimension = len(embedding)
                            print(f"Embedding dimension: {self.embedding_dimension}")
                    else:
                        print(f"Warning: No embedding in response for text {i+1}")
                        print(f"Response: {result}")
                        # Create zero embedding as fallback
                        fallback_dim = self.embedding_dimension or 768
                        all_embeddings.append([0.0] * fallback_dim)

                else:
                    print(
                        f"Warning: Request failed for text {i+1}: {response.status_code}"
                    )
                    print(f"Response: {response.text}")
                    fallback_dim = self.embedding_dimension or 768
                    all_embeddings.append([0.0] * fallback_dim)

            except Exception as e:
                print(f"Error getting embedding for text {i+1}: {e}")
                fallback_dim = self.embedding_dimension or 768
                all_embeddings.append([0.0] * fallback_dim)

            # Small delay to be nice to the server
            time.sleep(0.1)

        embeddings_array = np.array(all_embeddings)

        # Check for zero embeddings
        if len(embeddings_array) > 0:
            zero_count = np.sum(np.all(embeddings_array == 0, axis=1))
            if zero_count > 0:
                print(
                    f"Warning: {zero_count} out of {len(embeddings_array)} embeddings are zero"
                )

        # Normalize embeddings for better clustering
        from sklearn.preprocessing import normalize

        embeddings_array = normalize(embeddings_array, norm="l2")

        return embeddings_array

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        if self.embedding_dimension is None:
            # Test with a small text to get dimension
            test_embeddings = self.get_embeddings(["test"])
            if len(test_embeddings) > 0:
                self.embedding_dimension = len(test_embeddings[0])
            else:
                # Default dimensions for common models
                model_dims = {
                    "nomic-embed-text": 768,
                    "all-minilm": 384,
                    "mxbai-embed-large": 1024,
                }
                self.embedding_dimension = model_dims.get(self.model_name, 768)

        return self.embedding_dimension


class OllamaTextProcessor:
    """Text processor using Ollama for embeddings"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "nomic-embed-text",
    ):
        self.client = OllamaEmbeddingClient(base_url, model_name)
        self.embeddings = None
        self.processed_texts = None

        # Basic text cleaning components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        print(f"🦙 Using Ollama server at {base_url} with model {model_name}")

    def clean_text_for_semantic_analysis(self, text: str) -> str:
        """Clean text while preserving semantic meaning"""
        if not text:
            return ""

        # Convert to lowercase but preserve structure
        text = text.lower()

        # Remove URLs but keep context
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            " [URL] ",
            text,
        )

        # Replace GitHub mentions with placeholder to preserve context
        text = re.sub(r"@(\w+)", r"user \1", text)

        # Replace issue references with placeholder
        text = re.sub(r"#(\d+)", r"issue \1", text)

        # Clean code blocks but preserve that code was mentioned
        text = re.sub(r"```[^`]*```", " [CODE_BLOCK] ", text, flags=re.DOTALL)
        text = re.sub(r"`([^`]+)`", r" code \1 ", text)

        # Remove HTML tags but preserve content structure
        text = re.sub(r"<[^>]*>", " ", text)

        # Clean up excessive punctuation but preserve sentence structure
        text = re.sub(r"[^\w\s\.\!\?\,\;\:\-]", " ", text)
        text = re.sub(r"([\.!\?]){2,}", r"\1", text)  # Multiple punctuation

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Preserve sentence structure by ensuring proper spacing around punctuation
        text = re.sub(r"([\.!\?])\s*", r"\1 ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def extract_semantic_features(self, texts: List[str]) -> np.ndarray:
        """Extract semantic embeddings using Ollama"""
        print("Processing texts for semantic analysis with Ollama...")

        # Clean texts while preserving semantic structure
        self.processed_texts = [
            self.clean_text_for_semantic_analysis(text) for text in texts
        ]

        # Filter out very short texts
        filtered_texts = []
        for text in self.processed_texts:
            if len(text.split()) >= 3:  # At least 3 words for meaningful semantics
                filtered_texts.append(text)
            else:
                filtered_texts.append("empty content")  # Placeholder for short texts

        if not filtered_texts:
            print("Warning: No valid text data found after preprocessing.")
            # Return random embeddings as fallback
            embedding_dim = self.client.get_embedding_dimension()
            return np.random.rand(len(texts), embedding_dim) * 0.1

        print(f"Getting embeddings for {len(filtered_texts)} texts from Ollama...")

        try:
            # Get embeddings from Ollama server
            self.embeddings = self.client.get_embeddings(filtered_texts)

            if len(self.embeddings) == 0:
                raise Exception("No embeddings returned from server")

            print(f"✅ Generated embeddings with shape: {self.embeddings.shape}")
            return self.embeddings

        except Exception as e:
            print(f"❌ Error getting embeddings from Ollama: {e}")
            print("Falling back to TF-IDF embeddings...")

            # Fallback to TF-IDF if Ollama fails
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize

            vectorizer = TfidfVectorizer(
                max_features=min(1000, len(filtered_texts) * 10),
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8,
            )

            tfidf_matrix = vectorizer.fit_transform(filtered_texts)
            tfidf_normalized = normalize(tfidf_matrix, norm="l2")

            self.embeddings = tfidf_normalized.toarray()
            print(f"⚠️  Using TF-IDF fallback with shape: {self.embeddings.shape}")
            return self.embeddings

    def get_semantic_similarity_terms(
        self,
        texts: List[str],
        cluster_labels: np.ndarray,
        cluster_id: int,
        top_n: int = 10,
    ) -> List[Tuple[str, float]]:
        """Get representative terms for a cluster based on semantic similarity"""
        if self.embeddings is None:
            return []

        # Get indices of documents in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        if len(cluster_indices) == 0:
            return []

        # Get embeddings for this cluster
        cluster_embeddings = self.embeddings[cluster_indices]
        cluster_texts = [texts[i] for i in cluster_indices]

        # Find the centroid of the cluster
        cluster_centroid = np.mean(cluster_embeddings, axis=0)

        # Find texts closest to centroid (most representative)
        similarities = cosine_similarity([cluster_centroid], cluster_embeddings)[0]
        most_representative_indices = np.argsort(similarities)[::-1][
            : min(5, len(similarities))
        ]

        # Extract key terms from most representative texts
        representative_texts = [cluster_texts[i] for i in most_representative_indices]

        # Use simple frequency analysis on representative texts
        all_words = []
        for text in representative_texts:
            # Basic preprocessing for term extraction
            cleaned = self.clean_text_for_semantic_analysis(text)
            words = cleaned.split()

            # Filter for meaningful terms
            filtered_words = []
            for word in words:
                if (
                    len(word) >= 3
                    and word.isalpha()
                    and word not in self.stop_words
                    and word not in {"code", "issue", "user", "url", "code_block"}
                ):
                    filtered_words.append(word)

            all_words.extend(filtered_words)

        # Count term frequencies
        from collections import Counter

        word_counts = Counter(all_words)

        # Get top terms with their frequencies
        top_terms = word_counts.most_common(top_n)

        # Normalize scores to 0-1 range
        if top_terms:
            max_count = max(count for _, count in top_terms)
            normalized_terms = [(term, count / max_count) for term, count in top_terms]
        else:
            normalized_terms = []

        return normalized_terms

    def find_cluster_themes(
        self, texts: List[str], cluster_labels: np.ndarray, cluster_id: int
    ) -> Dict[str, any]:
        """Analyze cluster to find themes and characteristics"""
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        if len(cluster_indices) == 0:
            return {}

        cluster_texts = [texts[i] for i in cluster_indices]
        cluster_embeddings = self.embeddings[cluster_indices]

        # Find centroid
        centroid = np.mean(cluster_embeddings, axis=0)

        # Find most and least representative texts
        similarities = cosine_similarity([centroid], cluster_embeddings)[0]
        most_representative_idx = cluster_indices[np.argmax(similarities)]

        # Calculate cluster cohesion (average pairwise similarity)
        if len(cluster_embeddings) > 1:
            pairwise_sims = cosine_similarity(cluster_embeddings)
            # Exclude diagonal (self-similarity)
            mask = ~np.eye(pairwise_sims.shape[0], dtype=bool)
            cohesion = np.mean(pairwise_sims[mask])
        else:
            cohesion = 1.0

        return {
            "cluster_id": cluster_id,
            "size": len(cluster_indices),
            "cohesion": float(cohesion),
            "most_representative_text": (
                texts[most_representative_idx][:200] + "..."
                if len(texts[most_representative_idx]) > 200
                else texts[most_representative_idx]
            ),
            "centroid_similarity_avg": float(np.mean(similarities)),
            "centroid_similarity_std": float(np.std(similarities)),
        }


class OllamaClusteringAnalyzer:
    """Clustering analyzer using Ollama for embeddings"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "nomic-embed-text",
    ):
        self.text_processor = OllamaTextProcessor(base_url, model_name)
        self.embeddings = None
        self.cluster_results = {}

    def determine_optimal_clusters_semantic(self, embeddings: np.ndarray) -> int:
        """Determine optimal number of clusters using silhouette analysis"""
        n_samples = len(embeddings)

        if n_samples < MIN_CLUSTERS:
            print(
                f"Warning: Only {n_samples} samples available. Using {min(n_samples, 2)} clusters."
            )
            return min(n_samples, 2)

        # For semantic clustering, we can handle more clusters effectively
        max_k = min(
            MAX_CLUSTERS, n_samples // 3, 15
        )  # Allow up to 15 clusters for semantic analysis
        K_range = range(MIN_CLUSTERS, max_k + 1)

        if not K_range:
            return DEFAULT_NUM_CLUSTERS

        print("Evaluating optimal number of clusters...")
        silhouette_scores = []

        for k in tqdm(K_range, desc="Testing cluster counts"):
            try:
                # Use cosine distance for semantic embeddings
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)

                # Check if we actually got multiple clusters
                unique_labels = len(set(cluster_labels))
                if unique_labels > 1:
                    # Use cosine distance for silhouette score with semantic embeddings
                    score = silhouette_score(
                        embeddings, cluster_labels, metric="cosine"
                    )
                    silhouette_scores.append(score)
                else:
                    print(f"Warning: k={k} resulted in only {unique_labels} cluster(s)")
                    silhouette_scores.append(0)
            except Exception as e:
                print(f"Warning: Clustering with k={k} failed: {e}")
                silhouette_scores.append(0)

        if not silhouette_scores or all(score <= 0 for score in silhouette_scores):
            print("Warning: Silhouette analysis failed. Using default cluster count.")
            return min(DEFAULT_NUM_CLUSTERS, max_k)

        # Return the k with highest silhouette score
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(
            f"Optimal clusters: {optimal_k} (silhouette score: {max(silhouette_scores):.3f})"
        )
        return optimal_k

    def perform_semantic_kmeans(
        self, embeddings: np.ndarray, n_clusters: int
    ) -> Tuple[np.ndarray, Dict]:
        """Perform K-means clustering optimized for semantic embeddings"""

        # Ensure we don't try to create more clusters than samples
        n_samples = len(embeddings)
        effective_clusters = min(n_clusters, n_samples)

        if effective_clusters < 2:
            print(f"Warning: Only {n_samples} samples, creating single cluster")
            cluster_labels = np.zeros(n_samples, dtype=int)
            silhouette_avg = 0.0
            inertia = 0.0
        else:
            kmeans = KMeans(
                n_clusters=effective_clusters,
                random_state=42,
                n_init=10,
                algorithm="lloyd",
            )
            cluster_labels = kmeans.fit_predict(embeddings)
            inertia = kmeans.inertia_

            # Calculate silhouette score with error handling
            silhouette_avg = 0.0
            unique_labels = len(set(cluster_labels))

            if unique_labels > 1:
                try:
                    silhouette_avg = silhouette_score(
                        embeddings, cluster_labels, metric="cosine"
                    )
                except Exception as e:
                    print(f"Warning: Could not calculate silhouette score: {e}")
                    silhouette_avg = 0.0
            else:
                print(
                    f"Warning: All samples assigned to single cluster, silhouette score = 0"
                )

        metadata = {
            "algorithm": "ollama_kmeans",
            "n_clusters": effective_clusters,
            "actual_clusters": len(set(cluster_labels)),
            "silhouette_score": silhouette_avg,
            "embedding_dimension": embeddings.shape[1],
            "distance_metric": "cosine",
            "embedding_source": "ollama",
            "model_name": self.text_processor.client.model_name,
            "inertia": inertia,
        }

        return cluster_labels, metadata

    def perform_semantic_hierarchical(
        self, embeddings: np.ndarray, n_clusters: int
    ) -> Tuple[np.ndarray, Dict]:
        """Perform hierarchical clustering with cosine distance"""

        # Ensure we don't try to create more clusters than samples
        n_samples = len(embeddings)
        effective_clusters = min(n_clusters, n_samples)

        if effective_clusters < 2:
            print(f"Warning: Only {n_samples} samples, creating single cluster")
            cluster_labels = np.zeros(n_samples, dtype=int)
            silhouette_avg = 0.0
        else:
            clustering = AgglomerativeClustering(
                n_clusters=effective_clusters,
                metric="cosine",
                linkage="average",
            )
            cluster_labels = clustering.fit_predict(embeddings)

            # Calculate silhouette score with error handling
            silhouette_avg = 0.0
            unique_labels = len(set(cluster_labels))

            if unique_labels > 1:
                try:
                    silhouette_avg = silhouette_score(
                        embeddings, cluster_labels, metric="cosine"
                    )
                except Exception as e:
                    print(f"Warning: Could not calculate silhouette score: {e}")
                    silhouette_avg = 0.0
            else:
                print(
                    f"Warning: All samples assigned to single cluster, silhouette score = 0"
                )

        metadata = {
            "algorithm": "ollama_hierarchical",
            "n_clusters": effective_clusters,
            "actual_clusters": len(set(cluster_labels)),
            "silhouette_score": silhouette_avg,
            "embedding_dimension": embeddings.shape[1],
            "distance_metric": "cosine",
            "embedding_source": "ollama",
            "model_name": self.text_processor.client.model_name,
            "linkage": "average",
        }

        return cluster_labels, metadata

    def perform_semantic_dbscan(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """DBSCAN clustering optimized for semantic embeddings"""
        n_samples = len(embeddings)

        # For semantic embeddings, we need different eps values
        eps = 0.3  # Start with smaller eps for cosine distance
        min_samples = max(3, min(n_samples // 15, 10))

        if n_samples < 15:
            eps = 0.5
            min_samples = 2

        try:
            # Use precomputed cosine distance matrix for DBSCAN
            distance_matrix = pairwise_distances(embeddings, metric="cosine")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
            cluster_labels = dbscan.fit_predict(distance_matrix)
        except Exception as e:
            print(f"Warning: DBSCAN clustering failed ({e}). Creating single cluster.")
            cluster_labels = np.zeros(n_samples, dtype=int)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        # Calculate silhouette score if we have clusters
        silhouette_avg = 0
        if n_clusters > 1 and len(set(cluster_labels)) > 1:
            try:
                silhouette_avg = silhouette_score(
                    embeddings, cluster_labels, metric="cosine"
                )
            except Exception as e:
                print(f"Warning: Could not calculate silhouette score for DBSCAN: {e}")
                silhouette_avg = 0

        metadata = {
            "algorithm": "ollama_dbscan",
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "silhouette_score": silhouette_avg,
            "embedding_dimension": embeddings.shape[1],
            "distance_metric": "cosine",
            "embedding_source": "ollama",
            "model_name": self.text_processor.client.model_name,
            "eps": eps,
            "min_samples": min_samples,
        }

        return cluster_labels, metadata

    def reduce_dimensions_for_visualization(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reduce semantic embeddings for visualization"""
        n_samples, n_features = embeddings.shape

        print(f"Reducing {n_features}D embeddings to 2D for visualization...")

        # For semantic embeddings, we can be more aggressive with PCA
        max_components = min(50, n_features, n_samples - 1)

        # First reduce to reasonable size with PCA if needed
        if n_features > 50 and max_components >= 10:
            pca = PCA(n_components=max_components, random_state=42)
            pca_result = pca.fit_transform(embeddings)
            print(
                f"PCA reduced dimensions to {max_components} (explained variance: {pca.explained_variance_ratio_.sum():.3f})"
            )
        else:
            pca_result = embeddings

        # Handle edge cases
        if n_samples < 4:
            print("Warning: Too few samples for t-SNE visualization.")
            pca_2d_result = np.zeros((n_samples, 2))
            tsne_result = np.zeros((n_samples, 2))
            return pca_2d_result, tsne_result

        # t-SNE with appropriate parameters for semantic embeddings
        perplexity = min(30, max(5, (n_samples - 1) // 3))
        try:
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=perplexity,
                metric="cosine",
                init="pca",
                learning_rate="auto",
            )
            tsne_result = tsne.fit_transform(pca_result)
            print("✅ t-SNE visualization completed")
        except Exception as e:
            print(f"Warning: t-SNE failed ({e}). Using PCA for both visualizations.")
            tsne_result = None

        # PCA to 2D
        n_components_2d = min(2, pca_result.shape[1], n_samples - 1)
        if n_components_2d < 2:
            if n_components_2d == 1:
                pca_2d = PCA(n_components=1, random_state=42)
                pca_1d = pca_2d.fit_transform(pca_result)
                pca_2d_result = np.column_stack([pca_1d, np.zeros(n_samples)])
            else:
                pca_2d_result = np.zeros((n_samples, 2))
        else:
            pca_2d = PCA(n_components=2, random_state=42)
            pca_2d_result = pca_2d.fit_transform(pca_result)

        # Use PCA if t-SNE failed
        if tsne_result is None:
            tsne_result = pca_2d_result.copy()

        return pca_2d_result, tsne_result

    def analyze_issues_semantic(self, issues: List[Dict]) -> Dict:
        """Perform complete semantic clustering analysis using Ollama"""
        print("🦙 Starting Ollama semantic clustering analysis...")

        # Extract text data
        texts = [issue["full_text"] for issue in issues]

        # Extract semantic features using Ollama
        print("🔤 Extracting semantic embeddings from Ollama...")
        self.embeddings = self.text_processor.extract_semantic_features(texts)

        # Determine optimal number of clusters
        print("📊 Determining optimal number of clusters...")
        optimal_clusters = self.determine_optimal_clusters_semantic(self.embeddings)

        # Perform different clustering algorithms
        results = {}

        for algorithm in CLUSTERING_ALGORITHMS:
            print(f"🔍 Performing {algorithm} semantic clustering...")

            if algorithm == "kmeans":
                labels, metadata = self.perform_semantic_kmeans(
                    self.embeddings, optimal_clusters
                )
            elif algorithm == "hierarchical":
                labels, metadata = self.perform_semantic_hierarchical(
                    self.embeddings, optimal_clusters
                )
            elif algorithm == "dbscan":
                labels, metadata = self.perform_semantic_dbscan(self.embeddings)
            else:
                continue

            # Get semantic terms/themes for each cluster
            print(f"🏷️  Extracting cluster themes for {algorithm}...")
            cluster_terms = {}
            cluster_themes = {}
            unique_labels = set(labels)

            for cluster_id in unique_labels:
                if cluster_id == -1:  # Skip noise points in DBSCAN
                    continue

                # Get representative terms
                terms = self.text_processor.get_semantic_similarity_terms(
                    texts, labels, cluster_id, 10
                )
                cluster_terms[cluster_id] = terms

                # Get cluster themes
                themes = self.text_processor.find_cluster_themes(
                    texts, labels, cluster_id
                )
                cluster_themes[cluster_id] = themes

            # Add cluster assignments to issues
            clustered_issues = []
            for i, issue in enumerate(issues):
                issue_copy = issue.copy()
                issue_copy["cluster"] = int(labels[i])
                clustered_issues.append(issue_copy)

            results[algorithm] = {
                "labels": labels,
                "metadata": metadata,
                "cluster_terms": cluster_terms,
                "cluster_themes": cluster_themes,
                "issues": clustered_issues,
            }

        # Generate visualization coordinates
        print("📈 Generating visualization coordinates...")
        pca_coords, tsne_coords = self.reduce_dimensions_for_visualization(
            self.embeddings
        )

        results["visualization"] = {
            "pca_coordinates": pca_coords,
            "tsne_coordinates": tsne_coords,
        }

        self.cluster_results = results
        print("✅ Ollama semantic clustering analysis completed!")
        return results


def main():
    """Test the Ollama clustering analyzer"""
    pass


if __name__ == "__main__":
    main()
