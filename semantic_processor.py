"""
Semantic text processing and clustering module for GitHub issues analysis
Uses sentence transformers for semantic embeddings instead of TF-IDF
"""

import re
import warnings
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
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


class SemanticTextProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic text processor with sentence transformer model

        Popular models:
        - 'all-MiniLM-L6-v2': Fast and good performance, 384 dimensions
        - 'all-mpnet-base-v2': Better performance, slower, 768 dimensions
        - 'paraphrase-mpnet-base-v2': Good for semantic similarity
        - 'all-distilroberta-v1': Balanced performance
        """
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.processed_texts = None

        # Basic text cleaning components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        print(f"Loading sentence transformer model: {model_name}")
        self.model = self._load_model_with_fallbacks(model_name)

    def _load_model_with_fallbacks(self, model_name: str, offline_mode: bool = False):
        """Load sentence transformer model with multiple fallback strategies"""

        # Set offline environment variables if offline mode requested
        if offline_mode:
            import os

            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            print("🚫 Running in offline mode - no model downloads allowed")

        # Strategy 1: Try original model with offline mode first
        print(f"Attempting to load {model_name}...")
        try:
            # First try loading from cache (offline)
            model = SentenceTransformer(model_name, cache_folder=None)
            print(
                f"✅ Model loaded from cache. Embedding dimension: {model.get_sentence_embedding_dimension()}"
            )
            return model
        except Exception as e:
            print(f"⚠️  Cache load failed: {str(e)[:100]}...")
            if offline_mode:
                print("🚫 Offline mode: skipping online model downloads")
                # Skip to TF-IDF fallback immediately
                print(
                    "🚨 No cached models available. Creating TF-IDF-based embedding fallback..."
                )
                return self._create_tfidf_fallback()

        # Strategy 2: Try with different download settings
        try:
            print("Trying alternative download method...")
            import os

            os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Ensure online mode
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"  # Disable telemetry

            model = SentenceTransformer(model_name, trust_remote_code=True)
            print(
                f"✅ Model loaded successfully. Embedding dimension: {model.get_sentence_embedding_dimension()}"
            )
            return model
        except Exception as e:
            print(f"⚠️  Alternative download failed: {str(e)[:100]}...")

        # Strategy 3: Try smaller/fallback models
        fallback_models = [
            "all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L6-v2",
            "all-MiniLM-L12-v2",
        ]
        if model_name not in fallback_models:
            fallback_models.insert(0, model_name)

        for fallback_model in fallback_models:
            try:
                print(f"Trying fallback model: {fallback_model}")
                model = SentenceTransformer(fallback_model)
                print(f"✅ Fallback model {fallback_model} loaded successfully")
                return model
            except Exception as e:
                print(f"⚠️  Fallback {fallback_model} failed: {str(e)[:100]}...")
                continue

        # Strategy 4: Use a basic transformer model manually
        try:
            print("Attempting manual transformer model setup...")
            import torch
            from transformers import AutoModel, AutoTokenizer

            # Try a very basic model that might be cached
            model_name_basic = "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name_basic)
            transformer_model = AutoModel.from_pretrained(model_name_basic)

            # Create a basic sentence transformer-like wrapper
            class BasicSentenceTransformer:
                def __init__(self, tokenizer, model):
                    self.tokenizer = tokenizer
                    self.model = model
                    self.model.eval()

                def encode(
                    self,
                    sentences,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=32,
                ):
                    if isinstance(sentences, str):
                        sentences = [sentences]

                    embeddings = []
                    for i in range(0, len(sentences), batch_size):
                        batch = sentences[i : i + batch_size]
                        inputs = self.tokenizer(
                            batch,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512,
                        )

                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            # Use mean pooling of last hidden states
                            embeddings_batch = outputs.last_hidden_state.mean(dim=1)

                        if normalize_embeddings:
                            embeddings_batch = torch.nn.functional.normalize(
                                embeddings_batch, p=2, dim=1
                            )

                        if convert_to_numpy:
                            embeddings_batch = embeddings_batch.numpy()

                        embeddings.extend(embeddings_batch)

                    return np.array(embeddings) if convert_to_numpy else embeddings

                def get_sentence_embedding_dimension(self):
                    return 768  # DistilBERT dimension

            model = BasicSentenceTransformer(tokenizer, transformer_model)
            print("✅ Manual transformer model setup successful")
            return model

        except Exception as e:
            print(f"⚠️  Manual setup failed: {str(e)[:100]}...")

        # Strategy 5: Last resort - create a simple embedding model
        print(
            "🚨 All model loading failed. Creating simple TF-IDF-based embedding fallback..."
        )

        class SimpleTFIDFEmbedder:
            def __init__(self):
                from sklearn.feature_extraction.text import TfidfVectorizer

                self.vectorizer = TfidfVectorizer(
                    max_features=384, stop_words="english", ngram_range=(1, 2)
                )
                self.fitted = False

            def encode(
                self,
                sentences,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32,
            ):
                if isinstance(sentences, str):
                    sentences = [sentences]

                if not self.fitted:
                    self.vectorizer.fit(sentences)
                    self.fitted = True

                embeddings = self.vectorizer.transform(sentences)

                if normalize_embeddings:
                    from sklearn.preprocessing import normalize

                    embeddings = normalize(embeddings, norm="l2")

                if convert_to_numpy:
                    embeddings = embeddings.toarray()

                return embeddings

            def get_sentence_embedding_dimension(self):
                return 384

        model = SimpleTFIDFEmbedder()
        print("⚠️  Using TF-IDF fallback (semantic quality will be reduced)")
        return model

    def clean_text_for_semantic_analysis(self, text: str) -> str:
        """
        Clean text while preserving semantic meaning
        Less aggressive than keyword-based cleaning
        """
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
        """Extract semantic embeddings from texts using sentence transformers"""
        print("Processing texts for semantic analysis...")

        # Clean texts while preserving semantic structure
        self.processed_texts = [
            self.clean_text_for_semantic_analysis(text) for text in texts
        ]

        # Filter out very short texts (less informative for semantics)
        filtered_texts = []
        original_indices = []

        for i, text in enumerate(self.processed_texts):
            if len(text.split()) >= 3:  # At least 3 words for meaningful semantics
                filtered_texts.append(text)
                original_indices.append(i)
            else:
                filtered_texts.append("empty content")  # Placeholder for short texts
                original_indices.append(i)

        if not filtered_texts:
            print("Warning: No valid text data found after preprocessing.")
            # Return random embeddings as fallback
            embedding_dim = self.model.get_sentence_embedding_dimension()
            return np.random.rand(len(texts), embedding_dim) * 0.1

        print(f"Generating embeddings for {len(filtered_texts)} texts...")

        try:
            # Generate embeddings in batches for memory efficiency
            batch_size = 32
            embeddings_list = []

            for i in tqdm(
                range(0, len(filtered_texts), batch_size), desc="Generating embeddings"
            ):
                batch_texts = filtered_texts[i : i + batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # L2 normalize for better clustering
                )
                embeddings_list.append(batch_embeddings)

            # Concatenate all embeddings
            self.embeddings = np.vstack(embeddings_list)

            print(f"✅ Generated embeddings with shape: {self.embeddings.shape}")
            return self.embeddings

        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            # Fallback to random embeddings
            embedding_dim = self.model.get_sentence_embedding_dimension()
            return np.random.rand(len(texts), embedding_dim) * 0.1

    def get_semantic_similarity_terms(
        self,
        texts: List[str],
        cluster_labels: np.ndarray,
        cluster_id: int,
        top_n: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Get representative terms for a cluster based on semantic similarity
        Instead of TF-IDF scores, we find the most representative texts and extract key terms
        """
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
        """
        Analyze cluster to find themes and characteristics using semantic analysis
        """
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
        least_representative_idx = cluster_indices[np.argmin(similarities)]

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


class SemanticClusteringAnalyzer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.text_processor = SemanticTextProcessor(model_name)
        self.embeddings = None
        self.cluster_results = {}

    def determine_optimal_clusters_semantic(self, embeddings: np.ndarray) -> int:
        """
        Determine optimal number of clusters using silhouette analysis
        Optimized for semantic embeddings
        """
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

                # Use cosine distance for silhouette score with semantic embeddings
                score = silhouette_score(embeddings, cluster_labels, metric="cosine")
                silhouette_scores.append(score)
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
        # Use cosine distance which works better with normalized embeddings
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            algorithm="lloyd",  # More stable for high-dimensional data
        )
        cluster_labels = kmeans.fit_predict(embeddings)

        # Calculate silhouette score with cosine distance
        silhouette_avg = silhouette_score(embeddings, cluster_labels, metric="cosine")

        metadata = {
            "algorithm": "semantic_kmeans",
            "n_clusters": n_clusters,
            "silhouette_score": silhouette_avg,
            "embedding_dimension": embeddings.shape[1],
            "distance_metric": "cosine",
            "inertia": kmeans.inertia_,
        }

        return cluster_labels, metadata

    def perform_semantic_hierarchical(
        self, embeddings: np.ndarray, n_clusters: int
    ) -> Tuple[np.ndarray, Dict]:
        """Perform hierarchical clustering with cosine distance"""
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",  # Average linkage works well with cosine distance
        )
        cluster_labels = clustering.fit_predict(embeddings)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(embeddings, cluster_labels, metric="cosine")

        metadata = {
            "algorithm": "semantic_hierarchical",
            "n_clusters": n_clusters,
            "silhouette_score": silhouette_avg,
            "embedding_dimension": embeddings.shape[1],
            "distance_metric": "cosine",
            "linkage": "average",
        }

        return cluster_labels, metadata

    def perform_semantic_dbscan(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """DBSCAN clustering optimized for semantic embeddings"""
        n_samples = len(embeddings)

        # For semantic embeddings, we need different eps values
        # Cosine distance typically ranges from 0 to 2
        eps = 0.3  # Start with smaller eps for cosine distance
        min_samples = max(
            3, min(n_samples // 15, 10)
        )  # Adjusted for semantic clustering

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
            "algorithm": "semantic_dbscan",
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "silhouette_score": silhouette_avg,
            "embedding_dimension": embeddings.shape[1],
            "distance_metric": "cosine",
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
                metric="cosine",  # Use cosine distance for semantic embeddings
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
        """Perform complete semantic clustering analysis on issues"""
        print("🧠 Starting semantic clustering analysis...")

        # Extract text data
        texts = [issue["full_text"] for issue in issues]

        # Extract semantic features (embeddings)
        print("🔤 Extracting semantic embeddings...")
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
        print("✅ Semantic clustering analysis completed!")
        return results


def main():
    """Test the semantic clustering analyzer"""
    pass


if __name__ == "__main__":
    main()
