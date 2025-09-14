"""
Text processing and clustering module for GitHub issues analysis
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
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
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


class TextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Extend stop words with GitHub/Programming specific common terms
        github_stopwords = {
            # Generic responses and affirmatives
            "yes",
            "yeah",
            "yep",
            "ok",
            "okay",
            "sure",
            "thanks",
            "thank",
            "welcome",
            "please",
            "need",
            "needed",
            "want",
            "would",
            "could",
            "should",
            "will",
            "can",
            "may",
            "might",
            "must",
            "shall",
            "going",
            "get",
            "got",
            "getting",
            "make",
            "making",
            "made",
            "take",
            "taking",
            "took",
            "give",
            "giving",
            "gave",
            "put",
            "putting",
            "see",
            "seeing",
            "saw",
            "look",
            "looking",
            "looked",
            "know",
            "knowing",
            "knew",
            "think",
            "thinking",
            "thought",
            "try",
            "trying",
            "tried",
            "work",
            "working",
            "worked",
            "use",
            "using",
            "used",
            "run",
            "running",
            "ran",
            "add",
            "adding",
            "added",
            "set",
            "setting",
            "way",
            "ways",
            # Generic technical terms that are too common
            "code",
            "file",
            "files",
            "line",
            "lines",
            "function",
            "method",
            "class",
            "issue",
            "issues",
            "problem",
            "problems",
            "error",
            "errors",
            "bug",
            "bugs",
            "fix",
            "fixed",
            "fixing",
            "update",
            "updated",
            "updating",
            "change",
            "changed",
            "changing",
            "changes",
            "version",
            "versions",
            "build",
            "building",
            "built",
            "test",
            "testing",
            "tested",
            "tests",
            "feature",
            "features",
            "support",
            "supported",
            "supporting",
            "example",
            "examples",
            "case",
            "cases",
            "time",
            "times",
            "new",
            "old",
            "good",
            "bad",
            "right",
            "wrong",
            "current",
            "different",
            "same",
            "similar",
            "available",
            "possible",
            "simple",
            "easy",
            "hard",
            "difficult",
            "better",
            "best",
            "worse",
            "worst",
            "first",
            "last",
            "next",
            "previous",
            "existing",
            "original",
            "default",
            "custom",
            "main",
            "basic",
            "advanced",
            "complete",
            "full",
            "empty",
            "null",
            "true",
            "false",
            # Generic project terms
            "project",
            "repo",
            "repository",
            "branch",
            "commit",
            "commits",
            "pr",
            "pull",
            "request",
            "merge",
            "merged",
            "merging",
            "push",
            "pushed",
            "pushing",
            "clone",
            "cloned",
            "cloning",
            "fork",
            "forked",
            "forking",
            # Generic responses in issues
            "hello",
            "hi",
            "hey",
            "help",
            "helping",
            "helped",
            "question",
            "questions",
            "answer",
            "answers",
            "solution",
            "solutions",
            "documentation",
            "docs",
            "readme",
            "install",
            "installation",
            "setup",
            "configure",
            "configuration",
            "implement",
            "implementation",
            "implemented",
            "implementing",
            # Common filler words
            "actually",
            "really",
            "quite",
            "very",
            "pretty",
            "pretty",
            "kind",
            "sort",
            "like",
            "just",
            "only",
            "also",
            "even",
            "still",
            "already",
            "yet",
            "now",
            "then",
            "here",
            "there",
            "where",
            "when",
            "why",
            "how",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "something",
            "anything",
            "everything",
            "nothing",
            "someone",
            "anyone",
            "everyone",
            "nobody",
            "somewhere",
            "anywhere",
            "everywhere",
            "nowhere",
        }

        self.stop_words.update(github_stopwords)
        self.vectorizer = None
        self.feature_names = None

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # Remove GitHub mentions and issue references
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\d+", "", text)

        # Remove code blocks
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"`.*?`", "", text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove special characters but keep spaces
        text = re.sub(r"[^\w\s]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def is_meaningful_term(self, token: str) -> bool:
        """Check if a term is meaningful for clustering"""
        # Skip if it's a stop word
        if token in self.stop_words:
            return False

        # Skip very short terms
        if len(token) < MIN_WORD_LENGTH:
            return False

        # Skip non-alphabetic tokens
        if not token.isalpha():
            return False

        # Skip single character repeated (like "aa", "bb")
        if len(set(token)) == 1 and len(token) > 1:
            return False

        # Skip terms that are mostly vowels or mostly consonants
        vowels = set("aeiou")
        vowel_count = sum(1 for c in token if c in vowels)
        consonant_count = len(token) - vowel_count

        # Skip if too many vowels or too few vowels (likely not real words)
        if vowel_count == 0 or consonant_count == 0:
            return False

        if vowel_count / len(token) > 0.8 or vowel_count / len(token) < 0.1:
            return False

        # Skip common programming artifacts
        programming_patterns = {
            "http",
            "https",
            "www",
            "com",
            "org",
            "net",
            "html",
            "css",
            "js",
            "json",
            "xml",
            "api",
            "url",
            "uri",
            "uuid",
            "id",
            "ids",
            "var",
            "const",
            "let",
            "def",
            "func",
            "return",
            "import",
            "export",
            "true",
            "false",
            "null",
            "undefined",
            "none",
            "nil",
            "string",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "array",
            "obj",
            "object",
            "item",
            "value",
            "key",
            "data",
            "param",
            "arg",
            "args",
            "kwargs",
            "self",
            "this",
            "that",
        }

        if token.lower() in programming_patterns:
            return False

        return True

    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing with lemmatization and POS filtering"""
        # Clean text first
        text = self.clean_text(text)

        if not text:
            return ""

        try:
            # Tokenize
            tokens = word_tokenize(text)

            # Get POS tags to filter for meaningful parts of speech
            pos_tags = nltk.pos_tag(tokens)

            # Keep only nouns, adjectives, and some verbs (not auxiliary verbs)
            meaningful_pos = {
                "NN",
                "NNS",
                "NNP",
                "NNPS",  # Nouns
                "JJ",
                "JJR",
                "JJS",  # Adjectives
                "VB",
                "VBG",
                "VBN",
                "VBP",
                "VBZ",  # Verbs (excluding VBD for past tense)
                "RB",
                "RBR",
                "RBS",  # Adverbs (some can be meaningful)
            }

            # Filter tokens based on POS and meaningfulness
            filtered_tokens = []
            for token, pos in pos_tags:
                token_lower = token.lower()

                # Apply basic filters
                if not self.is_meaningful_term(token_lower):
                    continue

                # Apply POS filter for longer terms (more likely to be meaningful)
                if len(token_lower) >= 4:
                    if pos not in meaningful_pos:
                        continue

                # For shorter terms, be more restrictive (only nouns and adjectives)
                elif len(token_lower) == 3:
                    if pos not in {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}:
                        continue

                filtered_tokens.append(token_lower)

            # Lemmatize the filtered tokens
            lemmatized_tokens = [
                self.lemmatizer.lemmatize(token) for token in filtered_tokens
            ]

            # Remove duplicates while preserving order
            seen = set()
            final_tokens = []
            for token in lemmatized_tokens:
                if token not in seen and self.is_meaningful_term(token):
                    seen.add(token)
                    final_tokens.append(token)

            return " ".join(final_tokens)

        except Exception as e:
            print(
                f"Warning: Advanced preprocessing failed ({e}). Using basic preprocessing."
            )
            # Fallback to basic preprocessing
            tokens = word_tokenize(text)
            tokens = [
                token for token in tokens if self.is_meaningful_term(token.lower())
            ]
            tokens = [self.lemmatizer.lemmatize(token.lower()) for token in tokens]
            return " ".join(tokens)

    def extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract TF-IDF features from texts"""
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Filter out empty texts
        non_empty_texts = [text for text in processed_texts if text.strip()]

        if not non_empty_texts:
            print("Warning: No valid text data found after preprocessing.")
            # Return a minimal feature matrix
            return np.zeros((len(texts), 1))

        # Adjust min_df based on dataset size
        min_df = max(1, min(2, len(non_empty_texts) // 10))

        # Create TF-IDF vectorizer with adjusted parameters for better feature selection
        self.vectorizer = TfidfVectorizer(
            max_features=min(MAX_FEATURES, len(non_empty_texts) * 10),
            stop_words=None,  # We handle stop words in preprocessing
            ngram_range=NGRAM_RANGE,
            min_df=max(2, min_df),  # Require terms to appear in at least 2 documents
            max_df=0.7,  # Exclude terms that appear in more than 70% of documents
            sublinear_tf=True,  # Use log scaling for term frequencies
            norm="l2",  # L2 normalization
            use_idf=True,  # Use inverse document frequency
            smooth_idf=True,  # Smooth IDF weights
        )

        try:
            # Fit and transform
            feature_matrix = self.vectorizer.fit_transform(processed_texts)
            self.feature_names = self.vectorizer.get_feature_names_out()

            # Check if we got any features
            if feature_matrix.shape[1] == 0:
                print("Warning: No features extracted. Using fallback approach.")
                # Try with more relaxed parameters
                self.vectorizer = TfidfVectorizer(
                    max_features=min(100, len(non_empty_texts) * 5),
                    stop_words=None,  # Don't remove stop words
                    ngram_range=(1, 1),  # Only unigrams
                    min_df=1,
                    max_df=1.0,
                )
                feature_matrix = self.vectorizer.fit_transform(processed_texts)
                self.feature_names = self.vectorizer.get_feature_names_out()

            return feature_matrix.toarray()

        except Exception as e:
            print(
                f"Warning: TF-IDF vectorization failed ({e}). Using basic word count."
            )
            # Fallback: simple word count approach
            from sklearn.feature_extraction.text import CountVectorizer

            count_vectorizer = CountVectorizer(
                max_features=min(50, len(non_empty_texts) * 2),
                stop_words=None,
                min_df=1,
            )
            feature_matrix = count_vectorizer.fit_transform(processed_texts)
            self.feature_names = count_vectorizer.get_feature_names_out()
            return feature_matrix.toarray()

    def get_top_terms_for_cluster(
        self,
        feature_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_id: int,
        top_n: int = 10,
    ) -> List[Tuple[str, float]]:
        """Get top terms for a specific cluster"""
        if self.feature_names is None:
            return []

        # Get indices of documents in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        if len(cluster_indices) == 0:
            return []

        # Calculate mean TF-IDF scores for this cluster
        cluster_features = feature_matrix[cluster_indices]
        mean_scores = np.mean(cluster_features, axis=0)

        # Get top terms
        top_indices = np.argsort(mean_scores)[::-1][:top_n]
        top_terms = [(self.feature_names[i], mean_scores[i]) for i in top_indices]

        return top_terms


class ClusteringAnalyzer:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.feature_matrix = None
        self.cluster_results = {}

    def determine_optimal_clusters(self, feature_matrix: np.ndarray) -> int:
        """Determine optimal number of clusters using silhouette analysis"""
        n_samples = len(feature_matrix)

        # Ensure we have enough samples for clustering
        if n_samples < MIN_CLUSTERS:
            print(
                f"Warning: Only {n_samples} samples available. Using {min(n_samples, 2)} clusters."
            )
            return min(n_samples, 2)

        # Adjust the range based on available samples
        max_k = min(MAX_CLUSTERS, n_samples // 2, 10)  # Cap at 10 for small datasets
        K_range = range(MIN_CLUSTERS, max_k + 1)

        if not K_range:
            return DEFAULT_NUM_CLUSTERS

        silhouette_scores = []
        for k in K_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(feature_matrix)
                score = silhouette_score(feature_matrix, cluster_labels)
                silhouette_scores.append(score)
            except Exception as e:
                print(f"Warning: Clustering with k={k} failed: {e}")
                silhouette_scores.append(0)

        if not silhouette_scores or all(score == 0 for score in silhouette_scores):
            print("Warning: Silhouette analysis failed. Using default cluster count.")
            return min(DEFAULT_NUM_CLUSTERS, max_k)

        # Return the k with highest silhouette score
        optimal_k = K_range[np.argmax(silhouette_scores)]
        return optimal_k

    def perform_kmeans_clustering(
        self, feature_matrix: np.ndarray, n_clusters: int
    ) -> Tuple[np.ndarray, Dict]:
        """Perform K-means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(feature_matrix, cluster_labels)

        metadata = {
            "algorithm": "kmeans",
            "n_clusters": n_clusters,
            "silhouette_score": silhouette_avg,
            "cluster_centers": kmeans.cluster_centers_,
            "inertia": kmeans.inertia_,
        }

        return cluster_labels, metadata

    def perform_hierarchical_clustering(
        self, feature_matrix: np.ndarray, n_clusters: int
    ) -> Tuple[np.ndarray, Dict]:
        """Perform hierarchical clustering"""
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        cluster_labels = clustering.fit_predict(feature_matrix)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(feature_matrix, cluster_labels)

        metadata = {
            "algorithm": "hierarchical",
            "n_clusters": n_clusters,
            "silhouette_score": silhouette_avg,
            "linkage": "ward",
        }

        return cluster_labels, metadata

    def perform_dbscan_clustering(
        self, feature_matrix: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Perform DBSCAN clustering"""
        n_samples = len(feature_matrix)

        # Use a reasonable eps value based on feature matrix
        eps = 0.5
        min_samples = max(2, min(n_samples // 10, n_samples // 2))

        # For very small datasets, adjust parameters
        if n_samples < 10:
            eps = 0.3
            min_samples = 2

        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(feature_matrix)
        except Exception as e:
            print(f"Warning: DBSCAN clustering failed ({e}). Creating single cluster.")
            # Fallback: assign all points to one cluster
            cluster_labels = np.zeros(n_samples, dtype=int)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        # Calculate silhouette score if we have clusters and valid labels
        silhouette_avg = 0
        if n_clusters > 1 and len(set(cluster_labels)) > 1:
            try:
                silhouette_avg = silhouette_score(feature_matrix, cluster_labels)
            except Exception as e:
                print(f"Warning: Could not calculate silhouette score for DBSCAN: {e}")
                silhouette_avg = 0

        metadata = {
            "algorithm": "dbscan",
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "silhouette_score": silhouette_avg,
            "eps": eps,
            "min_samples": min_samples,
        }

        return cluster_labels, metadata

    def reduce_dimensions_for_visualization(
        self, feature_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reduce dimensions for visualization using PCA and t-SNE"""
        n_samples, n_features = feature_matrix.shape

        # Determine appropriate number of components for intermediate PCA
        # Use minimum of 50, n_features-1, and n_samples-1
        max_components = min(50, n_features, n_samples - 1)

        # Only do intermediate PCA if we have more than 50 features and enough samples
        if n_features > 50 and max_components >= 10:
            pca = PCA(n_components=max_components, random_state=42)
            pca_result = pca.fit_transform(feature_matrix)
        else:
            pca_result = feature_matrix

        # Ensure we have enough samples for t-SNE
        if n_samples < 4:
            print("Warning: Too few samples for t-SNE visualization. Skipping t-SNE.")
            # Just use PCA for both visualizations
            n_components_2d = min(2, pca_result.shape[1], n_samples - 1)
            if n_components_2d < 2:
                # Create dummy 2D coordinates
                pca_2d_result = np.zeros((n_samples, 2))
                tsne_result = np.zeros((n_samples, 2))
            else:
                pca_2d = PCA(n_components=n_components_2d, random_state=42)
                pca_2d_result = pca_2d.fit_transform(pca_result)
                if n_components_2d == 1:
                    # Add a second dimension of zeros
                    pca_2d_result = np.column_stack(
                        [pca_2d_result, np.zeros(n_samples)]
                    )
                tsne_result = pca_2d_result.copy()
        else:
            # t-SNE to 2 dimensions
            perplexity = min(30, max(1, (n_samples - 1) // 3))
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                tsne_result = tsne.fit_transform(pca_result)
            except Exception as e:
                print(f"Warning: t-SNE failed ({e}). Using PCA instead.")
                tsne_result = None

            # PCA to 2 dimensions for comparison
            n_components_2d = min(2, pca_result.shape[1], n_samples - 1)
            if n_components_2d < 2:
                # Create dummy 2D coordinates or use available dimensions
                if n_components_2d == 1:
                    pca_2d = PCA(n_components=1, random_state=42)
                    pca_1d = pca_2d.fit_transform(pca_result)
                    pca_2d_result = np.column_stack([pca_1d, np.zeros(n_samples)])
                else:
                    pca_2d_result = np.zeros((n_samples, 2))
            else:
                pca_2d = PCA(n_components=2, random_state=42)
                pca_2d_result = pca_2d.fit_transform(pca_result)

            # If t-SNE failed, use PCA result
            if tsne_result is None:
                tsne_result = pca_2d_result.copy()

        return pca_2d_result, tsne_result

    def analyze_issues(self, issues: List[Dict]) -> Dict:
        """Perform complete clustering analysis on issues"""
        print("Starting clustering analysis...")

        # Extract text data
        texts = [issue["full_text"] for issue in issues]

        # Extract features
        print("Extracting features...")
        self.feature_matrix = self.text_processor.extract_features(texts)

        # Determine optimal number of clusters
        print("Determining optimal number of clusters...")
        optimal_clusters = self.determine_optimal_clusters(self.feature_matrix)
        print(f"Optimal number of clusters: {optimal_clusters}")

        # Perform different clustering algorithms
        results = {}

        for algorithm in CLUSTERING_ALGORITHMS:
            print(f"Performing {algorithm} clustering...")

            if algorithm == "kmeans":
                labels, metadata = self.perform_kmeans_clustering(
                    self.feature_matrix, optimal_clusters
                )
            elif algorithm == "hierarchical":
                labels, metadata = self.perform_hierarchical_clustering(
                    self.feature_matrix, optimal_clusters
                )
            elif algorithm == "dbscan":
                labels, metadata = self.perform_dbscan_clustering(self.feature_matrix)
            else:
                continue

            # Get top terms for each cluster
            cluster_terms = {}
            unique_labels = set(labels)

            for cluster_id in unique_labels:
                if cluster_id == -1:  # Skip noise points in DBSCAN
                    continue
                terms = self.text_processor.get_top_terms_for_cluster(
                    self.feature_matrix, labels, cluster_id, 10
                )
                cluster_terms[cluster_id] = terms

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
                "issues": clustered_issues,
            }

        # Generate visualization coordinates
        print("Generating visualization coordinates...")
        pca_coords, tsne_coords = self.reduce_dimensions_for_visualization(
            self.feature_matrix
        )

        results["visualization"] = {
            "pca_coordinates": pca_coords,
            "tsne_coordinates": tsne_coords,
        }

        self.cluster_results = results
        return results


def main():
    """Test the clustering analyzer"""
    # This would be used for testing
    pass


if __name__ == "__main__":
    main()
