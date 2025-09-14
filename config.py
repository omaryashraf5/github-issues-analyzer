"""
Configuration file for GitHub Issues Analyzer
"""

# GitHub Repository Configuration
REPO_OWNER = "llamastack"
REPO_NAME = "llama-stack"
GITHUB_API_BASE = "https://api.github.com"

# Clustering Configuration
DEFAULT_NUM_CLUSTERS = 8
MIN_CLUSTERS = 3
MAX_CLUSTERS = 15

# Text Processing Configuration
MIN_WORD_LENGTH = 3
MAX_FEATURES = 1000
STOP_WORDS = "english"
NGRAM_RANGE = (1, 2)

# Clustering Algorithms to Use
CLUSTERING_ALGORITHMS = ["kmeans", "hierarchical", "dbscan"]

# Report Configuration
REPORT_TITLE = "GitHub Issues Clustering Analysis - Llama Stack"
OUTPUT_DIR = "reports"
INCLUDE_VISUALIZATIONS = True
GENERATE_WORDCLOUDS = True

# API Configuration
GITHUB_API_TIMEOUT = 30
ISSUES_PER_PAGE = 100
MAX_RETRIES = 3

# Filter Configuration
INCLUDE_CLOSED_ISSUES = False  # Only analyze open issues
EXCLUDE_PULL_REQUESTS = True
MIN_ISSUE_BODY_LENGTH = 10

# Llama Stack Configuration
LLAMASTACK_BASE_URL = "http://localhost:8321"
LLAMASTACK_DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LLAMASTACK_BATCH_SIZE = 8  # Process embeddings in batches
LLAMASTACK_TIMEOUT = 30  # Request timeout in seconds
LLAMASTACK_MAX_RETRIES = 3  # Maximum retries for failed requests
