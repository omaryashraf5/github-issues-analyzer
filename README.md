# GitHub Issues Analyzer

This project analyzes GitHub issues from the llama-stack repository, clusters them based on their descriptions using natural language processing techniques, and generates comprehensive reports.

## Features

- Fetches all issues from the llama-stack repository using GitHub API
- Multiple clustering approaches:
  - **Semantic**: Uses sentence transformers for semantic similarity clustering
  - **Llama Stack**: Leverages Llama Stack server for high-quality embeddings
- Clustering algorithms: K-means, Hierarchical, DBSCAN
- Generates comprehensive reports with interactive visualizations
- Supports different clustering parameters and methods

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up GitHub API token (optional but recommended for higher rate limits):
   - Create a personal access token on GitHub
   - Set the environment variable: `export GITHUB_TOKEN=your_token_here`

## Usage

### Basic Usage (Keyword-based clustering)
```bash
python main.py
```

### Semantic Clustering with Sentence Transformers
```bash
python main.py --semantic
```

### Ollama Semantic Clustering (Recommended)
First, start your Ollama server:
```bash
ollama serve
```

Pull an embedding model:
```bash
ollama pull nomic-embed-text
```

Then run the analyzer:
```bash
python main.py --ollama
```

Or with custom settings:
```bash
python main.py --ollama --ollama-url http://localhost:11434 --ollama-model nomic-embed-text
```

### Test Ollama Connection
Before running the full analysis, test your connection:
```bash
python test_ollama.py
```

### Llama Stack Semantic Clustering
First, start your Llama Stack server:
```bash
llama stack run
```

Then run the analyzer:
```bash
python main.py --llamastack
```

Or with custom server settings:
```bash
python main.py --llamastack --llamastack-url http://localhost:8321 --llamastack-model meta-llama/Llama-3.2-3B-Instruct
```

### Test Llama Stack Connection
Before running the full analysis, test your connection:
```bash
python test_llamastack.py
```

The script will:
1. Fetch all issues from the llama-stack repository
2. Preprocess the text data
3. Perform clustering analysis (keyword-based, semantic, or Llama Stack)
4. Generate reports in the `reports/` directory

## Output

The analysis generates:
- `issues_data.json`: Raw issue data
- `clustering_report.html`: Interactive HTML report with visualizations
- `cluster_summary.txt`: Text summary of clusters
- `issues_by_cluster.csv`: CSV file with issues assigned to clusters

## Configuration

Modify `config.py` to adjust:
- Number of clusters
- Clustering algorithms
- Text preprocessing parameters
- Report formatting options
