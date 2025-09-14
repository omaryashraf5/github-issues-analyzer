#!/usr/bin/env python3
"""
GitHub Issues Analyzer - Main Script

This script analyzes GitHub issues from the llama-stack repository,
clusters them based on their descriptions, and generates comprehensive reports.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

# Import our modules
from github_client import GitHubClient
from llamastack_processor import LlamaStackClusteringAnalyzer
from ollama_processor import OllamaClusteringAnalyzer
from report_generator import ReportGenerator
from semantic_processor import SemanticClusteringAnalyzer
from text_processor import ClusteringAnalyzer
from cluster_issues_exporter import generate_cluster_issues_csv
from config import *


def setup_output_directory() -> str:
    """Create and setup output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{OUTPUT_DIR}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        GitHub Issues Analyzer                               ║
║                     Clustering Analysis for Llama Stack                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def validate_environment():
    """Validate that all required dependencies are available"""
    try:
        import jinja2
        import matplotlib
        import nltk
        import numpy
        import pandas
        import plotly
        import requests
        import seaborn
        import sklearn
        import wordcloud

        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages: pip install -r requirements.txt")
        return False


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Analyze GitHub issues and generate clustering reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with default settings
  python main.py --clusters 10            # Force 10 clusters
  python main.py --algorithm kmeans       # Use only K-means
  python main.py --no-visualizations      # Skip visualizations
  python main.py --github-token TOKEN     # Use specific GitHub token
        """,
    )

    parser.add_argument(
        "--clusters",
        "-c",
        type=int,
        help=f"Number of clusters (default: auto-detect, max: {MAX_CLUSTERS})",
    )

    parser.add_argument(
        "--algorithm",
        "-a",
        choices=["kmeans", "hierarchical", "dbscan", "all"],
        default="all",
        help="Clustering algorithm to use (default: all)",
    )

    parser.add_argument(
        "--github-token",
        "-t",
        type=str,
        help="GitHub API token (can also use GITHUB_TOKEN env var)",
    )

    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip generating visualizations",
    )

    parser.add_argument(
        "--no-wordclouds", action="store_true", help="Skip generating word clouds"
    )

    parser.add_argument(
        "--output-dir", "-o", type=str, help="Output directory for reports"
    )

    parser.add_argument(
        "--fresh-data",
        action="store_true",
        help="Force fresh data download (ignore cached data)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only fetch and process data, skip clustering and reports",
    )

    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Use semantic clustering instead of keyword-based clustering",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        choices=[
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-mpnet-base-v2",
            "all-distilroberta-v1",
        ],
        help="Sentence transformer model for semantic clustering (default: all-MiniLM-L6-v2)",
    )

    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode (no model downloads, falls back to TF-IDF if needed)",
    )

    parser.add_argument(
        "--llamastack",
        action="store_true",
        help="Use Llama Stack server for embeddings (requires running llama-stack server)",
    )

    parser.add_argument(
        "--llamastack-url",
        type=str,
        default="http://localhost:8321",
        help="Llama Stack server URL (default: http://localhost:8321)",
    )

    parser.add_argument(
        "--llamastack-model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Llama Stack model ID for embeddings (default: meta-llama/Llama-3.2-3B-Instruct)",
    )

    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Use Ollama server for embeddings (requires running ollama serve)",
    )

    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )

    parser.add_argument(
        "--ollama-model",
        type=str,
        default="nomic-embed-text",
        help="Ollama model name for embeddings (default: nomic-embed-text)",
    )

    parser.add_argument(
        "--include-closed",
        action="store_true",
        help="Include closed issues in analysis (default: only open issues)",
    )

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Validate environment
    print("🔍 Validating environment...")
    if not validate_environment():
        sys.exit(1)
    print("✅ Environment validation passed")

    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = setup_output_directory()

    print(f"📁 Output directory: {output_dir}")

    # Override global config based on arguments
    import config

    if args.include_closed:
        config.INCLUDE_CLOSED_ISSUES = True
        print("📌 Including both open and closed issues")
    else:
        config.INCLUDE_CLOSED_ISSUES = False
        print("📌 Analyzing only open issues")

    # Initialize components
    github_client = GitHubClient(token=args.github_token)

    # Choose clustering approach based on arguments
    if args.ollama:
        print(f"🦙 Using Ollama semantic clustering")
        print(f"   Server URL: {args.ollama_url}")
        print(f"   Model: {args.ollama_model}")
        clustering_analyzer = OllamaClusteringAnalyzer(
            base_url=args.ollama_url, model_name=args.ollama_model
        )
        analysis_method = "ollama"
    elif args.llamastack:
        print(f"🦙 Using Llama Stack semantic clustering")
        print(f"   Server URL: {args.llamastack_url}")
        print(f"   Model: {args.llamastack_model}")
        clustering_analyzer = LlamaStackClusteringAnalyzer(
            base_url=args.llamastack_url, model_id=args.llamastack_model
        )
        analysis_method = "llamastack"
    elif args.semantic:
        print(f"🧠 Using semantic clustering with model: {args.embedding_model}")
        clustering_analyzer = SemanticClusteringAnalyzer(
            model_name=args.embedding_model
        )
        analysis_method = "semantic"
    else:
        print("🔤 Using keyword-based clustering")
        clustering_analyzer = ClusteringAnalyzer()
        analysis_method = "keyword"

    report_generator = ReportGenerator(output_dir)

    start_time = time.time()

    try:
        # Step 1: Fetch repository information
        print("\n" + "=" * 80)
        print("📊 FETCHING REPOSITORY INFORMATION")
        print("=" * 80)

        repo_info = github_client.get_repository_info()
        print(f"Repository: {repo_info.get('full_name', 'N/A')}")
        print(f"Description: {repo_info.get('description', 'N/A')}")
        print(f"Stars: {repo_info.get('stargazers_count', 0):,}")
        print(f"Forks: {repo_info.get('forks_count', 0):,}")
        print(f"Open Issues: {repo_info.get('open_issues_count', 0):,}")
        print(f"Language: {repo_info.get('language', 'N/A')}")

        # Step 2: Fetch and process issues
        print("\n" + "=" * 80)
        print("📥 FETCHING GITHUB ISSUES")
        print("=" * 80)

        issues_data_file = os.path.join(output_dir, "issues_data.json")

        if not args.fresh_data and os.path.exists(issues_data_file):
            print("📂 Loading cached issues data...")
            issues = github_client.load_issues_data(issues_data_file)
            print(f"Loaded {len(issues)} issues from cache")
        else:
            print("🌐 Fetching fresh issues data...")
            raw_issues = github_client.get_all_issues()
            print(f"Fetched {len(raw_issues)} raw issues from GitHub")
            issues = github_client.process_issues_data(raw_issues)
            print(f"Processed to {len(issues)} valid issues")
            github_client.save_issues_data(issues, issues_data_file)

        if not issues:
            print("❌ No issues found or failed to fetch issues")
            sys.exit(1)

        print(f"✅ Successfully processed {len(issues)} issues")

        # Print some statistics
        open_issues = sum(1 for issue in issues if issue["state"] == "open")
        closed_issues = len(issues) - open_issues

        print(f"   📊 Open issues: {open_issues:,}")
        print(f"   📊 Closed issues: {closed_issues:,}")

        # Get unique labels
        all_labels = set()
        for issue in issues:
            all_labels.update(issue["labels"])
        print(f"   📊 Unique labels: {len(all_labels)}")

        if args.dry_run:
            print("\n🏁 Dry run completed. Exiting...")
            sys.exit(0)

        # Step 3: Perform clustering analysis
        print("\n" + "=" * 80)
        print("🤖 PERFORMING CLUSTERING ANALYSIS")
        print("=" * 80)

        # Override algorithm selection if specified
        if args.algorithm != "all":
            global CLUSTERING_ALGORITHMS
            CLUSTERING_ALGORITHMS = [args.algorithm]

        # Override cluster count if specified
        if args.clusters:
            global DEFAULT_NUM_CLUSTERS
            DEFAULT_NUM_CLUSTERS = args.clusters

        # Override visualization settings
        if args.no_visualizations:
            global INCLUDE_VISUALIZATIONS
            INCLUDE_VISUALIZATIONS = False

        if args.no_wordclouds:
            global GENERATE_WORDCLOUDS
            GENERATE_WORDCLOUDS = False

        # Perform clustering
        if args.ollama:
            clustering_results = clustering_analyzer.analyze_issues_semantic(issues)
        elif args.llamastack:
            clustering_results = clustering_analyzer.analyze_issues_semantic(issues)
        elif args.semantic:
            clustering_results = clustering_analyzer.analyze_issues_semantic(issues)
        else:
            clustering_results = clustering_analyzer.analyze_issues(issues)

        if not clustering_results:
            print("❌ Clustering analysis failed")
            sys.exit(1)

        # Print clustering summary
        for algorithm, results in clustering_results.items():
            if algorithm == "visualization":
                continue

            metadata = results["metadata"]
            print(f"\n📊 {algorithm.upper()} Results:")
            print(f"   Clusters: {metadata['n_clusters']}")
            print(f"   Silhouette Score: {metadata['silhouette_score']:.3f}")

            if algorithm == "dbscan":
                print(f"   Noise Points: {metadata.get('n_noise_points', 0)}")

        # Step 4: Generate reports
        print("\n" + "=" * 80)
        print("📋 GENERATING REPORTS")
        print("=" * 80)

        # Generate reports
        report_files = report_generator.generate_complete_report(
            clustering_results, issues
        )

        # Generate cluster issues summary CSV if not already done by report generator
        if "cluster_csv" not in report_files:
            print("📊 Generating cluster issues summary CSV...")
            try:
                csv_path = generate_cluster_issues_csv(output_dir)
                if csv_path:
                    report_files["cluster_csv"] = os.path.basename(csv_path)
                    print(f"✅ Cluster issues CSV generated successfully")
            except Exception as e:
                print(f"⚠️ Could not generate cluster issues CSV: {str(e)}")

        # Step 5: Summary
        end_time = time.time()
        execution_time = end_time - start_time

        print("\n" + "=" * 80)
        print("🎉 ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 80)

        print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        print(f"📁 Output Directory: {output_dir}")
        print(f"📊 Issues Analyzed: {len(issues):,}")

        print("\n📄 Generated Reports:")
        for report_type, filename in report_files.items():
            filepath = os.path.join(output_dir, filename)
            file_size = os.path.getsize(filepath) / 1024  # KB
            print(f"   {report_type.upper()}: {filename} ({file_size:.1f} KB)")

        print(
            f"\n🌐 To view the HTML report, open: {os.path.join(output_dir, report_files.get('html', ''))}"
        )

        # Performance metrics
        print(f"\n📈 Performance Metrics:")
        print(f"   Issues per second: {len(issues)/execution_time:.1f}")
        print(
            f"   Memory usage: {sys.getsizeof(clustering_results) / 1024 / 1024:.1f} MB"
        )

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print(f"📍 Error type: {type(e).__name__}")

        # Print some debugging info
        import traceback

        print("\n🔍 Debug Information:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
