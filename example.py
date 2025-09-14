#!/usr/bin/env python3
"""
Example usage of the GitHub Issues Analyzer

This script demonstrates how to use the individual components
of the analyzer for custom analysis workflows.
"""

from github_client import GitHubClient
from report_generator import ReportGenerator
from text_processor import ClusteringAnalyzer


def quick_analysis_example():
    """Example of a quick analysis with minimal setup"""
    print("🚀 Quick Analysis Example")
    print("=" * 50)

    # Initialize components
    client = GitHubClient()
    analyzer = ClusteringAnalyzer()
    reporter = ReportGenerator()

    # Fetch first 50 issues for quick analysis
    print("Fetching issues...")
    raw_issues = client.get_all_issues()
    issues = client.process_issues_data(raw_issues[:50])  # Limit for quick demo

    print(f"Analyzing {len(issues)} issues...")

    # Perform clustering
    results = analyzer.analyze_issues(issues)

    # Generate simple text report
    text_report = reporter.generate_cluster_summary_text(results)
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS")
    print("=" * 50)
    print(text_report)


def custom_clustering_example():
    """Example of custom clustering with specific parameters"""
    print("🎯 Custom Clustering Example")
    print("=" * 50)

    # Load pre-fetched data (if available)
    client = GitHubClient()
    try:
        issues = client.load_issues_data()
        if not issues:
            print("No cached data found. Run main.py first or set fetch=True")
            return
    except FileNotFoundError:
        print("No cached data found. Run main.py first to fetch data.")
        return

    # Custom text processing
    analyzer = ClusteringAnalyzer()

    # Override some settings for custom analysis
    from config import CLUSTERING_ALGORITHMS

    CLUSTERING_ALGORITHMS.clear()
    CLUSTERING_ALGORITHMS.append("kmeans")  # Only use K-means

    # Perform custom analysis
    print(f"Performing custom clustering on {len(issues)} issues...")
    results = analyzer.analyze_issues(issues)

    # Extract specific information
    kmeans_results = results["kmeans"]
    print(f"\nFound {kmeans_results['metadata']['n_clusters']} clusters")
    print(f"Silhouette Score: {kmeans_results['metadata']['silhouette_score']:.3f}")

    # Show top terms for each cluster
    for cluster_id, terms in kmeans_results["cluster_terms"].items():
        print(f"\nCluster {cluster_id} top terms:")
        for term, score in terms[:5]:
            print(f"  - {term}: {score:.3f}")


def repository_stats_example():
    """Example of getting repository statistics"""
    print("📊 Repository Stats Example")
    print("=" * 50)

    client = GitHubClient()

    # Get repository information
    repo_info = client.get_repository_info()

    print(f"Repository: {repo_info.get('full_name')}")
    print(f"Description: {repo_info.get('description')}")
    print(f"Stars: {repo_info.get('stargazers_count', 0):,}")
    print(f"Forks: {repo_info.get('forks_count', 0):,}")
    print(f"Open Issues: {repo_info.get('open_issues_count', 0):,}")
    print(f"Primary Language: {repo_info.get('language')}")
    print(f"Created: {repo_info.get('created_at')}")
    print(f"Last Updated: {repo_info.get('updated_at')}")


def main():
    """Run example demonstrations"""
    examples = {
        "1": ("Repository Stats", repository_stats_example),
        "2": ("Quick Analysis (50 issues)", quick_analysis_example),
        "3": ("Custom Clustering", custom_clustering_example),
    }

    print("📋 GitHub Issues Analyzer - Examples")
    print("=" * 50)
    print("Choose an example to run:")

    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")

    print("  q. Quit")

    while True:
        choice = input("\nEnter your choice: ").strip().lower()

        if choice == "q":
            print("Goodbye! 👋")
            break
        elif choice in examples:
            name, func = examples[choice]
            print(f"\n🎯 Running: {name}")
            try:
                func()
            except Exception as e:
                print(f"❌ Error: {e}")

            input("\nPress Enter to continue...")
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
