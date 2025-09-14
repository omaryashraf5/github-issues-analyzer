"""
Report generation module for GitHub issues clustering analysis
"""

import json
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import seaborn as sns
from jinja2 import Template
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from config import *


class ReportGenerator:
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set matplotlib style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def generate_cluster_theme_description(
        self, cluster_issues: List[Dict], cluster_terms: List[Tuple], cluster_id: int
    ) -> str:
        """Generate a thematic description for a cluster based on issue content"""

        # Analyze issue titles and bodies for common patterns
        titles = [issue["title"].lower() for issue in cluster_issues]
        bodies = [issue.get("body", "").lower() for issue in cluster_issues]
        labels = []
        for issue in cluster_issues:
            labels.extend(issue.get("labels", []))

        # Get most common labels
        label_counts = Counter(labels)
        top_labels = [label for label, count in label_counts.most_common(3)]

        # Get top terms for context
        top_terms = [term[0] for term in cluster_terms[:8]]

        # Analyze patterns in titles to determine theme
        title_text = " ".join(titles)
        body_text = " ".join(bodies)
        combined_text = title_text + " " + body_text

        # Common patterns and their themes
        patterns = {
            # Documentation and guides
            "documentation": [
                "documentation",
                "docs",
                "guide",
                "tutorial",
                "readme",
                "example",
                "sample",
            ],
            "setup_installation": [
                "install",
                "setup",
                "configuration",
                "config",
                "deploy",
                "deployment",
                "getting started",
            ],
            # Technical issues
            "api_integration": [
                "api",
                "endpoint",
                "integration",
                "client",
                "sdk",
                "interface",
            ],
            "authentication": [
                "auth",
                "authentication",
                "login",
                "token",
                "credential",
                "permission",
            ],
            "performance": [
                "performance",
                "slow",
                "optimization",
                "memory",
                "cpu",
                "latency",
                "speed",
            ],
            "compatibility": [
                "compatibility",
                "version",
                "support",
                "breaking",
                "migration",
                "upgrade",
            ],
            # Errors and bugs
            "error_handling": [
                "error",
                "exception",
                "crash",
                "fail",
                "failure",
                "bug",
                "broken",
            ],
            "connectivity": [
                "connection",
                "network",
                "timeout",
                "unreachable",
                "connectivity",
            ],
            # Features and enhancements
            "feature_request": [
                "feature",
                "enhancement",
                "improvement",
                "add",
                "support for",
                "request",
            ],
            "ui_ux": [
                "ui",
                "interface",
                "user experience",
                "frontend",
                "display",
                "visualization",
            ],
            # Data and models
            "model_issues": [
                "model",
                "inference",
                "prediction",
                "training",
                "weights",
                "checkpoint",
            ],
            "data_processing": [
                "data",
                "dataset",
                "processing",
                "format",
                "parsing",
                "loading",
            ],
            # Development
            "testing": [
                "test",
                "testing",
                "unit test",
                "integration test",
                "validation",
            ],
            "build_deployment": [
                "build",
                "compile",
                "deploy",
                "ci",
                "pipeline",
                "automation",
            ],
        }

        # Score each theme based on keyword frequency
        theme_scores = {}
        for theme, keywords in patterns.items():
            score = 0
            for keyword in keywords:
                score += combined_text.count(keyword)
                score += sum(1 for term in top_terms if keyword in term.lower())
                score += sum(1 for label in top_labels if keyword in label.lower())
            theme_scores[theme] = score

        # Find the best matching theme
        best_theme = max(theme_scores.items(), key=lambda x: x[1])
        theme_name, theme_score = best_theme

        # Generate description based on theme
        descriptions = {
            "documentation": "Issues related to documentation, guides, examples, and educational content",
            "setup_installation": "Issues about installation, setup, configuration, and getting started",
            "api_integration": "Issues involving API usage, integrations, client libraries, and interfaces",
            "authentication": "Issues related to authentication, authorization, tokens, and access control",
            "performance": "Issues about performance optimization, speed, memory usage, and efficiency",
            "compatibility": "Issues related to version compatibility, migration, and platform support",
            "error_handling": "Issues reporting bugs, errors, crashes, and unexpected failures",
            "connectivity": "Issues related to network connections, timeouts, and connectivity problems",
            "feature_request": "Issues requesting new features, enhancements, and improvements",
            "ui_ux": "Issues about user interface, user experience, and frontend functionality",
            "model_issues": "Issues related to AI models, inference, training, and model-specific problems",
            "data_processing": "Issues about data handling, processing, formats, and dataset management",
            "testing": "Issues related to testing, validation, and quality assurance",
            "build_deployment": "Issues about build processes, deployment, CI/CD, and automation",
        }

        # If no strong theme match, create a generic description based on top terms and labels
        if theme_score == 0:
            if top_labels:
                return f"Issues primarily labeled as '{', '.join(top_labels[:2])}' focusing on {', '.join(top_terms[:3])}"
            else:
                return f"Issues focusing on {', '.join(top_terms[:4])} and related functionality"

        base_description = descriptions.get(
            theme_name, "Issues with mixed topics and concerns"
        )

        # Add context from labels if available
        if top_labels:
            label_context = f" (commonly labeled: {', '.join(top_labels[:2])})"
            base_description += label_context

        return base_description

    def extract_core_theme(self, theme_description: str) -> str:
        """Extract core theme without label information"""
        # Remove everything in parentheses (commonly labeled: ...)
        import re

        core_theme = re.split(r"\s*\(commonly labeled:", theme_description)[0].strip()
        return core_theme

    def merge_similar_theme_clusters(
        self, results: Dict, algorithm: str = "kmeans"
    ) -> Dict:
        """Merge clusters that have identical core themes"""
        if algorithm not in results:
            return results

        cluster_data = results[algorithm]
        cluster_terms = cluster_data["cluster_terms"]
        issues = cluster_data["issues"]

        # Group clusters by core theme (without label info)
        core_theme_to_clusters = {}
        cluster_themes = {}
        cluster_core_themes = {}

        for cluster_id in cluster_terms.keys():
            if cluster_id == -1:
                continue

            cluster_issues = [
                issue for issue in issues if issue["cluster"] == cluster_id
            ]
            top_terms = cluster_terms[cluster_id]
            full_theme_description = self.generate_cluster_theme_description(
                cluster_issues, top_terms, cluster_id
            )
            core_theme = self.extract_core_theme(full_theme_description)

            cluster_themes[cluster_id] = full_theme_description
            cluster_core_themes[cluster_id] = core_theme

            if core_theme not in core_theme_to_clusters:
                core_theme_to_clusters[core_theme] = []
            core_theme_to_clusters[core_theme].append(cluster_id)

        # Find core themes with multiple clusters
        themes_to_merge = {
            theme: clusters
            for theme, clusters in core_theme_to_clusters.items()
            if len(clusters) > 1
        }

        if not themes_to_merge:
            print("No clusters with identical core themes found for merging")
            return results

        print(f"Found {len(themes_to_merge)} core themes with multiple clusters:")
        for theme, clusters in themes_to_merge.items():
            print(f"  '{theme}': clusters {clusters}")

        # Create cluster mapping for merging
        cluster_mapping = {}
        merged_cluster_terms = {}

        # Process themes with multiple clusters first
        for core_theme, clusters_to_merge in themes_to_merge.items():
            # Use the first cluster as the target
            target_cluster = min(clusters_to_merge)

            print(
                f"Merging clusters {clusters_to_merge} -> {target_cluster} (theme: '{core_theme}')"
            )

            # Map all clusters in this theme to the target
            for cluster_id in clusters_to_merge:
                cluster_mapping[cluster_id] = target_cluster

            # Combine terms from all clusters in this theme
            combined_terms = {}
            for cluster_id in clusters_to_merge:
                for term, score in cluster_terms[cluster_id]:
                    if term in combined_terms:
                        combined_terms[term] = max(combined_terms[term], score)
                    else:
                        combined_terms[term] = score

            # Sort by score and take top terms
            merged_cluster_terms[target_cluster] = sorted(
                combined_terms.items(), key=lambda x: x[1], reverse=True
            )[:10]

        # Handle single-cluster themes
        for core_theme, clusters in core_theme_to_clusters.items():
            if len(clusters) == 1:
                cluster_id = clusters[0]
                cluster_mapping[cluster_id] = cluster_id
                merged_cluster_terms[cluster_id] = cluster_terms[cluster_id]

        # Apply cluster mapping to issues
        merged_issues = []
        for issue in issues:
            issue_copy = issue.copy()
            original_cluster = issue_copy["cluster"]
            issue_copy["cluster"] = cluster_mapping.get(
                original_cluster, original_cluster
            )
            merged_issues.append(issue_copy)

        # Update metadata
        merged_metadata = cluster_data["metadata"].copy()
        merged_metadata["n_clusters"] = len(set(cluster_mapping.values()))
        merged_metadata["original_clusters"] = cluster_data["metadata"]["n_clusters"]
        merged_metadata["merged_themes"] = len(themes_to_merge)
        merged_metadata["cluster_mapping"] = cluster_mapping

        # Recalculate labels array
        new_labels = []
        for issue in issues:
            original_cluster = issue["cluster"]
            new_cluster = cluster_mapping.get(original_cluster, original_cluster)
            new_labels.append(new_cluster)

        # Renumber clusters to be consecutive (0, 1, 2, ...)
        unique_clusters = sorted(set(cluster_mapping.values()))
        renumber_mapping = {
            old_id: new_id for new_id, old_id in enumerate(unique_clusters)
        }

        print(f"Renumbering clusters: {renumber_mapping}")

        # Apply renumbering to all data structures
        final_cluster_mapping = {}
        for orig_id, merged_id in cluster_mapping.items():
            final_cluster_mapping[orig_id] = renumber_mapping[merged_id]

        # Renumber cluster terms
        renumbered_cluster_terms = {}
        for old_id, terms in merged_cluster_terms.items():
            new_id = renumber_mapping[old_id]
            renumbered_cluster_terms[new_id] = terms

        # Renumber cluster themes
        renumbered_cluster_themes = {}
        for orig_cid in cluster_core_themes.keys():
            old_merged_id = cluster_mapping.get(orig_cid, orig_cid)
            new_id = renumber_mapping[old_merged_id]
            renumbered_cluster_themes[new_id] = cluster_core_themes[orig_cid]

        # Apply final renumbering to issues and labels
        final_issues = []
        final_labels = []
        for issue in merged_issues:
            issue_copy = issue.copy()
            old_cluster = issue_copy["cluster"]
            new_cluster = renumber_mapping[old_cluster]
            issue_copy["cluster"] = new_cluster
            final_issues.append(issue_copy)
            final_labels.append(new_cluster)

        # Create merged results with renumbered clusters
        merged_results = results.copy()
        merged_results[f"{algorithm}_merged"] = {
            "labels": np.array(final_labels),
            "metadata": merged_metadata,
            "cluster_terms": renumbered_cluster_terms,
            "cluster_themes": renumbered_cluster_themes,
            "issues": final_issues,
            "cluster_mapping": final_cluster_mapping,
            "renumber_mapping": renumber_mapping,
        }

        return merged_results

    def generate_cluster_summary_text(
        self, results: Dict, algorithm: str = "kmeans"
    ) -> str:
        """Generate a text summary of clustering results"""

        # Check if we should use merged results
        merged_algorithm = f"{algorithm}_merged"
        if merged_algorithm in results:
            print(f"Using merged clusters for summary ({merged_algorithm})")
            algorithm = merged_algorithm

        if algorithm not in results:
            return "No results available for the specified algorithm."

        cluster_data = results[algorithm]
        metadata = cluster_data["metadata"]
        cluster_terms = cluster_data["cluster_terms"]
        issues = cluster_data["issues"]

        summary = []
        summary.append(
            f"=== GitHub Issues Clustering Analysis - {algorithm.upper()} ==="
        )
        summary.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Repository: {REPO_OWNER}/{REPO_NAME}")
        summary.append("")
        summary.append(f"Algorithm: {metadata['algorithm']}")
        summary.append(f"Number of Clusters: {metadata['n_clusters']}")

        # Show original vs merged cluster info if applicable
        if "original_clusters" in metadata:
            summary.append(f"Original Clusters: {metadata['original_clusters']}")
            summary.append(f"Merged Themes: {metadata['merged_themes']}")

        summary.append(f"Silhouette Score: {metadata['silhouette_score']:.3f}")
        summary.append(f"Total Issues Analyzed: {len(issues)}")
        summary.append("")

        # Cluster distribution
        cluster_counts = Counter([issue["cluster"] for issue in issues])
        summary.append("Cluster Distribution:")
        for cluster_id, count in sorted(cluster_counts.items()):
            percentage = (count / len(issues)) * 100
            summary.append(
                f"  Cluster {cluster_id}: {count} issues ({percentage:.1f}%)"
            )
        summary.append("")

        # Detailed cluster analysis with themes
        for cluster_id in sorted(cluster_terms.keys()):
            if cluster_id == -1:
                continue

            cluster_issues = [
                issue for issue in issues if issue["cluster"] == cluster_id
            ]
            top_terms = cluster_terms[cluster_id]

            summary.append(f"--- Cluster {cluster_id} ---")
            summary.append(f"Issues: {len(cluster_issues)}")

            # Generate thematic description
            theme_description = self.generate_cluster_theme_description(
                cluster_issues, top_terms, cluster_id
            )
            summary.append(f"Theme: {theme_description}")

            # Most common labels in this cluster
            all_labels = []
            for issue in cluster_issues:
                all_labels.extend(issue["labels"])

            if all_labels:
                label_counts = Counter(all_labels)
                top_labels = label_counts.most_common(3)
                summary.append(
                    f"Common Labels: {', '.join([f'{label}({count})' for label, count in top_labels])}"
                )

            # All issue numbers in this cluster
            issue_numbers = sorted([issue["number"] for issue in cluster_issues])
            summary.append(f"Issue Numbers: {', '.join(map(str, issue_numbers))}")

            # Sample issues
            summary.append("Sample Issues:")
            for i, issue in enumerate(cluster_issues[:3]):
                title = (
                    issue["title"][:80] + "..."
                    if len(issue["title"]) > 80
                    else issue["title"]
                )
                summary.append(f"  #{issue['number']}: {title}")

            summary.append("")

        return "\n".join(summary)

    def generate_wordcloud(self, cluster_terms: Dict, cluster_id: int) -> str:
        """Generate word cloud for a cluster"""
        if cluster_id not in cluster_terms:
            return None

        terms_dict = {term[0]: term[1] for term in cluster_terms[cluster_id]}

        if not terms_dict:
            return None

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=50,
            colormap="viridis",
        ).generate_from_frequencies(terms_dict)

        # Save wordcloud
        filename = f"wordcloud_cluster_{cluster_id}.png"
        filepath = os.path.join(self.output_dir, filename)
        wordcloud.to_file(filepath)

        return filename

    def create_cluster_visualization(
        self, results: Dict, algorithm: str = "kmeans"
    ) -> str:
        """Create interactive cluster visualization"""
        if algorithm not in results or "visualization" not in results:
            return None

        cluster_data = results[algorithm]
        issues = cluster_data["issues"]
        viz_data = results["visualization"]

        # Use t-SNE coordinates for better visualization
        coords = viz_data["tsne_coordinates"]

        # Prepare data for plotting
        df = pd.DataFrame(
            {
                "x": coords[:, 0],
                "y": coords[:, 1],
                "cluster": [issue["cluster"] for issue in issues],
                "title": [issue["title"] for issue in issues],
                "number": [issue["number"] for issue in issues],
                "state": [issue["state"] for issue in issues],
                "labels": [", ".join(issue["labels"]) for issue in issues],
            }
        )

        # Create interactive scatter plot
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="cluster",
            hover_data=["number", "state", "labels"],
            hover_name="title",
            title=f"GitHub Issues Clustering Visualization - {algorithm.upper()}",
            color_continuous_scale="viridis",
        )

        fig.update_layout(
            width=1000,
            height=600,
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
        )

        # Save interactive plot
        filename = f"cluster_visualization_{algorithm}.html"
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)

        return filename

    def create_analysis_charts(
        self, results: Dict, algorithm: str = "kmeans"
    ) -> List[str]:
        """Create various analysis charts"""
        if algorithm not in results:
            return []

        cluster_data = results[algorithm]
        issues = cluster_data["issues"]

        chart_files = []

        # 1. Cluster size distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        cluster_counts = Counter([issue["cluster"] for issue in issues])
        clusters = list(cluster_counts.keys())
        counts = list(cluster_counts.values())

        ax1.bar(clusters, counts, color="skyblue", alpha=0.7)
        ax1.set_xlabel("Cluster ID")
        ax1.set_ylabel("Number of Issues")
        ax1.set_title("Issues per Cluster")
        ax1.grid(True, alpha=0.3)

        # 2. Issues state distribution by cluster
        cluster_states = {}
        for issue in issues:
            cluster_id = issue["cluster"]
            state = issue["state"]
            if cluster_id not in cluster_states:
                cluster_states[cluster_id] = {"open": 0, "closed": 0}
            cluster_states[cluster_id][state] += 1

        clusters = sorted(cluster_states.keys())
        open_counts = [cluster_states[c]["open"] for c in clusters]
        closed_counts = [cluster_states[c]["closed"] for c in clusters]

        width = 0.35
        x = np.arange(len(clusters))

        ax2.bar(x - width / 2, open_counts, width, label="Open", color="red", alpha=0.7)
        ax2.bar(
            x + width / 2,
            closed_counts,
            width,
            label="Closed",
            color="green",
            alpha=0.7,
        )
        ax2.set_xlabel("Cluster ID")
        ax2.set_ylabel("Number of Issues")
        ax2.set_title("Issue State Distribution by Cluster")
        ax2.set_xticks(x)
        ax2.set_xticklabels(clusters)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filename1 = f"cluster_analysis_{algorithm}.png"
        filepath1 = os.path.join(self.output_dir, filename1)
        plt.savefig(filepath1, dpi=300, bbox_inches="tight")
        plt.close()
        chart_files.append(filename1)

        # 3. Timeline analysis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Convert dates and group by month
        dates = []
        cluster_labels = []

        for issue in issues:
            try:
                date = pd.to_datetime(issue["created_at"]).to_period("M")
                dates.append(date)
                cluster_labels.append(issue["cluster"])
            except:
                continue

        if dates:
            df_timeline = pd.DataFrame({"date": dates, "cluster": cluster_labels})

            # Create pivot table for stacked area chart
            timeline_pivot = (
                df_timeline.groupby(["date", "cluster"]).size().unstack(fill_value=0)
            )

            # Plot stacked area chart
            timeline_pivot.plot(kind="area", stacked=True, ax=ax, alpha=0.7)
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Issues")
            ax.set_title("Issue Creation Timeline by Cluster")
            ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename2 = f"timeline_analysis_{algorithm}.png"
        filepath2 = os.path.join(self.output_dir, filename2)
        plt.savefig(filepath2, dpi=300, bbox_inches="tight")
        plt.close()
        chart_files.append(filename2)

        return chart_files

    def generate_html_report(self, results: Dict, issues: List[Dict]) -> str:
        """Generate comprehensive HTML report"""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #333;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
        }
        h1 { color: #2c3e50; }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .summary-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            border: none;
            color: white;
        }
        .summary-card .value {
            font-size: 2em;
            font-weight: bold;
        }
        .cluster-section {
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fafafa;
        }
        .cluster-header {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .terms-list {
            background: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }
        .issues-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: white;
        }
        .issues-table th, .issues-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .issues-table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .issues-table tr:hover {
            background-color: #f5f5f5;
        }
        .state-open {
            color: #d73027;
            font-weight: bold;
        }
        .state-closed {
            color: #1a9641;
            font-weight: bold;
        }
        .label-tag {
            display: inline-block;
            background: #e3f2fd;
            color: #1976d2;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin: 2px;
        }
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .algorithm-tabs {
            display: flex;
            margin: 20px 0;
        }
        .algorithm-tab {
            padding: 10px 20px;
            background: #e0e0e0;
            border: none;
            cursor: pointer;
            margin-right: 5px;
            border-radius: 5px 5px 0 0;
        }
        .algorithm-tab.active {
            background: #4CAF50;
            color: white;
        }
        .wordcloud-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <p><strong>Generated:</strong> {{ generation_date }}</p>
        <p><strong>Repository:</strong> <a href="https://github.com/{{ repo_owner }}/{{ repo_name }}">{{ repo_owner }}/{{ repo_name }}</a></p>

        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Issues</h3>
                <div class="value">{{ total_issues }}</div>
            </div>
            <div class="summary-card">
                <h3>Clusters Found</h3>
                <div class="value">{{ num_clusters }}</div>
            </div>
            <div class="summary-card">
                <h3>Silhouette Score</h3>
                <div class="value">{{ silhouette_score }}</div>
            </div>
            <div class="summary-card">
                <h3>Algorithm</h3>
                <div class="value">{{ algorithm_name }}</div>
            </div>
        </div>

        {% if chart_files %}
        <h2>📊 Analysis Charts</h2>
        <div class="chart-container">
            {% for chart in chart_files %}
            <img src="{{ chart }}" alt="Analysis Chart">
            {% endfor %}
        </div>
        {% endif %}

        {% if visualization_file %}
        <h2>🔍 Interactive Visualization</h2>
        <p><a href="{{ visualization_file }}" target="_blank">Open Interactive Cluster Visualization</a></p>
        {% endif %}

        <h2>🎯 Cluster Analysis</h2>
        {% for cluster_id, cluster_info in clusters.items() %}
        <div class="cluster-section">
            <div class="cluster-header">
                <h3>Cluster {{ cluster_id }} ({{ cluster_info.size }} issues)</h3>
            </div>

            <div class="terms-list">
                <strong>Top Terms:</strong> {{ cluster_info.top_terms }}
            </div>

            {% if cluster_info.wordcloud %}
            <div class="chart-container">
                <h4>Word Cloud</h4>
                <img src="{{ cluster_info.wordcloud }}" alt="Word Cloud for Cluster {{ cluster_id }}">
            </div>
            {% endif %}

            <h4>Issues in this Cluster:</h4>
            <table class="issues-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Title</th>
                        <th>State</th>
                        <th>Labels</th>
                        <th>Created</th>
                    </tr>
                </thead>
                <tbody>
                    {% for issue in cluster_info.issues[:10] %}
                    <tr>
                        <td><a href="{{ issue.url }}" target="_blank">#{{ issue.number }}</a></td>
                        <td>{{ issue.title }}</td>
                        <td class="state-{{ issue.state }}">{{ issue.state }}</td>
                        <td>
                            {% for label in issue.labels %}
                            <span class="label-tag">{{ label }}</span>
                            {% endfor %}
                        </td>
                        <td>{{ issue.created_at[:10] }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% if cluster_info.size > 10 %}
            <p><em>... and {{ cluster_info.size - 10 }} more issues</em></p>
            {% endif %}
        </div>
        {% endfor %}

        <h2>📈 Summary Statistics</h2>
        <pre>{{ summary_text }}</pre>
    </div>
</body>
</html>
        """

        # Prepare data for template
        algorithm = "kmeans"  # Default to kmeans for main report
        cluster_data = results[algorithm]
        metadata = cluster_data["metadata"]

        # Prepare cluster information
        clusters_info = {}
        for cluster_id, terms in cluster_data["cluster_terms"].items():
            if cluster_id == -1:
                continue

            cluster_issues = [
                issue
                for issue in cluster_data["issues"]
                if issue["cluster"] == cluster_id
            ]

            # Generate wordcloud
            wordcloud_file = None
            if GENERATE_WORDCLOUDS:
                wordcloud_file = self.generate_wordcloud(
                    cluster_data["cluster_terms"], cluster_id
                )

            # Get additional cluster information if available (semantic clustering)
            cluster_info = {
                "size": len(cluster_issues),
                "top_terms": ", ".join([term[0] for term in terms[:10]]),
                "issues": cluster_issues,
                "wordcloud": wordcloud_file,
            }

            # Add semantic-specific information if available
            if (
                "cluster_themes" in cluster_data
                and cluster_id in cluster_data["cluster_themes"]
            ):
                theme_data = cluster_data["cluster_themes"][cluster_id]
                cluster_info.update(
                    {
                        "cohesion": f"{theme_data.get('cohesion', 0):.3f}",
                        "representative_text": theme_data.get(
                            "most_representative_text", ""
                        ),
                        "semantic_quality": f"{theme_data.get('centroid_similarity_avg', 0):.3f}",
                    }
                )

            clusters_info[cluster_id] = cluster_info

        # Generate charts
        chart_files = self.create_analysis_charts(results, algorithm)

        # Generate visualization
        visualization_file = None
        if INCLUDE_VISUALIZATIONS:
            visualization_file = self.create_cluster_visualization(results, algorithm)

        # Generate summary text
        summary_text = self.generate_cluster_summary_text(results, algorithm)

        # Render template
        template = Template(html_template)
        html_content = template.render(
            title=REPORT_TITLE,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            repo_owner=REPO_OWNER,
            repo_name=REPO_NAME,
            total_issues=len(issues),
            num_clusters=metadata["n_clusters"],
            silhouette_score=f"{metadata['silhouette_score']:.3f}",
            algorithm_name=algorithm.upper(),
            clusters=clusters_info,
            chart_files=chart_files,
            visualization_file=visualization_file,
            summary_text=summary_text,
        )

        # Save HTML report
        html_filename = "clustering_report.html"
        html_filepath = os.path.join(self.output_dir, html_filename)
        with open(html_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_filename

    def save_issues_csv(self, results: Dict, algorithm: str = "kmeans") -> str:
        """Save clustered issues to CSV"""
        if algorithm not in results:
            return None

        issues = results[algorithm]["issues"]

        # Prepare data for CSV
        csv_data = []
        for issue in issues:
            csv_data.append(
                {
                    "issue_number": issue["number"],
                    "title": issue["title"],
                    "state": issue["state"],
                    "cluster": issue["cluster"],
                    "created_at": issue["created_at"],
                    "updated_at": issue["updated_at"],
                    "labels": "; ".join(issue["labels"]),
                    "author": issue["author"],
                    "comments_count": issue["comments_count"],
                    "url": issue["url"],
                }
            )

        # Save to CSV
        df = pd.DataFrame(csv_data)
        csv_filename = f"issues_by_cluster_{algorithm}.csv"
        csv_filepath = os.path.join(self.output_dir, csv_filename)
        df.to_csv(csv_filepath, index=False)

        return csv_filename

    def fetch_milestone_issues(
        self,
        repo_owner: str = REPO_OWNER,
        repo_name: str = REPO_NAME,
        milestone_number: int = 18,
    ) -> Set[int]:
        """
        Fetch issues from a specific milestone

        Args:
            repo_owner (str): Owner of the repository
            repo_name (str): Name of the repository
            milestone_number (int): Milestone number to fetch issues from

        Returns:
            Set[int]: Set of issue numbers in the milestone
        """
        # Try to get GITHUB_TOKEN from environment
        github_token = os.environ.get("GITHUB_TOKEN", "")
        headers = {}
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        milestone_issues = set()
        page = 1
        per_page = 100

        while True:
            url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"
            params = {
                "milestone": milestone_number,
                "state": "all",
                "per_page": per_page,
                "page": page,
            }

            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()

                issues = response.json()
                if not issues:
                    break

                for issue in issues:
                    if "pull_request" not in issue:  # Skip PRs
                        milestone_issues.add(issue["number"])

                page += 1

            except Exception as e:
                print(f"Error fetching milestone issues: {e}")
                break

        print(
            f"Fetched {len(milestone_issues)} issues from milestone {milestone_number}"
        )
        return milestone_issues

    def generate_cluster_issues_csv(
        self, results: Dict, algorithm: str = "kmeans"
    ) -> str:
        """Generate a TSV with issue numbers, GitHub links, issue titles, cluster numbers and triage status"""
        if algorithm not in results:
            return None

        issues = results[algorithm]["issues"]

        # Fetch issues from milestone/18 for triage flag
        milestone_issues = self.fetch_milestone_issues()

        # Extract the required information
        cluster_issues = []
        for issue in issues:
            issue_number = issue["number"]
            cluster = issue["cluster"]
            url = issue["url"]
            title = issue["title"]

            # Check if issue is in milestone/18 for triage
            is_triage = issue_number in milestone_issues

            cluster_issues.append(
                {
                    "issue_number": issue_number,
                    "cluster": cluster,
                    "github_url": url,
                    "issue_title": title,
                    "triage": is_triage,
                }
            )

        # Create DataFrame and save to TSV
        df = pd.DataFrame(cluster_issues)
        tsv_filename = "cluster_issues_summary.tsv"
        tsv_filepath = os.path.join(self.output_dir, tsv_filename)
        df.to_csv(tsv_filepath, index=False, sep="\t")

        return tsv_filename

    def generate_complete_report(
        self, results: Dict, issues: List[Dict]
    ) -> Dict[str, str]:
        """Generate all report components"""
        report_files = {}

        print("Generating reports...")

        # Apply cluster merging based on similar themes
        print("🔄 Checking for clusters with similar themes to merge...")
        results = self.merge_similar_theme_clusters(results)

        # Generate text summary
        summary_filename = "cluster_summary.txt"
        summary_text = self.generate_cluster_summary_text(results)
        summary_filepath = os.path.join(self.output_dir, summary_filename)
        with open(summary_filepath, "w", encoding="utf-8") as f:
            f.write(summary_text)
        report_files["summary"] = summary_filename

        # Generate HTML report
        html_filename = self.generate_html_report(results, issues)
        report_files["html"] = html_filename

        # Generate detailed CSV with all issue information
        csv_filename = self.save_issues_csv(results)
        if csv_filename:
            report_files["csv"] = csv_filename

        # Generate simplified CSV with just issue numbers, GitHub links, and cluster numbers
        algorithm = "kmeans_merged" if "kmeans_merged" in results else "kmeans"
        cluster_csv_filename = self.generate_cluster_issues_csv(results, algorithm)
        if cluster_csv_filename:
            report_files["cluster_csv"] = cluster_csv_filename

        # Save results JSON
        json_filename = "clustering_results.json"
        json_filepath = os.path.join(self.output_dir, json_filename)

        # Convert numpy arrays and data types to JSON-serializable formats
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64, np.integer)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        json_results = {}
        for key, value in results.items():
            if key == "visualization":
                json_results[key] = {
                    "pca_coordinates": value["pca_coordinates"].tolist(),
                    "tsne_coordinates": value["tsne_coordinates"].tolist(),
                }
            elif isinstance(value, dict) and "labels" in value:
                # Convert cluster_terms dictionary keys from numpy int32 to strings
                cluster_terms_converted = {}
                for cluster_id, terms in value["cluster_terms"].items():
                    cluster_terms_converted[str(cluster_id)] = convert_numpy_types(
                        terms
                    )

                json_results[key] = {
                    "labels": value["labels"].tolist(),
                    "metadata": convert_numpy_types(value["metadata"]),
                    "cluster_terms": cluster_terms_converted,
                    "issues": convert_numpy_types(value["issues"]),
                }
            else:
                json_results[key] = convert_numpy_types(value)

        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        report_files["json"] = json_filename

        print(f"Reports generated in '{self.output_dir}' directory:")
        for report_type, filename in report_files.items():
            print(f"  {report_type.upper()}: {filename}")

        return report_files


def main():
    """Test report generation"""
    pass


if __name__ == "__main__":
    main()
