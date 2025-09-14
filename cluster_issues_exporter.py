"""
Cluster issues exporter - Generate a CSV file with issue numbers, GitHub links, and cluster numbers
"""

import os
import json
import requests
import pandas as pd
from typing import Dict, List, Set


def fetch_milestone_issues(repo_owner: str, repo_name: str, milestone_number: int = 18) -> Set[int]:
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
            "page": page
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

        except requests.RequestException as e:
            print(f"Error fetching milestone issues: {e}")
            break

    print(f"Fetched {len(milestone_issues)} issues from milestone {milestone_number}")
    return milestone_issues


def generate_cluster_issues_csv(results_dir: str) -> str:
    """
    Generate a CSV file with issue numbers, GitHub links, and cluster numbers.

    Args:
        results_dir (str): Directory containing the clustering results

    Returns:
        str: Path to the generated CSV file
    """
    # Identify the algorithm used for the report
    summary_path = os.path.join(results_dir, "cluster_summary.txt")
    with open(summary_path, 'r') as f:
        lines = f.readlines()
        # Extract algorithm from first line
        algorithm_line = lines[0]
        algorithm = algorithm_line.strip().split(' - ')[1].lower()

        # Extract repository information
        repo_line = [line for line in lines if line.startswith('Repository:')][0]
        repo = repo_line.strip().split(': ')[1]
        repo_owner, repo_name = repo.split('/')

    # Try to load issues from CSV first (if it exists)
    issues_csv_path = os.path.join(results_dir, f"issues_by_cluster_{algorithm.split('_')[0]}.csv")
    if os.path.exists(issues_csv_path):
        df = pd.read_csv(issues_csv_path)
        issues_data = df.to_dict('records')
    else:
        # If CSV doesn't exist, try to load from JSON
        json_path = os.path.join(results_dir, "clustering_results.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Could not find clustering results in {results_dir}")

        with open(json_path, 'r') as f:
            results = json.load(f)

        issues_data = results[algorithm]["issues"]

    # Fetch issues from milestone/18
    milestone_issues = fetch_milestone_issues(repo_owner, repo_name)

    # Extract the required information
    cluster_issues = []
    for issue in issues_data:
        issue_number = issue.get("issue_number", issue.get("number"))
        cluster = issue.get("cluster")
        title = issue.get("title", "")

        # Construct GitHub URL if not present
        if "url" not in issue:
            url = f"https://github.com/{repo}/issues/{issue_number}"
        else:
            url = issue.get("url")

        # Check if issue is in milestone/18
        is_triage = issue_number in milestone_issues

        cluster_issues.append({
            "issue_number": issue_number,
            "cluster": cluster,
            "github_url": url,
            "issue_title": title,
            "triage": is_triage
        })

    # Create DataFrame and save to TSV
    df = pd.DataFrame(cluster_issues)
    output_path = os.path.join(results_dir, "cluster_issues_summary.tsv")
    df.to_csv(output_path, index=False, sep='\t')

    print(f"TSV file generated at: {output_path}")
    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Use the most recent results directory if none provided
        base_dir = os.path.dirname(os.path.abspath(__file__))
        report_dirs = [d for d in os.listdir(base_dir) if d.startswith('reports_')]
        if not report_dirs:
            print("No report directories found.")
            sys.exit(1)

        results_dir = os.path.join(base_dir, sorted(report_dirs)[-1])
        print(f"Using most recent results directory: {results_dir}")

    generate_cluster_issues_csv(results_dir)
