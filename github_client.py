"""
GitHub API client for fetching issues data
"""

import json
import os
import time
from typing import Dict, List, Optional

import requests
from tqdm import tqdm
from config import *


class GitHubClient:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "GitHub-Issues-Analyzer",
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"

        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _make_request(self, url: str, params: Dict = None) -> Dict:
        """Make a request to GitHub API with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(
                    url, params=params, timeout=GITHUB_API_TIMEOUT
                )

                # Handle rate limiting
                if (
                    response.status_code == 403
                    and "rate limit" in response.text.lower()
                ):
                    reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                    sleep_time = max(reset_time - int(time.time()), 60)
                    print(f"Rate limit hit. Sleeping for {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    raise e
                print(f"Request failed (attempt {attempt + 1}): {e}")
                time.sleep(2**attempt)  # Exponential backoff

        return {}

    def get_repository_info(self) -> Dict:
        """Get basic repository information"""
        url = f"{GITHUB_API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}"
        return self._make_request(url)

    def get_all_issues(self) -> List[Dict]:
        """Fetch all issues from the repository using GitHub Search API to avoid PRs"""
        issues = []
        seen_issue_ids = set()  # Track issue IDs to avoid duplicates
        page = 1

        print(f"Fetching issues from {REPO_OWNER}/{REPO_NAME}...")
        print(
            f"Configuration: ISSUES_PER_PAGE={ISSUES_PER_PAGE}, INCLUDE_CLOSED_ISSUES={INCLUDE_CLOSED_ISSUES}"
        )

        # Use GitHub Search API to specifically search for issues (not PRs)
        state_filter = "" if INCLUDE_CLOSED_ISSUES else "state:open"
        search_query = f"repo:{REPO_OWNER}/{REPO_NAME} is:issue {state_filter}".strip()

        print(f"Search query: {search_query}")

        while True:
            params = {
                "q": search_query,
                "page": page,
                "per_page": min(ISSUES_PER_PAGE, 100),  # Search API max is 100
                "sort": "created",
                "order": "desc",
            }

            url = f"{GITHUB_API_BASE}/search/issues"
            print(f"Fetching page {page}...")

            try:
                search_result = self._make_request(url, params)

                if not search_result or "items" not in search_result:
                    print(f"No more items returned on page {page}")
                    break

                page_issues = search_result["items"]
                total_count = search_result.get("total_count", 0)

                if not page_issues:
                    print(f"No items in page {page}")
                    break

                # Filter duplicates and track unique issues
                unique_issues = []
                duplicates_found = 0

                for issue in page_issues:
                    issue_id = issue.get("id")
                    if issue_id not in seen_issue_ids:
                        seen_issue_ids.add(issue_id)
                        unique_issues.append(issue)
                    else:
                        duplicates_found += 1

                if duplicates_found > 0:
                    print(f"   Found {duplicates_found} duplicate issues on this page (skipped)")

                print(
                    f"Page {page}: Got {len(page_issues)} items, {len(unique_issues)} unique issues (total available: {total_count})"
                )

                issues.extend(unique_issues)
                print(f"Total unique issues fetched so far: {len(issues)}")

                # Break if we got less than a full page or no new unique issues
                if len(page_issues) < min(ISSUES_PER_PAGE, 100) or len(unique_issues) == 0:
                    if len(unique_issues) == 0:
                        print("No new unique issues found, stopping pagination")
                    else:
                        print(
                            f"Got {len(page_issues)} items (less than requested), this was the last page."
                        )
                    break

                page += 1
                time.sleep(0.1)  # Be nice to the API

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 422:
                    print(
                        f"Search API returned 422 (Unprocessable Entity) on page {page}"
                    )
                    print(
                        "This usually means we've reached the end of available results"
                    )
                    break
                else:
                    raise e

        print("=" * 60)
        print(f"FETCH SUMMARY:")
        print(f"Total unique issues fetched: {len(issues)}")
        print(f"Pages processed: {page}")
        print(f"Issue IDs tracked: {len(seen_issue_ids)}")
        print("=" * 60)

        return issues

    def process_issues_data(self, issues: List[Dict]) -> List[Dict]:
        """Process and clean issues data for analysis"""
        processed_issues = []

        for issue in tqdm(issues, desc="Processing issues"):
            # Skip issues with very short bodies
            body = issue.get("body", "") or ""
            if len(body.strip()) < MIN_ISSUE_BODY_LENGTH:
                continue

            processed_issue = {
                "id": issue["id"],
                "number": issue["number"],
                "title": issue["title"],
                "body": body,
                "state": issue["state"],
                "created_at": issue["created_at"],
                "updated_at": issue["updated_at"],
                "labels": [label["name"] for label in issue.get("labels", [])],
                "assignees": [
                    assignee["login"] for assignee in issue.get("assignees", [])
                ],
                "milestone": (
                    issue.get("milestone", {}).get("title", None)
                    if issue.get("milestone")
                    else None
                ),
                "author": issue.get("user", {}).get("login", "unknown"),
                "comments_count": issue.get("comments", 0),
                "url": issue["html_url"],
            }

            # Combine title and body for text analysis
            processed_issue["full_text"] = (
                f"{processed_issue['title']} {processed_issue['body']}"
            )

            processed_issues.append(processed_issue)

        return processed_issues

    def save_issues_data(self, issues: List[Dict], filename: str = "issues_data.json"):
        """Save issues data to JSON file"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(issues, f, indent=2, ensure_ascii=False)
        print(f"Issues data saved to {filename}")

    def load_issues_data(self, filename: str = "issues_data.json") -> List[Dict]:
        """Load issues data from JSON file"""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"File {filename} not found. Fetching fresh data...")
            return []


def main():
    """Test the GitHub client"""
    client = GitHubClient()

    # Get repository info
    repo_info = client.get_repository_info()
    print(f"Repository: {repo_info.get('full_name', 'N/A')}")
    print(f"Description: {repo_info.get('description', 'N/A')}")
    print(f"Stars: {repo_info.get('stargazers_count', 0)}")
    print(f"Open Issues: {repo_info.get('open_issues_count', 0)}")

    # Fetch all issues
    issues = client.get_all_issues()
    processed_issues = client.process_issues_data(issues)
    client.save_issues_data(processed_issues)

    print(f"Processed {len(processed_issues)} issues for analysis")


if __name__ == "__main__":
    main()
