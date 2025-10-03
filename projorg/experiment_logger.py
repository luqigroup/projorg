"""Experiment logging utilities for tracking reproducibility
information."""

import json
import os
from datetime import datetime
from typing import Optional

import git

from .project_path import logsdir


def get_git_info(repo_path: Optional[str] = None):
    """Get git repository information including hash, branch, and diff.

    Args:
        repo_path: Path to git repository. If None, uses current
        directory.

    Returns:
        dict: Dictionary containing git information including:
            - hash: Current commit hash
            - short_hash: Short version of commit hash
            - branch: Current branch name
            - is_dirty: Whether there are uncommitted changes
            - diff: Git diff if there are uncommitted changes
            - remote_url: Remote repository URL
            - commit_message: Latest commit message
            - commit_date: Latest commit date
    """
    try:
        if repo_path is None:
            repo = git.Repo(os.getcwd(), search_parent_directories=True)
        else:
            repo = git.Repo(repo_path)

        git_info = {
            "hash": repo.head.commit.hexsha,
            "short_hash": repo.head.commit.hexsha[:7],
            "branch": repo.active_branch.name,
            "is_dirty": repo.is_dirty(),
            "commit_message": repo.head.commit.message.strip(),
            "commit_date": repo.head.commit.committed_datetime.isoformat(),
        }

        # Get remote URL if available
        try:
            git_info["remote_url"] = repo.remotes.origin.url
        except:
            git_info["remote_url"] = "No remote configured"

        # Get diff if there are uncommitted changes
        if repo.is_dirty():
            git_info["diff"] = repo.git.diff()
            git_info["untracked_files"] = repo.untracked_files
        else:
            git_info["diff"] = None
            git_info["untracked_files"] = []

        return git_info

    except Exception as e:
        return {
            "error": f"Could not retrieve git information: {str(e)}",
            "hash": "unknown",
            "is_dirty": None,
        }


def log_experiment_info(
    args,
    config_file: Optional[str] = None,
    additional_info: Optional[dict] = None,
):
    """Log comprehensive experiment information to the logs directory.

    This creates several files in the experiment's log directory:
    - experiment_info.json: Complete experiment metadata in JSON format
    - experiment_config.json: Copy of the input configuration
    - git_diff.patch: Git diff if there are uncommitted changes
    - experiment_info.txt: Human-readable summary

    Args:
        args: Namespace object containing experiment arguments.
        config_file: Optional path to the original config file.
        additional_info: Optional dictionary of additional information to log.
    """
    # Create logs directory
    log_path = logsdir(args.experiment)

    # Get timestamp
    timestamp = datetime.now().isoformat()

    # Get git information for code
    code_git_info = get_git_info()

    # Compile all experiment information
    experiment_info = {
        "timestamp": timestamp,
        "experiment_name": args.experiment,
        "arguments": vars(args),
        "code_git": code_git_info,
    }

    # Add additional info if provided
    if additional_info:
        experiment_info["additional_info"] = additional_info

    # Save complete info as JSON
    with open(os.path.join(log_path, "experiment_info.json"), "w") as f:
        json.dump(experiment_info, f, indent=2, default=str)

    # Save config file if provided
    if config_file and os.path.exists(config_file):
        with open(config_file, "r") as f_in:
            config_data = json.load(f_in)
        with open(
            os.path.join(log_path, "experiment_config.json"), "w"
        ) as f_out:
            json.dump(config_data, f_out, indent=2)

    # Save git diff if dirty
    if code_git_info.get("is_dirty") and code_git_info.get("diff"):
        with open(os.path.join(log_path, "git_diff.patch"), "w") as f:
            f.write(code_git_info["diff"])

    # Create human-readable summary
    summary = _create_summary(experiment_info)
    with open(os.path.join(log_path, "experiment_info.txt"), "w") as f:
        f.write(summary)

    print(f"\n{'=' * 60}")
    print("Experiment information logged to:")
    print(f"  {log_path}")
    print(f"{'=' * 60}\n")

    # Warn if there are uncommitted changes
    if code_git_info.get("is_dirty"):
        print("WARNING: There are uncommitted changes in your code!")
        print(
            f"Git diff saved to: {os.path.join(log_path, 'git_diff.patch')}\n"
        )


def _create_summary(experiment_info: dict) -> str:
    """Create a human-readable summary of experiment information.

    Args:
        experiment_info: Dictionary containing experiment metadata.

    Returns:
        str: Formatted summary text.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT INFORMATION")
    lines.append("=" * 70)
    lines.append("")

    # Timestamp and experiment name
    lines.append(f"Timestamp: {experiment_info['timestamp']}")
    lines.append(f"Experiment Name: {experiment_info['experiment_name']}")
    lines.append("")

    # Code git information
    lines.append("-" * 70)
    lines.append("CODE REPOSITORY")
    lines.append("-" * 70)
    code_git = experiment_info.get("code_git", {})
    if "error" in code_git:
        lines.append(f"Error: {code_git['error']}")
    else:
        lines.append(f"Commit Hash: {code_git.get('hash', 'unknown')}")
        lines.append(f"Short Hash: {code_git.get('short_hash', 'unknown')}")
        lines.append(f"Branch: {code_git.get('branch', 'unknown')}")
        lines.append(f"Remote URL: {code_git.get('remote_url', 'unknown')}")
        lines.append(f"Commit Date: {code_git.get('commit_date', 'unknown')}")
        lines.append(
            f"Commit Message: {code_git.get('commit_message', 'unknown')}"
        )
        lines.append(
            f"Has Uncommitted Changes: {'YES' if code_git.get('is_dirty') else 'NO'}"
        )
        if code_git.get("untracked_files"):
            lines.append(
                f"Untracked Files: {', '.join(code_git['untracked_files'])}"
            )
    lines.append("")

    # Data git information
    data_git = experiment_info.get("data_git")
    if data_git:
        lines.append("-" * 70)
        lines.append("DATA REPOSITORY")
        lines.append("-" * 70)
        if "error" in data_git:
            lines.append(f"Error: {data_git['error']}")
        else:
            lines.append(f"Commit Hash: {data_git.get('hash', 'unknown')}")
            lines.append(f"Short Hash: {data_git.get('short_hash', 'unknown')}")
            lines.append(f"Branch: {data_git.get('branch', 'unknown')}")
            lines.append(
                f"Has Uncommitted Changes: {'YES' if data_git.get('is_dirty') else 'NO'}"
            )
        lines.append("")

    # Arguments
    lines.append("-" * 70)
    lines.append("EXPERIMENT ARGUMENTS")
    lines.append("-" * 70)
    for key, value in experiment_info.get("arguments", {}).items():
        lines.append(f"{key}: {value}")
    lines.append("")

    # Additional info
    if experiment_info.get("additional_info"):
        lines.append("-" * 70)
        lines.append("ADDITIONAL INFORMATION")
        lines.append("-" * 70)
        for key, value in experiment_info["additional_info"].items():
            lines.append(f"{key}: {value}")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)
