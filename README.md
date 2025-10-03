<h2 align="center">projorg: Python project organization and management</h2>

A lightweight package for managing Python project structure, experiment tracking, and reproducibility.

## Features

- Consistent project directory structure (data, plots, checkpoints, logs)
- Automatic experiment naming based on input arguments
- Configuration management with JSON files and command-line overrides
- Comprehensive experiment logging (git hashes, timestamps, diffs)
- Cloud backup integration via rclone
- Hyperparameter grid search utilities

## Requirements

Your project must be in a git repository for the logging and path resolution features to work properly. The package uses git to track code versions, generate experiment logs, and locate the project root directory.

## Installation

```bash
pip install projorg
```

### Development installation

```bash
git clone https://github.com/alisiahkoohi/projorg
cd projorg/
pip install -e .
```

## Quick Start

Create a configuration file in `configs/my_config.json` with your
experiment parameters:

```json
{
    "experiment_name": "my_experiment",
    "learning_rate": 0.001,
    "batch_size": 32,
    "seed": 42
}
```

Use in your script:

```python
from projorg import setup_environment, datadir, plotsdir

# Setup with automatic logging
args = setup_environment(
    "my_config.json",
    ignore_arg_list=["experiment_name"],
    sequence_args_and_types=[("batch_size", int)]
)

# Access organized directories
data_path = datadir(args.experiment)
plots_path = plotsdir(args.experiment)
```

Override config values from command line:

```bash
python scripts/my_script.py --learning_rate 0.01 --batch_size 64
```

## Example

See `scripts/example_script.py` for a complete working example:

```bash
python scripts/example_script.py
```

## Core Functions

- `setup_environment()`: Initialize experiment with automatic logging of git commit hash, timestamp, configuration, and code diffs
- `datadir()`, `plotsdir()`, `checkpointsdir()`, `logsdir()`: Get organized paths
- `query_arguments()`: Generate hyperparameter combinations for grid search
- `upload_to_cloud()`: Backup experiments to cloud storage

## Logging

The `setup_environment()` function automatically creates detailed logs in the `logs/` directory for each experiment, including:

- Current git commit hash and branch
- Execution timestamp
- Full configuration file used
- Git diff showing uncommitted changes
- Any additional metadata you provide

This ensures complete reproducibility by capturing the exact code state for every experiment run.

## Questions

Contact alisk@ucf.edu

## Author

Ali Siahkoohi