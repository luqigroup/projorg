"""Updated setup_experiment.py with integrated logging."""

import os
from typing import Optional

from .config import make_experiment_name, process_sequence_arguments
from .experiment_logger import log_experiment_info
from .hyperparam_utils import query_arguments
from .project_path import configsdir


def setup_environment(
    config_file: str,
    ignore_arg_list: list = [],
    sequence_args_and_types: list = [],
    additional_info: Optional[dict] = None,
    log_info: bool = True,
):
    """Setup experiment environment with comprehensive logging.

    Args:
        config_file: Name of the configuration file in the configs directory.
        ignore_arg_list: List of argument names to ignore when creating experiment name.
        sequence_args_and_types: List of tuples (arg_name, type) for sequence arguments.
        additional_info: Optional dictionary of additional information to log.
        log_info: Whether to log experiment information. Default is True.

    Returns:
        args: Namespace object containing parsed arguments and
        experiment name.
    """
    # Parse arguments
    args = query_arguments(config_file)[0]
    args.experiment = make_experiment_name(
        args, ignore_arg_list=ignore_arg_list
    )
    args = process_sequence_arguments(args, sequence_args_and_types)

    # Log experiment information
    if log_info:
        config_file_path = os.path.join(configsdir(), config_file)
        log_experiment_info(
            args,
            config_file=config_file_path,
            additional_info=additional_info,
        )

    return args
