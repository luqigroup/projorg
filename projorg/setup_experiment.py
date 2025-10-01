from .config import make_experiment_name, process_sequence_arguments
from .hyperparam_utils import query_arguments


def setup_environment(
    config_file: str,
    ignore_arg_list: list = [],
    sequence_args_and_types: list = [],
):
    # Parse arguments
    args = query_arguments(config_file)[0]
    args.experiment = make_experiment_name(
        args, ignore_arg_list=ignore_arg_list
    )
    args = process_sequence_arguments(args, sequence_args_and_types)

    return args
