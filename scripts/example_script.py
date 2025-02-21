# pylint: disable=E1102
# pylint: disable=invalid-name
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from projorg import (
    datadir,
    logsdir,
    make_experiment_name,
    plotsdir,
    process_sequence_arguments,
    query_arguments,
)

CONFIG_FILE = "example_config_file.json"


def gaussian_random_fields(num_fields=1, input_size=[128, 128], alpha=4.0):
    kx, ky = np.meshgrid(
        np.fft.fftfreq(input_size[1]), np.fft.fftfreq(input_size[0])
    )
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1  # Avoid division by zero
    power_spectrum = k ** (-alpha)
    power_spectrum[0, 0] = 0  # Remove DC component

    # Generate multiple fields at once
    noise = np.fft.fft2(np.random.normal(size=[num_fields] + input_size))
    fields = np.fft.ifft2(noise * np.sqrt(power_spectrum)).real

    return fields


def do_compute(args: argparse.ArgumentParser) -> None:
    """Runs a fictitious computation to showcase the project structure.

    Args:
        args (argparse.ArgumentParser): The command line arguments.
    """

    # Generate a random field.
    field = gaussian_random_fields(num_fields=1, input_size=args.input_size)

    # Save the field to a file.
    np.save(
        os.path.join(datadir(args.experiment), "random_field.npy"),
        field,
    )

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(
        field[0],
        aspect=1,
        cmap="YlGnBu",
        interpolation="lanczos",
    )
    plt.title("Gaussian random field")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(
        os.path.join(plotsdir(args.experiment), "random_field.png"),
        format="png",
        bbox_inches="tight",
        dpi=400,
        pad_inches=0.02,
    )
    plt.close(fig)

    # Write the experiment arguments and path to data and plots to a log
    # text file.
    with open(
        os.path.join(logsdir(args.experiment), "experiment_log.txt"),
        "w",
    ) as f:
        f.write(f"Experiment arguments: {args}\n")
        f.write(f"Data path: {datadir(args.experiment, mkdir=False)}\n")
        f.write(f"Plots path: {plotsdir(args.experiment, mkdir=False)}\n")

    # Print path to path to logs, data and plots.
    print(f"Logs path: {logsdir(args.experiment, mkdir=False)}")
    print(f"Data path: {datadir(args.experiment, mkdir=False)}")
    print(f"Plots path: {plotsdir(args.experiment, mkdir=False)}")


if "__main__" == __name__:
    # Read input arguments from a json file and make an experiment name.
    args = query_arguments(CONFIG_FILE)[0]
    args.experiment = make_experiment_name(args)
    args = process_sequence_arguments(args, [("input_size", int)])

    # Run the computation.
    do_compute(args)
