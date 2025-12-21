#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import csv
import itertools
from pathlib import Path


def generate_dataset(output_file: str = "matmul_dataset.csv", start: int = 256, end: int = 1024, step: int = 32):
    """
    Generate a CSV dataset with all combinations of m, n, k values.

    Args:
        output_file: Path to the output CSV file
        start: Starting value for dimensions (inclusive)
        end: Ending value for dimensions (inclusive)
        step: Step size between values
    """
    # Generate the range of values
    values = list(range(start, end + 1, step))

    # Create all combinations of m, n, k
    combinations = list(itertools.product(values, repeat=3))

    # Get the directory of this script
    script_dir = Path(__file__).parent
    output_path = script_dir / output_file

    # Write to CSV
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["m", "n", "k"])  # Header
        writer.writerows(combinations)

    print(f"Generated {len(combinations)} combinations")
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    generate_dataset()
