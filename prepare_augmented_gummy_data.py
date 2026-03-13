"""
Prepare train/validation CSV files for the Tetrasense gummy experiments.

What this script does
---------------------
1. Reads the original CSV file.
2. Splits ORIGINAL rows into train and validation sets.
3. Augments ONLY the training rows.
4. Writes:
   - an augmented training CSV
   - a real-only validation CSV

Why this is safer
-----------------
This avoids leakage caused by augmenting rows first and then letting the site
split similar samples across train/validation.

CSV format expected
-------------------
- Row 0: voltage values, then an empty separator column, then label names.
- Rows 1..N: waveform values, then an empty separator column, then label values.

Example tail of a row:
    ..., <last waveform value>, "", THC, CBD, CBDa, THCa

No CLI is used. Edit the constants below and run:
    python prepare_augmented_gummy_data.py
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd



# SCRIPT CONFIGURATION
INPUT_CSV = Path("gummies_m2_v2.csv")
OUTPUT_TRAIN_CSV = Path("gummies_m2_v2_train_augmented.csv")
OUTPUT_VALIDATION_CSV = Path("gummies_m2_v2_validation_real.csv")

RANDOM_SEED = 42
VALIDATION_FRACTION = 0.20

# If True, rows with larger THC/CBD concentrations are more likely to be chosen
# as augmentation sources. This directly targets the underestimation issue in
# the high-concentration region.
USE_CONCENTRATION_AWARE_SAMPLING = True

# Main labels to target in sampling. These are the relevant compounds in the
# gummy experiments according to your notes.
SAMPLING_LABELS = ["THC", "CBD"]

# Number of synthetic rows to create, relative to the size of the REAL training set.
# 0.5 means "create synthetic rows equal to 50% of the original training rows".
AUGMENTATION_MULTIPLIER = 1.0

# Choose which augmentations are available for sampling.
# Recommended starting point:
# ENABLED_AUGMENTATIONS = ["gaussian_noise", "peak_shift", "smooth_noise"]
ENABLED_AUGMENTATIONS = [
    "gaussian_noise",
    "peak_shift",
    "smooth_noise",

]

# Probability of selecting each augmentation for a synthetic row.
# They will be normalized automatically over the ENABLED_AUGMENTATIONS list.
AUGMENTATION_WEIGHTS = {
    "gaussian_noise": 0.60,
    "peak_shift": 0.30,
    "smooth_noise": 0.10,
}
# Apply at most this many weak augmentations to each synthetic row.
# Keep this at 1 unless you have a very good reason to raise it.
MAX_AUGMENTATIONS_PER_SAMPLE = 1


# ----------------------------- Augmentation strength -------------------------
# All strengths are intentionally mild and amplitude-preserving.

# Gaussian noise scaled to each waveform's own std.
GAUSSIAN_STD_FRACTION = 0.03  # 3% of waveform std

# Random noise: uniform perturbation scaled to waveform std.
RANDOM_NOISE_STD_FRACTION = 0.02  # 2% of waveform std

# Smooth noise: low-frequency noise created by smoothing white noise.
SMOOTH_NOISE_STD_FRACTION = 0.02
SMOOTH_NOISE_KERNEL_SIZE = 11  # odd number recommended

# Peak/sample shift: tiny shift along the sample axis using interpolation.
# 0.02 = at most 2% of the waveform length.
MAX_PEAK_SHIFT_FRACTION = 0.02

# Mild baseline drift: low-order smooth additive trend.
BASELINE_DRIFT_STD_FRACTION = 0.01


LABEL_COLUMNS = ["THC", "CBD", "CBDa", "THCa"]
SEPARATOR_COLUMN_NAME = "__sep__"


@dataclass
class ParsedDataset:
    voltage_row: np.ndarray
    waveforms: pd.DataFrame
    labels: pd.DataFrame


def read_dataset(path: Path) -> ParsedDataset:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    header = rows[0]
    if len(header) < 6:
        raise ValueError("CSV does not look like the expected waveform+labels format.")

    if header[-4:] != LABEL_COLUMNS:
        raise ValueError(
            f"Expected last 4 header cells to be {LABEL_COLUMNS}, got {header[-4:]!r}"
        )

    waveform_width = len(header) - 5  # waveform cols + separator + 4 labels
    waveform_values = header[:waveform_width]
    separator_header = header[waveform_width]
    if separator_header != "":
        # Not fatal, but the current files use an empty separator.
        print(f"Warning: expected empty separator header, got {separator_header!r}")

    voltage_row = np.array([float(x) for x in waveform_values], dtype=float)

    waveform_rows: List[List[float]] = []
    label_rows: List[List[float]] = []

    for i, row in enumerate(rows[1:], start=1):
        if len(row) != len(header):
            raise ValueError(f"Row {i} has {len(row)} columns, expected {len(header)}")

        waveform_part = row[:waveform_width]
        label_part = row[-4:]

        waveform_rows.append([float(x) for x in waveform_part])
        label_rows.append([float(x) for x in label_part])

    waveforms = pd.DataFrame(waveform_rows)
    labels = pd.DataFrame(label_rows, columns=LABEL_COLUMNS)
    return ParsedDataset(voltage_row=voltage_row, waveforms=waveforms, labels=labels)


def write_dataset(
    path: Path,
    voltage_row: np.ndarray,
    waveforms: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    header = [f"{x:.9g}" for x in voltage_row] + [""] + LABEL_COLUMNS

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i in range(len(waveforms)):
            waveform = waveforms.iloc[i].to_numpy(dtype=float)
            label = labels.iloc[i][LABEL_COLUMNS].to_numpy(dtype=float)
            row = [f"{x:.9g}" for x in waveform] + [""] + [f"{x:.9g}" for x in label]
            writer.writerow(row)


def split_real_data(
    waveforms: pd.DataFrame,
    labels: pd.DataFrame,
    validation_fraction: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(waveforms)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_val = max(1, int(round(n * validation_fraction)))
    val_idx = np.sort(indices[:n_val])
    train_idx = np.sort(indices[n_val:])

    train_waveforms = waveforms.iloc[train_idx].reset_index(drop=True)
    train_labels = labels.iloc[train_idx].reset_index(drop=True)
    val_waveforms = waveforms.iloc[val_idx].reset_index(drop=True)
    val_labels = labels.iloc[val_idx].reset_index(drop=True)

    return train_waveforms, train_labels, val_waveforms, val_labels


def moving_average(arr: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size, dtype=float) / kernel_size
    padded = np.pad(arr, (kernel_size // 2, kernel_size // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def add_gaussian_noise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = max(np.std(x) * GAUSSIAN_STD_FRACTION, 1e-12)
    return x + rng.normal(0.0, sigma, size=x.shape)


def add_random_noise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = max(np.std(x) * RANDOM_NOISE_STD_FRACTION, 1e-12)
    width = np.sqrt(3.0) * sigma
    return x + rng.uniform(-width, width, size=x.shape)


def add_smooth_noise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = max(np.std(x) * SMOOTH_NOISE_STD_FRACTION, 1e-12)
    white = rng.normal(0.0, sigma, size=x.shape)
    smooth = moving_average(white, SMOOTH_NOISE_KERNEL_SIZE)
    return x + smooth


def add_baseline_drift(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    t = np.linspace(-1.0, 1.0, n)
    a = rng.normal(0.0, np.std(x) * BASELINE_DRIFT_STD_FRACTION)
    b = rng.normal(0.0, np.std(x) * BASELINE_DRIFT_STD_FRACTION)
    drift = a * t + b * (t ** 2 - np.mean(t ** 2))
    return x + drift


def apply_peak_shift(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    max_shift = max(1, int(round(n * MAX_PEAK_SHIFT_FRACTION)))
    shift = rng.integers(-max_shift, max_shift + 1)
    if shift == 0:
        return x.copy()

    original_idx = np.arange(n, dtype=float)
    shifted_idx = original_idx - shift
    return np.interp(shifted_idx, original_idx, x, left=x[0], right=x[-1])


AUGMENTATION_FUNCTIONS = {
    "gaussian_noise": add_gaussian_noise,
    "random_noise": add_random_noise,
    "smooth_noise": add_smooth_noise,
    "baseline_drift": add_baseline_drift,
    "peak_shift": apply_peak_shift,
}


def normalized_augmentation_weights() -> Tuple[List[str], np.ndarray]:
    chosen = [name for name in ENABLED_AUGMENTATIONS if name in AUGMENTATION_FUNCTIONS]
    if not chosen:
        raise ValueError("ENABLED_AUGMENTATIONS is empty or invalid.")
    weights = np.array([AUGMENTATION_WEIGHTS.get(name, 0.0) for name in chosen], dtype=float)
    if np.all(weights <= 0):
        weights = np.ones(len(chosen), dtype=float)
    weights = weights / weights.sum()
    return chosen, weights


def build_sampling_probabilities(labels: pd.DataFrame) -> np.ndarray:
    n = len(labels)
    if not USE_CONCENTRATION_AWARE_SAMPLING:
        return np.full(n, 1.0 / n)

    score = np.zeros(n, dtype=float)

    # Use quantile ranks so the highest THC/CBD values get higher sampling weight.
    for col in SAMPLING_LABELS:
        if col not in labels.columns:
            continue
        rank = labels[col].rank(method="average", pct=True).to_numpy(dtype=float)
        score += rank

    if np.all(score == 0):
        return np.full(n, 1.0 / n)

    # Make the top end more likely without being too extreme.
    score = 0.25 + score ** 2
    return score / score.sum()


def augment_training_set(
    train_waveforms: pd.DataFrame,
    train_labels: pd.DataFrame,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num_real = len(train_waveforms)
    num_synthetic = int(round(num_real * AUGMENTATION_MULTIPLIER))
    if num_synthetic <= 0:
        return train_waveforms.copy(), train_labels.copy()

    source_probs = build_sampling_probabilities(train_labels)
    aug_names, aug_probs = normalized_augmentation_weights()

    synthetic_waveforms: List[np.ndarray] = []
    synthetic_labels: List[np.ndarray] = []

    source_indices = rng.choice(np.arange(num_real), size=num_synthetic, replace=True, p=source_probs)

    for idx in source_indices:
        x = train_waveforms.iloc[idx].to_numpy(dtype=float).copy()
        y = train_labels.iloc[idx].to_numpy(dtype=float).copy()

        k = max(1, MAX_AUGMENTATIONS_PER_SAMPLE)
        chosen_count = 1 if k == 1 else rng.integers(1, k + 1)
        chosen_augs = rng.choice(aug_names, size=chosen_count, replace=False, p=aug_probs)

        for aug_name in chosen_augs:
            x = AUGMENTATION_FUNCTIONS[aug_name](x, rng)

        synthetic_waveforms.append(x)
        synthetic_labels.append(y)

    augmented_waveforms = pd.concat(
        [train_waveforms, pd.DataFrame(synthetic_waveforms, columns=train_waveforms.columns)],
        ignore_index=True,
    )
    augmented_labels = pd.concat(
        [train_labels, pd.DataFrame(synthetic_labels, columns=train_labels.columns)],
        ignore_index=True,
    )

    return augmented_waveforms, augmented_labels


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    dataset = read_dataset(INPUT_CSV)

    train_wf, train_lb, val_wf, val_lb = split_real_data(
        dataset.waveforms,
        dataset.labels,
        VALIDATION_FRACTION,
        rng,
    )

    aug_train_wf, aug_train_lb = augment_training_set(train_wf, train_lb, rng)

    write_dataset(OUTPUT_TRAIN_CSV, dataset.voltage_row, aug_train_wf, aug_train_lb)
    write_dataset(OUTPUT_VALIDATION_CSV, dataset.voltage_row, val_wf, val_lb)

    print("Done.")
    print(f"Input rows (real only):         {len(dataset.waveforms)}")
    print(f"Training rows (real only):      {len(train_wf)}")
    print(f"Validation rows (real only):    {len(val_wf)}")
    print(f"Training rows (after augment):  {len(aug_train_wf)}")
    print(f"Wrote: {OUTPUT_TRAIN_CSV}")
    print(f"Wrote: {OUTPUT_VALIDATION_CSV}")
    print(f"Enabled augmentations: {ENABLED_AUGMENTATIONS}")
    print(f"Concentration-aware sampling: {USE_CONCENTRATION_AWARE_SAMPLING}")


if __name__ == "__main__":
    main()
