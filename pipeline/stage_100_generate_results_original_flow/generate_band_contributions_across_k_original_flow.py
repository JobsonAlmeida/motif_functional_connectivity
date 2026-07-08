import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

subjects = [f"sub-{i:02d}" for i in range(1, 11)]

max_type_name = "max_1_2_3"   # lags = [1, 2, 3]

def plot_band_contributions_across_k_original_flow(
        k_values=[20, 30, 40, 50, 60, 70, 80, 90, 100],
        bands=[
            (12, 15), (15, 18), (18, 21), (21, 24),
            (24, 27), (27, 30), (30, 33), (33, 36),
            (36, 39), (39, 42), (42, 45),
        ]):

    base_path = (
        project_root
        / "processed_data"
        / "stage_90_maximum_matrix_based_SVM"
    )

    output_path = (
        project_root
        / "results"
        / current_file.parents[0].name
        / "band_contributions_across_k_features"
    )

    os.makedirs(output_path, exist_ok=True)

    k_values = list(k_values)

    n_bands = len(bands)
    n_channels = 128
    n_measures = 2

    band_contribution_results = {}

    for k in k_values:

        band_counts = np.zeros(n_bands, dtype=int)

        for subject in subjects:

            file_path = base_path / f"{subject}_ranked_features_svm_results.pkl"

            if not file_path.exists():
                print(f"Arquivo não encontrado: {file_path}")
                continue

            with open(file_path, "rb") as f:
                results = pickle.load(f)

            ranked_feature_indices = results[subject][max_type_name]["ranked_feature_indices"]

            topk_indices = ranked_feature_indices[:k]

            for feature_idx in topk_indices:

                band_idx, channel_idx, measure_idx = np.unravel_index(
                    feature_idx,
                    (n_bands, n_channels, n_measures)
                )

                band_counts[band_idx] += 1

        relative_band_frequency = band_counts / band_counts.sum()

        band_contribution_results[k] = relative_band_frequency

    y = np.array([
        band_contribution_results[k]
        for k in k_values
    ]).T

    band_labels = [f"[{low}, {high}]" for low, high in bands]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.stackplot(k_values, y, labels=band_labels)

    ax.set_title(
        "Relative contribution of frequency bands across K\n"
        "Delayed motif synchronization (lags = [1, 2, 3])",
        fontsize=14
    )

    ax.set_xlabel("Top-K selected features")
    ax.set_ylabel("Relative band contribution")
    ax.set_ylim(0, 1)

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    save_path = output_path / "band_contributions.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figura salva em: {save_path}")


if __name__ == "__main__":
    plot_band_contributions_across_k_original_flow()