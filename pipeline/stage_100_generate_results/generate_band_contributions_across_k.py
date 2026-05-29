import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

experiment_name = "experiment_1"

base_path = (
    project_root
    / "results"
    / current_file.parents[0].name
    / "run_stage_90_in_loop"
    / experiment_name
)

output_path = (
    project_root
    / "results"
    / current_file.parents[0].name
    / "generate_frequencies_across_k"
    / experiment_name
)

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]

bands = [
    (12, 15),
    (15, 18),
    (18, 21),
    (21, 24),
    (24, 27),
    (27, 30),
    (30, 33),
    (33, 36),
    (36, 39),
    (39, 42),
    (42, 45),
]

k_values = [20, 40, 60, 80, 100]

max_type_name = "max_1_2_3"   # lags = [1, 2, 3]

def plot_band_contributions_across_k():

    band_contribution_results = {}

    for k in k_values:

        band_counts = np.zeros(len(bands), dtype=int)

        for subject in subjects:

            file_path = base_path / f"{subject}_maximum_matrices_svm_results_k_{k}.pkl"

            if not file_path.exists():
                print(f"Arquivo não encontrado: {file_path}")
                continue

            with open(file_path, "rb") as f:
                        results = pickle.load(f)

            selected_features = results[subject][max_type_name]["selected_features"]

            for fold_features in selected_features:
                for feature in fold_features:
                    band_idx = feature["band"]

                    band_counts[band_idx] += 1

        relative_band_frequency = band_counts / band_counts.sum()

        band_contribution_results[k] = relative_band_frequency

    # y contém banda nas linhas e K nas colunas
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

    # a invesão afeta somente a legenda, não o gráfico que já foi desenhado
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)
    )

    save_path = output_path / "band_contributions.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    #plt.close()

    print(f"Figura salva em: {save_path}")


if __name__ == "__main__":
    plot_band_contributions_across_k()