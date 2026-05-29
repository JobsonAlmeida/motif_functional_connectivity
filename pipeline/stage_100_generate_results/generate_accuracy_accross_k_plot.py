# generate_accuracy_across_k.py

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
    / "generate_accuracy_across_k"
    / experiment_name
)

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]

k_features = [20, 40, 60, 80, 100]

max_type_names = [
    "max_0_1_2_3",
    "max_1_2_3"
]


def load_results_across_k():
    results_across_k = {
        max_type: {subject: [] for subject in subjects}
        for max_type in max_type_names
    }

    for k in k_features:
        for subject in subjects:

            file_path = base_path / f"{subject}_maximum_matrices_svm_results_k_{k}.pkl"

            if not file_path.exists():
                print(f"Arquivo não encontrado: {file_path}")
                continue

            with open(file_path, "rb") as f:
                results = pickle.load(f)

            for max_type in max_type_names:
                mean_accuracy = results[subject][max_type]["mean_accuracy"]

                results_across_k[max_type][subject].append(mean_accuracy)

    return results_across_k


def plot_accuracy_across_k(results_across_k):
    for max_type in max_type_names:

        plt.figure(figsize=(12, 6))

        for subject in subjects:
            accuracies = results_across_k[max_type][subject]

            if len(accuracies) != len(k_features):
                continue

            plt.plot(
                k_features,
                np.array(accuracies),
                marker="o",
                label=subject
            )

        plt.title(f"Mean SVM Accuracy across k_features - {max_type}")
        plt.xlabel("k_features")
        plt.ylabel("Mean accuracy")
        plt.xticks(k_features)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Subject", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        save_path = output_path / f"mean_accuracy_across_k_{max_type}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Imagem salva em: {save_path}")


def generate_accuracy_accross_k_plot():

    results_across_k = load_results_across_k()
    plot_accuracy_across_k(results_across_k)


if __name__ == "__main__":

    generate_accuracy_accross_k_plot()