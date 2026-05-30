# generate_accuracy_across_k.py

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

subjects = [f"sub-{i:02d}" for i in range(1, 11)]

max_type_names = [
    "max_0_1_2_3",
    "max_1_2_3"
]


def load_results_across_k(k_features, base_path):
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


def plot_accuracy_across_k(results_across_k, k_features, output_path):
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


def obtain_top_n_accuracies_by_subject_and_k():
    None

def plot_top_n_accuracies():

   with open(json_path, "r") as f:
        top_results = json.load(f)

    for max_type_name, subjects_data in top_results.items():

        subjects = list(subjects_data.keys())
        x = np.arange(len(subjects))

        width = 0.8 / top_n

        plt.figure(figsize=(14, 7))

        for rank in range(top_n):

            accuracies = []
            k_values = []

            for subject in subjects:

                top_item = subjects_data[subject][rank]

                accuracies.append(top_item["mean_accuracy"])
                k_values.append(top_item["k_features"])

            bars = plt.bar(
                x + rank * width,
                accuracies,
                width=width,
                label=f"Top {rank + 1}"
            )

            for bar, k_value in zip(bars, k_values):

                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"k={k_value}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90
                )

        plt.xticks(
            x + width * (top_n - 1) / 2,
            subjects,
            rotation=45
        )

        plt.xlabel("Subject")
        plt.ylabel("Mean accuracy")
        plt.title(f"Top {top_n} accuracies by subject - {max_type_name}")

        plt.legend()
        plt.grid(True, axis="y")
        plt.tight_layout()

        save_path = (
            output_path
            / experiment_name
            / f"{max_type_name}_top_{top_n}_accuracies_with_k.png"
        )

        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Imagem salva em: {save_path}")
    



def generate_top_n_accuracies_plot(
        experiment_name = "experiment_base",
        top_n = 3
):
    
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
        /experiment_name
    )

    os.makedirs(output_path, exist_ok=True)

    top_n_accuracies_by_subjetc_and_k = obtain_top_n_accuracies_by_subject_and_k(top_n, base_path)

    plot_top_n_accuracies(top_n_accuracies_by_subjetc_and_k, output_path)


if __name__ == "__main__":

    generate_top_n_accuracies_plot()