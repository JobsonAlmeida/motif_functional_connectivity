import os
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import json

from pipeline.stage_100_generate_results.generate_confusion_matrix_for_maximum_matrix import run_generate_confusion_matrix_for_maximum_matrix

from pipeline.stage_90_maximum_matrix_based_SVM.maximum_matrix_based_SVM_old import (
    run_maximum_matrix_based_SVM
)

from pipeline.config.channel_mapping import INDICES_CHANNELS_MAPPING

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

base_path = project_root / "processed_data" / "stage_90_maximum_matrix_based_SVM"

result_name = "run_stage_90"
output_path = project_root / "results" / current_file.parents[0].name / result_name
os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]

max_type_names = {
    0: "max_0_1_2_3",
    1: "max_1_2_3"
}


def run_stage_90(
    experiment_name = "experiment_base",
    use_feature_selector=True,
    max_k_features=None,
    specific_k_features = False,
    selected_epochs=None,
    selected_bands=None,
    selected_channels=None,
    selected_measures=None
):

    # # Lauch error if both are True at the same time
    # if max_k_features and specific_k_features:
    #     raise ValueError("Cannot use 'max_k_features' and 'specific_k_feature' at the same time.")

    if max_k_features is None:

        selected_bands_size = 11 if selected_bands is None else len(selected_bands)
        selected_channels_size = 128 if selected_channels is None else len(selected_channels)
        selected_measures_size = 2 if selected_measures is None else len(selected_measures)

        max_k_features = (
            selected_bands_size
            * selected_channels_size
            * selected_measures_size
        )


    # salvando as configurações do experimento
    config = {
        "use_feature_selector": use_feature_selector,
        "max_k_features": max_k_features,
        "selected_epochs": selected_epochs,
        "selected_bands": selected_bands,
        "selected_channels": selected_channels,
        "selected_measures": selected_measures
    }

    config_path = output_path / experiment_name / "experiment_config.json"   
    try:
        config_path.parent.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"Error: The experiment '{experiment_name}' already exists in '{output_path}'. Delete the folder or change the name.")
   

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


    if use_feature_selector is not True:

        k_features = None

        run_maximum_matrix_based_SVM(
            config,
            config_path,
            use_feature_selector=use_feature_selector,
            k_features=k_features,
            selected_epochs=selected_epochs,
            selected_bands=selected_bands,
            selected_channels=selected_channels,
            selected_measures=selected_measures
        )

        run_generate_confusion_matrix_for_maximum_matrix(output_path/experiment_name)

    elif use_feature_selector is True and isinstance(specific_k_features, (int)) and specific_k_features != 0:

        run_maximum_matrix_based_SVM(
            config,
            config_path,
            use_feature_selector=use_feature_selector,
            k_features=specific_k_features,
            selected_epochs=selected_epochs,
            selected_bands=selected_bands,
            selected_channels=selected_channels,
            selected_measures=selected_measures
        )

        run_generate_confusion_matrix_for_maximum_matrix(output_path/experiment_name)
        

    elif use_feature_selector is True and isinstance(max_k_features, (int)) and max_k_features != 0:


        # Guarda os resultados assim:
        # accuracies["max_0_1_2_3"]["sub-01"] = [acc_k1, acc_k2, ...]
        accuracies = {
            "max_0_1_2_3": {subject: [] for subject in subjects},
            "max_1_2_3": {subject: [] for subject in subjects}
        }

        k_values = list(range(1, max_k_features + 1))


        for k_features in k_values:

            print(f"\nExecutando SVM com k_features = {k_features}")

            run_maximum_matrix_based_SVM(
                config,
                config_path,
                use_feature_selector=use_feature_selector,
                k_features=k_features,
                selected_epochs=selected_epochs,
                selected_bands=selected_bands,
                selected_channels=selected_channels,
                selected_measures=selected_measures
            )

            for subject in subjects:

                file_path = base_path / f"{subject}_maximum_matrices_svm_results.pkl"

                if not file_path.exists():
                    print(f"Arquivo não encontrado: {file_path}")
                    continue

                with open(file_path, "rb") as f:
                    results = pickle.load(f)

                for max_type_name in max_type_names.values():

                    mean_accuracy = results[subject][max_type_name]["mean_accuracy"]

                    accuracies[max_type_name][subject].append(mean_accuracy)

        generate_accuracy_plots(
            experiment_name,
            accuracies=accuracies,
            k_values=k_values,
            
        )

        top_n = min(3, len(k_values))    
        save_top_n_accuracies(
            experiment_name,
            accuracies,
            k_values,
            top_n=top_n
        )

        plot_top_n_accuracies_from_json(
            experiment_name,
            top_n=top_n
        )

def generate_accuracy_plots(experiment_name, accuracies, k_values):

    for max_type_name, subject_results in accuracies.items():

        plt.figure(figsize=(12, 7))

        for subject, mean_accuracies in subject_results.items():

            plt.plot(
                k_values[:len(mean_accuracies)],
                mean_accuracies,
                marker="o",
                label=subject
            )

        plt.title(f"Mean accuracy vs k_features - {max_type_name}")
        plt.xlabel("k_features")
        plt.ylabel("Mean accuracy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        save_path = output_path / experiment_name/ f"{max_type_name}_mean_accuracy_by_k_features.png"

        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Imagem salva em: {save_path}")

        save_path = output_path / experiment_name/ f"accuracies.npy"
        np.save(save_path, accuracies)

        save_path = output_path / experiment_name/ f"k_values.npy"
        np.save(save_path, k_values)

def save_top_n_accuracies(
    experiment_name,
    accuracies,
    k_values,
    top_n=3
):

    top_results = {}

    for max_type_name, subject_results in accuracies.items():

        top_results[max_type_name] = {}

        for subject, mean_accuracies in subject_results.items():

            mean_accuracies = np.array(mean_accuracies)

            k_array = np.array(
                k_values[:len(mean_accuracies)]
            )

            top_indices = np.argsort(
                mean_accuracies
            )[-top_n:][::-1]

            top_results[max_type_name][subject] = [

                {
                    "rank": rank + 1,

                    "k_features": int(
                        k_array[index]
                    ),

                    "mean_accuracy": float(
                        mean_accuracies[index]
                    )
                }

                for rank, index in enumerate(top_indices)
            ]

    save_path = (
        output_path
        / experiment_name
        / f"top_{top_n}_accuracies.json"
    )

    with open(save_path, "w") as f:

        json.dump(
            top_results,
            f,
            indent=4
        )

    print(
        f"Top {top_n} accuracies saved in: {save_path}"
    )


def plot_top_n_accuracies_from_json(experiment_name, top_n=3):

    json_path = output_path / experiment_name / f"top_{top_n}_accuracies.json"

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

if __name__ == "__main__":

    run_stage_90()