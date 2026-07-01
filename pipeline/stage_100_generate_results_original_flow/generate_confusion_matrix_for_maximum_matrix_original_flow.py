import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

# base_path = project_root / "processed_data" / "stage_90_maximum_matrix_based_SVM"

#result_name = "confusion_maximum_matrices"
# output_path = project_root / "results" / current_file.parents[0].name / result_name


subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]


def run_generate_confusion_matrix_for_maximum_matrix_original_flow(
        k_feature = 3 ,
        experiment_name = "experiment_base",
    ):

    # base_path = (
    #     project_root
    #     / "results"
    #     / current_file.parents[0].name
    #     / "run_stage_90_in_loop"
    #     / experiment_name

    # )

    base_path = (
        project_root
        / "processed_data"
        / "stage_90_maximum_matrix_based_SVM"
    )

    output_path = (
        project_root
        / "results"
        / current_file.parents[0].name
        / "confusion_maximum_matrices"
        
    )

    os.makedirs(output_path, exist_ok=True)

    class_names = ["up", "down", "left", "right"]

    for max_type_name in ["max_0_1_2_3", "max_1_2_3" ]:

        fig, axes = plt.subplots(2, 5, figsize=(16, 7))
        axes = axes.ravel()

        for idx, subject in enumerate(subjects):

            file_path = os.path.join(
                base_path,
                f"{subject}_ranked_features_svm_results.pkl"
            )

            with open(file_path, "rb") as f:
                results = pickle.load(f)

            cm = results[subject][max_type_name]["best_confusion_matrix"]

            best_idx = np.argmax(np.diag(cm))
            worst_idx = np.argmin(np.diag(cm))

            best_class = class_names[best_idx]
            worst_class = class_names[worst_idx]

            ax = axes[idx]

            im = ax.imshow(cm, vmin=0, vmax=1)

            ax.set_title(
                f"{subject} | best={best_class}, worst={worst_class},\nbest_k={results[subject][max_type_name]["best_k"]}",
                fontsize=9
            )

            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))

            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names)

            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{cm[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8
                    )

        fig.suptitle(
            f"Normalized mean confusion matrices | {max_type_name}",
            #f"\nk feature = {k_feature}",
            fontsize=14
        )

        #cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        # cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.04)
        # cbar.set_label("Proportion (row-normalized)")

        # cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.03)
        # cbar.set_label("Proportion (row-normalized)", fontsize=10)

        #plt.tight_layout(rect=[0, 0, 0.95, 0.94])
        # plt.tight_layout()

        fig.subplots_adjust(
            left=0.05,
            right=0.86,
            bottom=0.10,
            top=0.88,
            wspace=0.45,
            hspace=0.45
        )

        cax = fig.add_axes([0.89, 0.22, 0.015, 0.55])

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Proportion (row-normalized)", fontsize=10)

        #plt.tight_layout(rect=[0, 0, 0.92, 0.95])

        save_path = os.path.join(
            output_path,
            f"figure_confusion_matrices_{max_type_name}.png"
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

if __name__ == "__main__":

    run_generate_confusion_matrix_for_maximum_matrix_original_flow()