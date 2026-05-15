import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

base_path = project_root / "processed_data" / "stage_90_maximum_matrix_based_SVM"

result_name = "accuracy_table"
output_path = project_root / "results" / current_file.parents[0].name / result_name

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]


def run_generate_confusion_matrix_for_maximum_matrix():

    class_names = ["up", "down", "left", "right"]

    for max_type_name in ["max_1_2_3", "max_0_1_2_3" ]:

        fig, axes = plt.subplots(2, 5, figsize=(16, 7))
        axes = axes.ravel()

        for idx, subject in enumerate(subjects):

            file_path = os.path.join(
                base_path,
                f"{subject}_maximum_matrices_svm_results.pkl"
            )

            with open(file_path, "rb") as f:
                results = pickle.load(f)

            cm = results[subject][max_type_name]["mean_confusion_matrix"]

            best_idx = np.argmax(np.diag(cm))
            worst_idx = np.argmin(np.diag(cm))

            best_class = class_names[best_idx]
            worst_class = class_names[worst_idx]

            ax = axes[idx]

            im = ax.imshow(cm, vmin=0, vmax=1)

            ax.set_title(
                f"{subject} | best={best_class}, worst={worst_class}",
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
            fontsize=14
        )

        cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("Proportion (row-normalized)")

        plt.tight_layout(rect=[0, 0, 0.95, 0.94])

        save_path = os.path.join(
            output_path,
            f"figure_confusion_matrices_{max_type_name}.png"
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        #plt.show()

if __name__ == "__main__":

    run_generate_confusion_matrix_for_maximum_matrix()