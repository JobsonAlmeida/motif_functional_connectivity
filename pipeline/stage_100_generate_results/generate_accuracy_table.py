import os
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
import networkx as nx

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

import pickle


current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

base_path = project_root / "processed_data" / "stage_80_lag_based_SVM"
output_path = project_root / "processed_data" / current_file.parents[0].name

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]

def run_generate_accuracy_table():

    for subject in subjects:

        graph_sessions = []
        label_sessions = []

        for session in sessions:

            """___Abrindo o aquivo a ser processado___"""

            file_path = os.path.join(
                base_path,
                f"{subject}_{session}_graph_measures.npy"
            )

            labels_path = os.path.join(
                base_path,
                f"{subject}_{session}_graph_measures_labels.npy"
            )

            if not os.path.exists(file_path):
                print(f"Arquivo não encontrado: {file_path}")
                continue

            if not os.path.exists(labels_path):
                print(f"Rótulos não encontrado: {labels_path}")
                continue


            graph_measures = np.load(file_path)
            labels = np.load(labels_path)

            graph_sessions.append(graph_measures)
            label_sessions.append(labels)

        # Unindo as 3 sessões do sujeito
        subject_graph_measures = np.concatenate(graph_sessions, axis=0)
        labels = np.concatenate(label_sessions, axis=0)

        print(f"\n{subject} \nShape dos dados: {subject_graph_measures.shape} \nShape dos rótulos: {labels.shape}", )

        results[subject] = {}

        for lag_idx in range(subject_graph_measures.shape[2]):

            # Seleciona um lag específico
            X = subject_graph_measures[:, :, lag_idx, :, :]

            # Transforma em matriz 2D: (n_epochs, n_features)
            X = X.reshape(X.shape[0], -1)

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight=None,
                    tol=1e-3,
                    max_iter=-1 # o treinamento 
                ))
            ])

            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            )

            scores = cross_val_score(
                model,
                X,
                labels,
                cv=cv,
                scoring="accuracy"
            )

            results[subject][f"lag_{lag_idx}"] = {
                "scores": scores,
                "mean_accuracy": scores.mean(),
                "std_accuracy": scores.std()
            }

            save_path = os.path.join(
                output_path,
                f"{subject}_lag_{lag_idx}_svm_results.pkl"
            )

            with open(save_path, "wb") as f:
                pickle.dump(results, f)

            print(
                subject,
                f"lag_{lag_idx}",
                "scores:", scores,
                "mean:", scores.mean()
            )


if __name__ == "__main__":

    run_generate_accuracy_table()