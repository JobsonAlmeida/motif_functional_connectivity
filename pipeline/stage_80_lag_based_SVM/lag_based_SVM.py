import os
import numpy as np
from pathlib import Path

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, confusion_matrix

import pickle

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

base_path = project_root / "processed_data" / "stage_60_synchronization_matrix_graph_measure"
output_path = project_root / "processed_data" / current_file.parents[0].name

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]


def print_dict_keys(d, indent=0):

    for key, value in d.items():

        print("  " * indent + f"- {key}")

        if isinstance(value, dict):
            print_dict_keys(value, indent + 1)

def run_lag_based_SVM():


    for subject in subjects:

        results = {}
        graph_sessions = []
        label_sessions = []

        print(f"\n{subject}")

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

            print(f"Shape dos dados da seção: {graph_measures.shape}  \nShape dos rótulos da seção: {labels.shape}")

            graph_sessions.append(graph_measures)
            label_sessions.append(labels)

            

        # Unindo as 3 sessões do sujeito

        if len(graph_sessions) != len(sessions):
            print(f"Pulando {subject}: sessões incompletas")
            continue

        subject_graph_measures = np.concatenate(graph_sessions, axis=0)
        labels = np.concatenate(label_sessions, axis=0)

        if subject_graph_measures.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{subject}: número de épocas diferente do número de rótulos"
            )
        
        print(f"\n{subject} \nShape dos dados: {subject_graph_measures.shape} \nShape dos rótulos: {labels.shape}", )

        results[subject] = {}

        for lag_idx in range(subject_graph_measures.shape[2]):

            # Seleciona um lag específico
            X = subject_graph_measures[:, :, lag_idx, :, :]

            # Transforma em matriz 2D: (n_epochs, n_features)
            X = X.reshape(X.shape[0], -1)

            print(f"\n{subject} \nShape da matriz de features para lag {lag_idx}: {X.shape}")

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight=None,
                    tol=1e-3,
                    max_iter=-1 # Sem número máximo de iterações
                ))
            ])

            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            )


            scores = []
            confusion_matrices = []

            for train_idx, test_idx in cv.split(X, labels):

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                scores.append(acc)

                cm = confusion_matrix(
                    y_test,
                    y_pred,
                    labels=np.unique(labels),
                    normalize="true"
                )

                confusion_matrices.append(cm)

            scores = np.array(scores)
            confusion_matrices = np.array(confusion_matrices)

            mean_confusion_matrix = confusion_matrices.mean(axis=0)

            results[subject][f"lag_{lag_idx}"] = {
                "scores": scores,
                "mean_accuracy": scores.mean(),
                "std_accuracy": scores.std(),
                "confusion_matrices": confusion_matrices,
                "mean_confusion_matrix": mean_confusion_matrix
            }


            print(
                subject,
                f"lag_{lag_idx}",
                "scores:", scores,
                "mean:", scores.mean()
            )

        print_dict_keys(results)            


        save_path = os.path.join(
            output_path,
            f"{subject}_svm_results.pkl"
        )

        with open(save_path, "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":

    run_lag_based_SVM()