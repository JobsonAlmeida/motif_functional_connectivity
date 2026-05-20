import os
import numpy as np
from pathlib import Path

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif


import pickle

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

base_path = project_root / "processed_data" / "stage_70_maximum_matrix_graph_measure"
output_path = project_root / "processed_data" / current_file.parents[0].name

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]

def run_maximum_matrix_based_SVM():


    for subject in subjects:

        results = {}
        graph_sessions = []
        label_sessions = []

        for session in sessions:

            """___Abrindo o aquivo a ser processado___"""

            file_path = os.path.join(
                base_path,
                f"{subject}_{session}_maximum_matrices_graph_measures.npy"
            )

            labels_path = os.path.join(
                base_path,
                f"{subject}_{session}_maximum_matrices_graph_measures_labels.npy"
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

        max_type_names = {
            0: "max_0_1_2_3",
            1: "max_1_2_3"
        }

        for max_type in range(subject_graph_measures.shape[2]):

            # Seleciona um tipo de matriz de máximo:
            # max_type = 0 -> Max(0,1,2,3)
            # max_type = 1 -> Max(1,2,3)
            X = subject_graph_measures[:, :, max_type, :, :]

            # Transforma em matriz 2D: (n_epochs, n_features)
            
            X = X.reshape(X.shape[0], -1) #X.shape = (n_epochs, n_bands * n_channels * n_measures)

            model = Pipeline([
                ("scaler", StandardScaler()),

                ("selector", SelectKBest(
                    score_func=f_classif,
                    k=2000
                )),               

                ("svm", SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight=None,
                    tol=1e-3,
                    max_iter=-1 
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


                        
            results[subject][max_type_names[max_type]] = {
                "scores": scores,
                "mean_accuracy": scores.mean(),
                "std_accuracy": scores.std(),
                "confusion_matrices": confusion_matrices,
                "mean_confusion_matrix": mean_confusion_matrix
            }


            print(
                subject,
                max_type_names[max_type],
                "scores:", scores,
                "mean:", scores.mean()
            )

        save_path = os.path.join(
            output_path,
            f"{subject}_maximum_matrices_svm_results.pkl"
        )

        with open(save_path, "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":

    run_maximum_matrix_based_SVM()