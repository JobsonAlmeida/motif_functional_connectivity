import os
import numpy as np
from pathlib import Path

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif


import pickle
import json


from pipeline.config.channel_mapping import CHANNELS_INDICES_MAPPING



current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

base_path = project_root / "processed_data" / "stage_70_maximum_matrix_graph_measure"
output_path = project_root / "processed_data" / current_file.parents[0].name

os.makedirs(output_path, exist_ok=True)

#subjects = [f"sub-{i:02d}" for i in range(1, 11)]
subjects = ["sub-01"]

sessions = [f"ses-{i:02d}" for i in range(1, 4)]



def run_maximum_matrix_based_SVM_sequential_feature_selector(
    config = {},
    config_path = "./pipeline/stage_90_1_maximum_matrix_based_SVM_sequential_feature_selector/experiment_config.json",
    use_feature_selector=True,
    k_features=1,
    selected_epochs=None,
    selected_bands=None,
    selected_channels=None,
    selected_measures=None
    ):


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

            #X.shape = (epochs, bands, max_types, channels, measures)
            # Seleciona um tipo de matriz de máximo:
            # max_type = 0 -> Max(0,1,2,3)
            # max_type = 1 -> Max(1,2,3)
            X = subject_graph_measures[:, :, max_type, :, :] #X.shape = (epochs, bands, channels, measures)
            labels_current = labels.copy()

            
            # Seleção de épocas
            if selected_epochs is not None:
                X = X[selected_epochs]
                labels_current = labels_current[selected_epochs]

            # Seleção de bandas
            if selected_bands is not None:
                X = X[:, selected_bands, :, :]

            # Seleção de canais
            if selected_channels is not None:
                selected_channels_indices = [
                    CHANNELS_INDICES_MAPPING[ch]
                    for ch in selected_channels
                ]
                X = X[:, :, selected_channels_indices, :]

            # Seleção de measures
            if selected_measures is not None:
                X = X[:, :, :, selected_measures]

            X = X.reshape(X.shape[0], -1)

            # model = Pipeline([
            #     ("scaler", StandardScaler()),

            #     ("selector", SelectKBest(
            #         score_func=f_classif,
            #         k=2000
            #     )),               

            #     ("svm", SVC(
            #         kernel="rbf",
            #         C=1.0,
            #         gamma="scale",
            #         class_weight=None,
            #         tol=1e-3,
            #         max_iter=-1 
            #     ))

            # ])

            steps = [
                ("scaler", StandardScaler())
            ]

            # if use_feature_selector:
            #     steps.append(
            #         ("selector", SelectKBest(
            #             score_func=f_classif,
            #             k=k_features
            #         ))
            #     )

            if use_feature_selector:

                selector_estimator = SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight=None,
                    tol=1e-3,
                    max_iter=-1
                )

                steps.append(
                    ("selector", SequentialFeatureSelector(
                        estimator=selector_estimator,
                        n_features_to_select=k_features,
                        direction="forward",
                        scoring="accuracy",
                        cv=5,
                        n_jobs=-1
                    ))
                )

            steps.append(
                ("svm", SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight=None,
                    tol=1e-3,
                    max_iter=-1
                ))
            )

            model = Pipeline(steps)

            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            )

            if "model_config" not in config:

                model_config = {
                    key: str(value)
                    for key, value in model.get_params().items()
                }   

                config["model_config"] = model_config  
                
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)              


            scores = []
            confusion_matrices = []

            for train_idx, test_idx in cv.split(X, labels_current):

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = labels_current[train_idx], labels_current[test_idx]

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                scores.append(acc)

                cm = confusion_matrix(
                    y_test,
                    y_pred,
                    labels=np.unique(labels_current),
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

    run_maximum_matrix_based_SVM_sequential_feature_selector()