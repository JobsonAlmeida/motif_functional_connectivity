import os
import numpy as np
from pathlib import Path

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector

import pickle

from pipeline.config.build_feature_selector import build_feature_selector
from pipeline.config.channel_mapping import INDICES_CHANNELS_MAPPING

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

base_path = project_root / "processed_data" / "stage_70_maximum_matrix_graph_measure"
output_path = project_root / "processed_data" / current_file.parents[0].name

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]

def evaluate_svm_cv(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = []
    cms = []

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        ))
    ])

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        scores.append(accuracy_score(y_test, y_pred))

        cms.append(confusion_matrix(
            y_test,
            y_pred,
            labels=np.unique(y),
            normalize="true"
        ))

    scores = np.array(scores)
    cms = np.array(cms)

    return scores, cms

def run_maximum_matrix_based_SVM_original_flow(
    k_features = 3,
    feature_selector = "f_classif",
    output_path = output_path,
    
    ):

    os.makedirs(output_path, exist_ok=True)

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

            X = subject_graph_measures[:, :, max_type, :, :]
            X_original = X.copy()

            n_epochs = X.shape[0]
            n_bands = X.shape[1]
            n_channels = X.shape[2]
            n_measures = X.shape[3]

            X = X.reshape(n_epochs, -1)
            y = labels.copy()

            n_features = X.shape[1]

            # ===============================
            # 1) Avaliação individual
            # ===============================

            individual_results = []

            for feature_idx in range(n_features):

                X_feature = X[:, feature_idx].reshape(-1, 1)

                scores, cms = evaluate_svm_cv(X_feature, y)

                band_idx, channel_idx, measure_idx = np.unravel_index(
                    feature_idx,
                    (n_bands, n_channels, n_measures)
                )

                individual_results.append({
                    "feature_idx": int(feature_idx),
                    "band_index": int(band_idx),
                    "channel_index": int(channel_idx),
                    "channel_name": INDICES_CHANNELS_MAPPING[channel_idx],
                    "measure": int(measure_idx),
                    "scores": scores,
                    "mean_accuracy": scores.mean(),
                    "std_accuracy": scores.std(),
                    "confusion_matrices": cms,
                    "mean_confusion_matrix": cms.mean(axis=0)
                })

            # Ranking da melhor para a pior feature
            ranking = sorted(
                individual_results,
                key=lambda x: x["mean_accuracy"],
                reverse=True
            )

            ranked_feature_indices = [
                item["feature_idx"] for item in ranking
            ]

            # ===============================
            # 2) Avaliação acumulada top-k
            # ===============================

            topk_results = []

            for k in range(1, n_features + 1):

                selected_indices = ranked_feature_indices[:k]

                X_topk = X[:, selected_indices]

                scores, cms = evaluate_svm_cv(X_topk, y)

                topk_results.append({
                    "k": k,
                    "selected_feature_indices": selected_indices,
                    "scores": scores,
                    "mean_accuracy": scores.mean(),
                    "std_accuracy": scores.std(),
                    "confusion_matrices": cms,
                    "mean_confusion_matrix": cms.mean(axis=0)
                })

            # ===============================
            # 3) Melhor grupo de features
            # ===============================

            best_result = max(
                topk_results,
                key=lambda x: x["mean_accuracy"]
            )

            best_k = best_result["k"]
            best_feature_indices = best_result["selected_feature_indices"]

            best_features_info = []

            for feature_idx in best_feature_indices:

                band_idx, channel_idx, measure_idx = np.unravel_index(
                    feature_idx,
                    (n_bands, n_channels, n_measures)
                )

                best_features_info.append({
                    "feature_idx": int(feature_idx),
                    "band_index": int(band_idx),
                    "channel_index": int(channel_idx),
                    "channel_name": INDICES_CHANNELS_MAPPING[channel_idx],
                    "measure": int(measure_idx)
                })

            results[subject][max_type_names[max_type]] = {
                "individual_ranking": ranking,
                "ranked_feature_indices": ranked_feature_indices,
                "topk_results": topk_results,
                "best_k": best_k,
                "best_accuracy": best_result["mean_accuracy"],
                "best_std_accuracy": best_result["std_accuracy"],
                "best_scores": best_result["scores"],
                "best_confusion_matrix": best_result["mean_confusion_matrix"],
                "best_features_info": best_features_info
            }

            print(
                subject,
                max_type_names[max_type],
                "best k:", best_k,
                "best mean accuracy:", best_result["mean_accuracy"]
            )

        # save_path = os.path.join(
        #     output_path,
        #     f"{subject}_maximum_matrices_svm_results_k_{k_features}.pkl"
        # )

        # with open(save_path, "wb") as f:
        #     pickle.dump(results, f)

        save_path = os.path.join(
            output_path,
            f"{subject}_ranked_features_svm_results.pkl"
        )

        with open(save_path, "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":

    run_maximum_matrix_based_SVM_original_flow()