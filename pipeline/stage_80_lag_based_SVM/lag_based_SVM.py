import os
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
import networkx as nx

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

base_path = project_root / "processed_data" / "stage_60_synchronization_matrix_graph_measure"
output_path = project_root / "processed_data" / current_file.parents[0].name

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]

def run_lag_based_SVM():
        
    data_by_subject_and_lag = {}

    for subject in subjects:
        
        for session in sessions:

            session_arrays = []

            """___Abrindo o aquivo a ser processado___"""

            file_path = os.path.join(
                base_path,
                f"{subject}_{session}_graph_measures.npy"
            )

            if not os.path.exists(file_path):
                print(f"Arquivo não encontrado: {file_path}")
                continue

            graph_measures = np.load(file_path)

            session_arrays.append(graph_measures)

        subject_graph_measures = np.concatenate(session_arrays, axis=0)

        data_by_subject_and_lag[subject] = {}


        for lag_idx in range(subject_graph_measures.shape[2]):

            X_lag = subject_graph_measures[:, :, lag_idx, :, :]
            X_lag = X_lag.reshape(X_lag.shape[0], -1)

            data_by_subject_and_lag[subject][f"lag_{lag_idx}"] = X_lag

            print(subject, f"lag_{lag_idx}", X_lag.shape)


            # """___Processando os dados___"""

            # print(f"\nProcessando {current_file.parents[0].name}\n{subject} {session}...")

            # sync_matrices = np.load(file_path)  #sync_matrices.shape = (n_epochs, n_bands, n_lag, n_channels, n_channels)
        
            # graph_measures = obtain_graph_measures(sync_matrices) #graph_measures.shape = (n_epochs, n_bands, n_lags, n_channels, 2)

            # """___Salvando os dados processados__"""

            # save_path = os.path.join(
            #     output_path,
            #     f"{subject}_{session}_graph_measures.npy"
            # )

            # np.save(save_path, graph_measures)

            # print(f"Salvo em: {save_path}")
            # print(f"Shape: {graph_measures.shape}")

if __name__ == "__main__":

    run_lag_based_SVM()