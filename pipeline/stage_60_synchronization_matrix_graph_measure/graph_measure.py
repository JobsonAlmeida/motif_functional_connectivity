import os
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
import networkx as nx

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

base_path = project_root / "processed_data" / "stage_40_synchronization_matrix"
output_path = project_root / "processed_data" / current_file.parents[0].name

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]

def obtain_graph_measures(sync_matrices: np.ndarray) -> np.ndarray:

    n_epochs, n_bands, n_lags, n_channels_i, n_channels_j = sync_matrices.shape

    if n_channels_i != n_channels_j:
        raise ValueError("A matriz de conectividade precisa ser quadrada.")

    n_channels = n_channels_i

    n_measures = 2
    graph_measures = np.zeros(
        (n_epochs, n_bands, n_lags, n_channels, n_measures),
        dtype=np.float32
    )


    #Obtenado o out-strength
    graph_measures[:, :, :, :, 0] = sync_matrices.sum(axis=-1)

    #Calculando o PageRank
    for epoch in range(n_epochs):
        for band in range(n_bands):
            for lag in range(n_lags):

                A = sync_matrices[epoch, band, lag]

                # Converte a matriz A em um grafo direcionado
                G = nx.from_numpy_array(A, create_using=nx.DiGraph)

                """
                nx.pagerank Calcula o PageRank de cada nó (canal)
                weight="weight" faz usar os valores de A[i, j] como pesos reais
                pr_dict é um dicionário. Exemplo:
                {
                    0: 0.012,
                    1: 0.008,
                    ...
                    127: 0.015
                }

                cada chave = canal
                cada valor = PageRank
                """                
                pr_dict = nx.pagerank(G, weight="weight")

                graph_measures[epoch, band, lag, :, 1] = np.array(
                    [pr_dict[ch] for ch in range(n_channels)]
                )

                print(graph_measures[epoch, band, lag])

    graph_measures[:, :, :, :, 1] = graph_measures[:, :, :, :, 1]*100

    return graph_measures 

def run_synchronization_matrix_graph_measure():
        
    for subject in subjects:
        for session in sessions:

            """___Abrindo o aquivo a ser processado___"""

            file_path = os.path.join(
                base_path,
                f"{subject}_{session}_sync_matrices.npy"
            )

            if not os.path.exists(file_path):
                print(f"Arquivo não encontrado: {file_path}")
                continue

            """___Processando os dados___"""

            print(f"\nProcessando {current_file.parents[0].name}\n{subject} {session}...")

            sync_matrices = np.load(file_path)  #sync_matrices.shape = (n_epochs, n_bands, n_lag, n_channels, n_channels)
        
            graph_measures = obtain_graph_measures(sync_matrices) #graph_measures.shape = (n_epochs, n_bands, n_lags, n_channels, 2)

            """___Salvando os dados processados__"""

            save_path = os.path.join(
                output_path,
                f"{subject}_{session}_graph_measures.npy"
            )

            np.save(save_path, graph_measures)

            print(f"Salvo em: {save_path}")
            print(f"Shape: {graph_measures.shape}")

if __name__ == "__main__":

    run_synchronization_matrix_graph_measure()