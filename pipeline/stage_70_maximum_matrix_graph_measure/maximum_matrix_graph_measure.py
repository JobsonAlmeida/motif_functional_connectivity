import os
import numpy as np
from pathlib import Path
import networkx as nx

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

base_path = project_root / "processed_data" / "stage_50_maximum_matrix"
output_path = project_root / "processed_data" / current_file.parents[0].name

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]

def obtain_graph_measures(sync_matrices: np.ndarray) -> np.ndarray:

    n_epochs, n_bands, n_max_types, n_channels_i, n_channels_j = sync_matrices.shape

    if n_channels_i != n_channels_j:
        raise ValueError("A matriz de conectividade precisa ser quadrada.")

    n_channels = n_channels_i

    n_measures = 2
    graph_measures = np.zeros(
        (n_epochs, n_bands, n_max_types, n_channels, n_measures),
        dtype=np.float32
    )


    #Obtenado o out-strength
    graph_measures[:, :, :, :, 0] = sync_matrices.sum(axis=-1)

    #Calculando o PageRank
    for epoch in range(n_epochs):
        for band in range(n_bands):
            for max_type in range(n_max_types):

                A = sync_matrices[epoch, band, max_type]

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
                pr_dict = nx.pagerank(
                    G, 
                    weight="weight",
                    max_iter=1000,
                    tol=1e-6
                )

                graph_measures[epoch, band, max_type, :, 1] = np.array(
                    [pr_dict[ch] for ch in range(n_channels)]
                )

                #print(graph_measures[epoch, band, max_type])

    graph_measures[:, :, :, :, 1] = graph_measures[:, :, :, :, 1]*100

    return graph_measures 

def run_maximum_matrix_graph_measure():
        
    for subject in subjects:
        for session in sessions:

            """___Abrindo o aquivo a ser processado___"""

            file_path = os.path.join(
                base_path,
                f"{subject}_{session}_maximum_matrices.npy"
            )

            labels_path = os.path.join(
                base_path,
                f"{subject}_{session}_maximum_matrices_labels.npy"
            )

            if not os.path.exists(file_path):
                print(f"Arquivo não encontrado: {file_path}")
                continue

            if not os.path.exists(labels_path):
                print(f"Rótulos não encontrado: {labels_path}")
                continue

            """___Processando os dados___"""

            print(f"\nProcessando {current_file.parents[0].name}\n{subject} {session}...")

            maximum_matrices = np.load(file_path) #maximum_matrices.shape = (n_epochs, n_bands, 2, n_channels, n_channels)
            labels = np.load(labels_path)

            if labels.shape[0] != maximum_matrices.shape[0]:
                raise ValueError(
                    f"Número de rótulos diferente do número de épocas: "
                    f"{labels.shape[0]} rótulos vs {maximum_matrices.shape[0]} épocas"
                )
        
            maximum_matrices_graph_measures = obtain_graph_measures(maximum_matrices) #graph_measures.shape = (n_epochs, n_bands, 2, n_channels, 2)


            """___Salvando os dados processados__"""

            save_path = os.path.join(
                output_path,
                f"{subject}_{session}_maximum_matrices_graph_measures.npy"
            )

            labels_save_path = os.path.join(
                output_path,
                f"{subject}_{session}_maximum_matrices_graph_measures_labels.npy"
            )

            np.save(save_path, maximum_matrices_graph_measures)
            np.save(labels_save_path, labels)

            "___ imprimindo na tela___"

            print(f"Dados salvos em: {save_path}")
            print(f"Rótulos salvos em: {labels_save_path}")

            print(f"Shape dos dados: {maximum_matrices_graph_measures.shape}")
            print(f"Shape dos rótulos: {labels.shape}") 

if __name__ == "__main__":

    run_maximum_matrix_graph_measure()