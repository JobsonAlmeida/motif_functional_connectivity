import os
import numpy as np
from pathlib import Path


current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

base_path = project_root / "processed_data" / "stage_40_synchronization_matrix"
output_path = project_root / "processed_data" / current_file.parents[0].name

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]

def obtain_maximum_matrices(sync_matrices: np.ndarray) -> np.ndarray:
    
    max_0_1_2_3 = np.max(sync_matrices[:, :, 0:4, :, :], axis=2) #shape (n_epochs, n_bands, n_channels, n_channels)
    max_1_2_3 = np.max(sync_matrices[:, :, 1:4, :, :], axis=2)   #shape (n_epochs, n_bands, n_channels, n_channels)

    maximum_matrices = np.stack([max_0_1_2_3, max_1_2_3],axis=2) #shape (n_epochs, n_bands, 2, n_channels, n_channels)

    return maximum_matrices


def run_maximum_matrix():
        
    for subject in subjects:
        for session in sessions:

            """___Abrindo o aquivo a ser processado___"""

            file_path = os.path.join(
                base_path,
                f"{subject}_{session}_sync_matrices.npy"
            )

            labels_path = os.path.join(
                base_path,
                f"{subject}_{session}_sync_matrices_labels.npy"
            )

            if not os.path.exists(file_path):
                print(f"Arquivo não encontrado: {file_path}")
                continue

            if not os.path.exists(labels_path):
                print(f"Rótulos não encontrados: {labels_path}")
                continue


            """___Processando os dados___"""

            print(f"\nProcessando {current_file.parents[0].name}\n{subject} {session}...")

            sync_matrices = np.load(file_path) #sync_matrices.shape = (n_epochs, n_bands, n_lag, n_channels, n_channels)
            labels = np.load(labels_path)
        
            maximum_matrices = obtain_maximum_matrices(sync_matrices) #maximum_matrices.shape = (n_epochs, n_bands, 2, n_channels, n_channels)

            if labels.shape[0] != maximum_matrices.shape[0]:
                raise ValueError(
                    f"Número de rótulos diferente do número de épocas: "
                    f"{labels.shape[0]} rótulos vs {maximum_matrices.shape[0]} épocas"
                )


            """___Salvando os dados processados___"""

            save_path = os.path.join(
                output_path,
                f"{subject}_{session}_maximum_matrices.npy"
            )

            labels_save_path = os.path.join(
                output_path,
                f"{subject}_{session}_maximum_matrices_labels.npy"
            )

            np.save(save_path, maximum_matrices)
            np.save(labels_save_path, labels)

            "___Imprimindo na tela___"

            print(f"Dados salvos em: {save_path}")
            print(f"Rótulos salvos em: {labels_save_path}")

            print(f"Shape dos dados: {maximum_matrices.shape}")
            print(f"Shape dos rótulos: {labels.shape}")           

if __name__ == "__main__":

    run_maximum_matrix()