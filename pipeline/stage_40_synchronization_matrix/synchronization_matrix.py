import os
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view


current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

base_path = project_root / "processed_data" / "stage_30_motif_sequence"
output_path = project_root / "processed_data" / current_file.parents[0].name

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]

def obtain_synchronization_matrices(array: np.ndarray) -> np.ndarray:
    """
    array: (n_epochs, n_bands, n_channels, n_motifs)

    retorna:
    sync_matrices: (n_epochs, n_bands, n_lags, n_channels, n_channels)

    Para cada lag, compara as sequências de motifs entre todos os pares de canais.
    """

    n_lags = 4
    n_epochs, n_bands, n_channels, n_motifs = array.shape

    sync_matrices = np.zeros(
        (n_epochs, n_bands, n_lags, n_channels, n_channels),
        dtype=np.float32
    )

    for lag in range(n_lags):

        if lag == 0:
            seq1 = array
            seq2 = array
        else:
            seq1 = array[..., :-lag]
            seq2 = array[..., lag:]

        # Obtendo matches channel a channel   
        # A última dimensão tem True e False para cada motif comparado 
        matches = seq1[:, :, :, None, :] == seq2[:, :, None, :, :]  # matches.shape = (n_epochs, n_bands, n_channels, n_channels, n_motifs) 

        # Calcula a soma (contagem de acertos)
        sync = matches.sum(axis=-1) #sync.shape = (n_epochs, n_bands, n_channels, n_channels)

        # Normaliza pelo valor máximo presente em cada matriz
        sync = (sync - sync.min()) / (sync.max() - sync.min())

        sync_matrices[:, :, lag, :, :] = sync #sync_matrices.shape = (n_epochs, n_bands, n_lag, n_channels, n_channels)

    return sync_matrices


def run_synchronization_matrix():
        
    for subject in subjects:
        for session in sessions:

            """___Abrindo o aquivo a ser processado___"""

            file_path = os.path.join(
                base_path,
                f"{subject}_{session}_inner_bands_motifs.npy"
            )

            if not os.path.exists(file_path):
                print(f"Arquivo não encontrado: {file_path}")
                continue

            """___Processando os dados___"""

            print(f"\nProcessando {current_file.parents[0].name}\n{subject} {session}...")

            array_bands_motifs = np.load(file_path) #(épocas × bandas × canais × motifs)
        
            sync_matrices = obtain_synchronization_matrices(array_bands_motifs) #sync_matrices.shape = (n_epochs, n_bands, n_lag, n_channels, n_channels)

            """___Salvando os dados processados__"""

            save_path = os.path.join(
                output_path,
                f"{subject}_{session}_sync_matrices.npy"
            )

            np.save(save_path, sync_matrices)

            print(f"Salvo em: {save_path}")
            print(f"Shape: {sync_matrices.shape}")

if __name__ == "__main__":

    run_synchronization_matrix()