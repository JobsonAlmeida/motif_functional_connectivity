import os
import numpy as np
from pathlib import Path


current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

base_path = project_root / "processed_data" / "stage_30_motif_sequence"
output_path = project_root / "processed_data" / current_file.parents[0].name

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]


def minmax_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Normaliza uma matriz por min-max.

    Se todos os valores forem iguais, retorna uma matriz de zeros
    para evitar divisão por zero.
    """

    matrix = matrix.astype(np.float32)

    min_value = matrix.min()
    max_value = matrix.max()

    if max_value == min_value:
        return np.zeros_like(matrix, dtype=np.float32)

    return (matrix - min_value) / (max_value - min_value)


def compute_lag_matrix(motif_sequences: np.ndarray, lag: int) -> np.ndarray:
    """
    Calcula a matriz de sincronização para um lag específico.

    Parâmetros
    ----------
    motif_sequences:
        Array com shape (n_channels, n_motifs).

    lag:
        Lag temporal usado na comparação.

    Retorna
    -------
    lag_matrix:
        Matriz com shape (n_channels, n_channels), onde cada posição
        [i, j] contém o número de coincidências entre o canal i e o canal j
        considerando o deslocamento temporal escolhido.
    """

    if motif_sequences.ndim != 2:
        raise ValueError("motif_sequences deve ter shape (n_channels, n_motifs).")

    n_channels, n_motifs = motif_sequences.shape

    if lag < 0:
        raise ValueError("lag deve ser >= 0.")

    if lag >= n_motifs:
        raise ValueError("lag deve ser menor que o número de motifs.")

    if lag == 0:
        seq1 = motif_sequences
        seq2 = motif_sequences
    else:
        seq1 = motif_sequences[:, :-lag]
        seq2 = motif_sequences[:, lag:]

    matches = seq1[:, None, :] == seq2[None, :, :]

    lag_matrix = matches.sum(axis=-1).astype(np.float32)

    return lag_matrix


def obtain_max_synchronization_matrices(
    array: np.ndarray,
    normalize: bool = True,
    zero_diagonal: bool = True
) -> np.ndarray:
    """
    Calcula as matrizes de máximo a partir das sequências de motifs.

    Parâmetros
    ----------
    array:
        Array vindo do motif_sequence.py com shape:
        (n_epochs, n_bands, n_channels, n_motifs).

    normalize:
        Se True, aplica normalização min-max em cada matriz final.

    zero_diagonal:
        Se True, zera a diagonal principal, removendo auto-conexões.

    Retorna
    -------
    max_matrices:
        Array com shape:
        (n_epochs, n_bands, 2, n_channels, n_channels).

        max_matrices[:, :, 0, :, :] -> Max(0, 1, 2, 3)
        max_matrices[:, :, 1, :, :] -> Max(1, 2, 3)
    """

    if array.ndim != 4:
        raise ValueError(
            "array deve ter 4 dimensões: "
            "(n_epochs, n_bands, n_channels, n_motifs)."
        )

    n_epochs, n_bands, n_channels, n_motifs = array.shape

    if n_motifs <= 3:
        raise ValueError("É necessário ter mais de 3 motifs para calcular lags até 3.")

    max_matrices = np.zeros(
        (n_epochs, n_bands, 2, n_channels, n_channels),
        dtype=np.float32
    )

    for epoch in range(n_epochs):
        for band in range(n_bands):

            motif_sequences = array[epoch, band]  # (n_channels, n_motifs)

            lag_0 = compute_lag_matrix(motif_sequences, lag=0)
            lag_1 = compute_lag_matrix(motif_sequences, lag=1)
            lag_2 = compute_lag_matrix(motif_sequences, lag=2)
            lag_3 = compute_lag_matrix(motif_sequences, lag=3)

            max_0_3 = np.maximum.reduce([lag_0, lag_1, lag_2, lag_3])
            max_1_3 = np.maximum.reduce([lag_1, lag_2, lag_3])

            if zero_diagonal:
                np.fill_diagonal(max_0_3, 0.0)
                np.fill_diagonal(max_1_3, 0.0)

            if normalize:
                max_0_3 = minmax_normalize_matrix(max_0_3)
                max_1_3 = minmax_normalize_matrix(max_1_3)

            max_matrices[epoch, band, 0] = max_0_3
            max_matrices[epoch, band, 1] = max_1_3

    return max_matrices


def run_max_synchronization_matrix():

    for subject in subjects:
        for session in sessions:

            file_path = os.path.join(
                base_path,
                f"{subject}_{session}_inner_bands_motifs.npy"
            )

            if not os.path.exists(file_path):
                print(f"Arquivo não encontrado: {file_path}")
                continue

            print(f"\nProcessando {current_file.parents[0].name}\n{subject} {session}...")

            array_bands_motifs = np.load(file_path)
            # Shape esperado: (épocas, bandas, canais, motifs)

            max_matrices = obtain_max_synchronization_matrices(
                array_bands_motifs,
                normalize=True,
                zero_diagonal=True
            )

            save_path = os.path.join(
                output_path,
                f"{subject}_{session}_max_sync_matrices.npy"
            )

            np.save(save_path, max_matrices)

            print(f"Salvo em: {save_path}")
            print(f"Shape: {max_matrices.shape}")
            print("Índice 0: Max(0, 1, 2, 3)")
            print("Índice 1: Max(1, 2, 3)")

    print(f"\nFinalizado {current_file.parents[0].name}.")


if __name__ == "__main__":

    run_max_synchronization_matrix()