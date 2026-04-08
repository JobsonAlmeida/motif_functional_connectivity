import os
import numpy as np

base_path = "filtered_bands_motifs"
output_path = "filtered_bands_matrices"

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]


def motif_sync_two_sequences(seq1: np.ndarray, seq2: np.ndarray, lag: int) -> float:
    """
    Calcula a sincronização entre duas sequências de motifs para um lag fixo.
    Retorna o número de motifs coincidentes.
    """
    if seq1.ndim != 1 or seq2.ndim != 1:
        raise ValueError("seq1 e seq2 devem ser vetores 1D.")

    if len(seq1) != len(seq2):
        raise ValueError("seq1 e seq2 devem ter o mesmo comprimento.")

    if lag < 0:
        raise ValueError("lag deve ser >= 0.")

    if lag >= len(seq1):
        raise ValueError("lag muito grande para o tamanho das sequências.")

    if lag == 0:
        s1 = seq1
        s2 = seq2
    else:
        s1 = seq1[:-lag]
        s2 = seq2[lag:]

    matches = (s1 == s2)

    return float(np.sum(matches))


def sync_matrices_for_all_lags(
    trial_band_motifs: np.ndarray,
    lags
) -> np.ndarray:
    """
    Calcula uma matriz de sincronização para cada lag,
    para um único trial e uma única banda.

    Entrada:
        trial_band_motifs.shape = (n_channels, n_motifs)

    Saída:
        sync_matrices_lags.shape = (n_lags, n_channels, n_channels)
    """
    if trial_band_motifs.ndim != 2:
        raise ValueError(
            "trial_band_motifs deve ter shape (n_channels, n_motifs)"
        )

    n_channels = trial_band_motifs.shape[0]
    n_lags = len(lags)

    sync_matrices_lags = np.empty(
        (n_lags, n_channels, n_channels),
        dtype=np.float32
    )

    for lag_idx, lag in enumerate(lags):
        sync_matrix = np.empty((n_channels, n_channels), dtype=np.float32)

        for i in range(n_channels):
            for j in range(n_channels):
                seq1 = trial_band_motifs[i]
                seq2 = trial_band_motifs[j]

                sync_matrix[i, j] = motif_sync_two_sequences(seq1, seq2, lag)

        sync_matrices_lags[lag_idx] = sync_matrix

    return sync_matrices_lags


def sync_matrices_for_trials_bands_lags(
    x_bands_motifs: np.ndarray,
    lags
) -> np.ndarray:
    """
    Calcula as matrizes de sincronização para todos os trials e bandas considerando os lags.

    Entrada:
        x_bands_motifs.shape = (n_trials, n_bands, n_channels, n_motifs)

    Saída:
        sync_matrices.shape = (n_trials, n_bands, n_lags, n_channels, n_channels)
    """
    if x_bands_motifs.ndim != 4:
        raise ValueError(
            "x_bands_motifs deve ter shape "
            "(n_trials, n_bands, n_channels, n_motifs)"
        )

    n_trials, n_bands, n_channels, _ = x_bands_motifs.shape
    n_lags = len(lags)

    sync_matrices = np.empty(
        (n_trials, n_bands, n_lags, n_channels, n_channels),
        dtype=np.float32
    )

    for trial in range(n_trials):
        for band in range(n_bands):
            trial_band_motifs = x_bands_motifs[trial, band, :, :]

            sync_matrices[trial, band] = sync_matrices_for_all_lags(
                trial_band_motifs,
                lags=lags
            )

    return sync_matrices


for subject in subjects:
    for session in sessions:

        file_path = os.path.join(
            base_path,
            f"{subject}_{session}_inner_bands_motifs.npy"
        )

        if not os.path.exists(file_path):
            print(f"Arquivo não encontrado: {file_path}")
            continue

        print(f"\nProcessando {subject} {session}...")

        x_bands_motifs = np.load(file_path)  # (épocas, bandas, canais, motifs)

        sync_matrices = sync_matrices_for_trials_bands_lags(
            x_bands_motifs,
            lags=(0, 1, 2, 3)
        )  # (trials, bandas, lags, canais, canais)

        save_path = os.path.join(
            output_path,
            f"{subject}_{session}_inner_bands_matrices.npy"
        )

        np.save(save_path, sync_matrices)

        print(f"Salvo em: {save_path}")
        print(f"Shape: {sync_matrices.shape}")