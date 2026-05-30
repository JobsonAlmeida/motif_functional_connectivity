import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

base_path = project_root / "processed_data" / "stage_50_maximum_matrix"

result_name = "per_subject_band_wise_effect"
output_path = project_root / "results" / current_file.parents[0].name / result_name

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]

default_band_labels = [
    "[12, 15]",
    "[15, 18]",
    "[18, 21]",
    "[21, 24]",
    "[24, 27]",
    "[27, 30]",
    "[30, 33]",
    "[33, 36]",
    "[36, 39]",
    "[39, 42]",
    "[42, 45]",
]


def get_maximum_matrix_types(data: np.ndarray):
    """
    Aceita dois possíveis formatos:

    1) Formato com lags separados:
       shape = (n_epochs, n_bands, 4, n_channels, n_channels)

       Nesse caso:
       - eixo 2 índice 0 = lag 0
       - eixo 2 índice 1 = lag 1
       - eixo 2 índice 2 = lag 2
       - eixo 2 índice 3 = lag 3

       Então o código calcula:
       - Max(0,1,2,3)
       - Max(1,2,3)

    2) Formato já processado pela stage 50:
       shape = (n_epochs, n_bands, 2, n_channels, n_channels)

       Nesse caso:
       - eixo 2 índice 0 = Max(0,1,2,3)
       - eixo 2 índice 1 = Max(1,2,3)
    """

    if data.ndim != 5:
        raise ValueError(
            f"A array deveria ter 5 dimensões, mas veio com shape {data.shape}"
        )

    if data.shape[2] == 4:
        # Caso em que ainda existem 4 lags separados: lag 0, lag 1, lag 2, lag 3
        max_0_1_2_3 = np.max(data[:, :, 0:4, :, :], axis=2)
        max_1_2_3 = np.max(data[:, :, 1:4, :, :], axis=2)

    elif data.shape[2] == 2:
        # Caso em que a stage 50 já gerou as duas matrizes de máximo
        max_0_1_2_3 = data[:, :, 0, :, :]
        max_1_2_3 = data[:, :, 1, :, :]

    else:
        raise ValueError(
            f"Terceira dimensão inesperada: {data.shape[2]}. "
            "Esperado 4 para lags separados ou 2 para matrizes de máximo."
        )

    return max_0_1_2_3, max_1_2_3


def mean_off_diagonal(matrix_data: np.ndarray) -> np.ndarray:
    """
    Recebe:
        matrix_data.shape = (n_epochs, n_bands, n_channels, n_channels)

    Retorna:
        offdiag_mean.shape = (n_epochs, n_bands)

    Para cada época e cada banda, calcula a média dos valores fora da diagonal.
    """

    n_channels = matrix_data.shape[-1]

    off_diagonal_mask = ~np.eye(n_channels, dtype=bool)

    offdiag_mean = matrix_data[:, :, off_diagonal_mask].mean(axis=2)

    return offdiag_mean


def sum_off_diagonal(matrix_data: np.ndarray) -> np.ndarray:
    """
    Versão alternativa: soma dos valores fora da diagonal.

    Use esta função no lugar de mean_off_diagonal se quiser reproduzir
    a descrição literal do artigo, que fala em somar os pesos off-diagonal.
    """

    n_channels = matrix_data.shape[-1]

    off_diagonal_mask = ~np.eye(n_channels, dtype=bool)

    offdiag_sum = matrix_data[:, :, off_diagonal_mask].sum(axis=2)

    return offdiag_sum


def compute_subject_delta(subject: str, use_mean: bool = True) -> np.ndarray:
    """
    Para um sujeito, carrega as sessões disponíveis e calcula:

        delta = mean(Max(0,1,2,3)) - mean(Max(1,2,3))

    O resultado final tem shape:

        delta.shape = (n_bands,)
    """

    subject_sessions = []

    for session in sessions:

        file_path = base_path / f"{subject}_{session}_maximum_matrices.npy"

        if not file_path.exists():
            print(f"Arquivo não encontrado: {file_path}")
            continue

        matrices = np.load(file_path)

        print(f"{subject} {session} | shape original: {matrices.shape}")

        subject_sessions.append(matrices)

    if len(subject_sessions) == 0:
        raise FileNotFoundError(
            f"Nenhum arquivo encontrado para {subject} em {base_path}"
        )

    subject_matrices = np.concatenate(subject_sessions, axis=0)

    print(f"{subject} | shape após juntar sessões: {subject_matrices.shape}")

    max_0_1_2_3, max_1_2_3 = get_maximum_matrix_types(subject_matrices)

    if use_mean:
        values_max_0_1_2_3 = mean_off_diagonal(max_0_1_2_3)
        values_max_1_2_3 = mean_off_diagonal(max_1_2_3)
    else:
        values_max_0_1_2_3 = sum_off_diagonal(max_0_1_2_3)
        values_max_1_2_3 = sum_off_diagonal(max_1_2_3)

    mean_max_0_1_2_3 = values_max_0_1_2_3.mean(axis=0)
    mean_max_1_2_3 = values_max_1_2_3.mean(axis=0)

    delta = mean_max_0_1_2_3 - mean_max_1_2_3

    return delta


def run_generate_per_subject_band_wise_effect():

    # True  -> usa média off-diagonal, escala mais interpretável
    # False -> usa soma off-diagonal, escala maior
    use_mean = True

    delta_matrix = []

    for subject in subjects:
        delta = compute_subject_delta(subject, use_mean=use_mean)
        delta_matrix.append(delta)

    delta_matrix = np.array(delta_matrix)

    n_bands = delta_matrix.shape[1]

    if n_bands <= len(default_band_labels):
        band_labels = default_band_labels[:n_bands]
    else:
        band_labels = [f"band {i + 1}" for i in range(n_bands)]

    subject_labels = [f"sub {i}" for i in range(1, len(subjects) + 1)]

    df = pd.DataFrame(
        delta_matrix,
        index=subject_labels,
        columns=band_labels
    )

    csv_path = output_path / "figure_5_delta_values.csv"
    df.to_csv(csv_path)

    print(f"\nCSV salvo em: {csv_path}")

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        delta_matrix,
        aspect="auto",
        cmap="viridis"
    )

    if use_mean:
        title_metric = "mean off-diagonal connectivity"
        colorbar_label = "Δ mean off-diagonal MS edge weight (a.u.)"
    else:
        title_metric = "summed off-diagonal connectivity"
        colorbar_label = "Δ summed off-diagonal MS edge weight (a.u.)"

    ax.set_title(
        f"Per-subject band-wise Δ {title_metric} (Max(0–3) − Max(1–3))\n"
        "Connectivity values between both approaches were computed from off-diagonal matrix weights",
        fontsize=11
    )

    ax.set_xlabel("Band (Hz)")
    ax.set_ylabel("Participant")

    ax.set_xticks(np.arange(n_bands))
    ax.set_xticklabels(band_labels, rotation=35, ha="right")

    ax.set_yticks(np.arange(len(subject_labels)))
    ax.set_yticklabels(subject_labels)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(colorbar_label)

    plt.tight_layout()

    save_path = output_path / "figure_5_per_subject_band_wise_effect.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"Figura salva em: {save_path}")


if __name__ == "__main__":
    run_generate_per_subject_band_wise_effect()