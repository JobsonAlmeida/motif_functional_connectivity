import os
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view


current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

base_path = project_root / "processed_data" / "stage_20_band_filtering"
output_path = project_root / "processed_data" / current_file.parents[0].name

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]

def obtain_motif_sequences(x_bands: np.ndarray) -> np.ndarray:

    if x_bands.ndim != 4:
        raise ValueError("x_bands deve ter 4 dimensões")

    windows = sliding_window_view(x_bands, window_shape=3, axis=-1) #windows.shape = (n_trials, n_bands, n_channels, n_motifs (n_times-2), 3)

    orders = np.argsort(windows, axis=-1, kind="mergesort") #orders.shape = (n_trials, n_bands, n_channels, n_motifs, 3)

    x_bands_motifs = np.full(orders.shape[:-1], -1, dtype=np.int32) #orders.shape = (n_trials, n_bands, n_channels, n_motifs)

    mapping = {
        (0, 1, 2): 0,
        (0, 2, 1): 1,
        (1, 0, 2): 2,
        (1, 2, 0): 3,
        (2, 0, 1): 4,
        (2, 1, 0): 5,
    }

    for order_tuple, motif_id in mapping.items():
        mask = np.all(orders == order_tuple, axis=-1)
        x_bands_motifs[mask] = motif_id

    all_equal = (
        (windows[..., 0] == windows[..., 1]) & #windows[:, :, :, :, 0] == windows[:, :, :, :, 1]
        (windows[..., 0] == windows[..., 2])
    )

    x_bands_motifs[all_equal] = 6

    return x_bands_motifs


for subject in subjects:
    for session in sessions:

        """___Abrindo o aquivo a ser processado___"""

        file_path = os.path.join(
            base_path,
            f"{subject}_{session}_inner_bands.npy"
        )

        if not os.path.exists(file_path):
            print(f"Arquivo não encontrado: {file_path}")
            continue

        print(f"\nProcessando {subject} {session}...")

        X_bands = np.load(file_path) #(épocas × bandas × canais × tempo)

        """___Processando os dados através___"""
       
        X_bands_motifs = obtain_motif_sequences(X_bands)

        """___Salvando os dados processados__"""

        save_path = os.path.join(
            output_path,
            f"{subject}_{session}_inner_bands_motifs.npy"
        )

        np.save(save_path, X_bands_motifs)

        print(f"Salvo em: {save_path}")
        print(f"Shape: {X_bands_motifs.shape}")
