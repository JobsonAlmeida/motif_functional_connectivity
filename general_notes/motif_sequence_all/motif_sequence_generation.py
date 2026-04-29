import os
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view


# Montando o caminho baseado no local do arquivo .py, e não a partir do diretório de execução do terminal do python
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]

base_path = project_root / "processed_data" / "mod_1_band_filtering"
output_path = project_root / "processed_data" / "mod_2_motif_sequence_generation"

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]

MOTIF_MAP_3 = {
    (0, 1, 2): 0,
    (0, 2, 1): 1,
    (1, 0, 2): 2,
    (1, 2, 0): 3,
    (2, 0, 1): 4,
    (2, 1, 0): 5,
    (0, 0, 0): 6,

}

def obtain_motif_sequence(signal: np.ndarray, motif_length: int = 3) -> np.ndarray:
    """
    Converte um sinal 1D em sequência de motifs ordinais.

    Parâmetros
    ----------
    signal : np.ndarray
        Vetor 1D com o sinal temporal.
    motif_length : int
        Tamanho do motif. Aqui vamos usar 3.

    Retorno
    -------
    np.ndarray
        Sequência de motifs codificados como inteiros.
    """
    if motif_length != 3:
        raise NotImplementedError("Por enquanto esta função implementa apenas motifs de tamanho 3.")

    signal = np.asarray(signal) # garante que o array seja array do numpy
 
    if signal.ndim != 1:
        raise ValueError("O sinal deve ser 1D.")

    n = len(signal)
    if n < motif_length:
        raise ValueError("O sinal é menor que o tamanho do motif.")

    motifs_sequence = []

    for i in range(n - motif_length + 1):
        window = signal[i:i + motif_length]

        if window[0] == window[1] and window[0] == window[2]:
            order = (0, 0, 0)
        else:
            order = tuple(np.argsort(window, kind="mergesort"))

        motif_id = MOTIF_MAP_3[order]
        motifs_sequence.append(motif_id)

    return np.array(motifs_sequence, dtype=np.int32)

def motif_sequences_all(x_bands: np.ndarray) -> np.ndarray:
    """
    Converte sinais filtrados em sequências de motifs.

    Entrada:
        x_bands.shape = (n_trials, n_bands, n_channels, n_times)

    Saída:
        x_bands_motifs.shape = (n_trials, n_bands, n_channels, n_motifs)
    """

    if x_bands.ndim != 4:
        raise ValueError("x_bands deve ter 4 dimensões")
    
    n_trials, n_bands, n_channels, n_times = x_bands.shape
    n_motifs = n_times - 2  # motif tamanho 3

    # pré-alocando o array de 4d (importante para performance)
    x_bands_motifs = np.empty(
        (n_trials, n_bands, n_channels, n_motifs),
        dtype=np.int32
    )

    for trial in range(n_trials):
        for band in range(n_bands):
            for channel in range(n_channels):

                signal = x_bands[trial, band, channel, :]

                motif_seq = obtain_motif_sequence(signal)

                x_bands_motifs[trial, band, channel, :] = motif_seq

    return x_bands_motifs


def motif_sequences_all_fast(x_bands: np.ndarray) -> np.ndarray:

    if x_bands.ndim != 4:
        raise ValueError("x_bands deve ter 4 dimensões")

    windows = sliding_window_view(x_bands, window_shape=3, axis=-1) #windows.shape = (n_trials, n_bands, n_channels, n_times-2, 3)

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

        """Abrindo o aquivo a ser processado"""
        file_path = os.path.join(
            base_path,
            f"{subject}_{session}_inner_bands.npy"
        )

        if not os.path.exists(file_path):
            print(f"Arquivo não encontrado: {file_path}")
            continue

        print(f"\nProcessando {subject} {session}...")

        X_bands = np.load(file_path) #(épocas × bandas × canais × tempo)

        """Processando os dados através """
        # X_bands_motifs = motif_sequences_all(X_bands)


        import time

        inicio = time.time()
        X_old = motif_sequences_all(X_bands)
        fim = time.time()
        print(f"Tempo de execução motif_sequences_all: {fim - inicio:.4f} segundos")


        inicio = time.time()
        X_fast = motif_sequences_all_fast(X_bands)
        fim = time.time()
        print(f"Tempo de execução motif_sequences_all_fast: {fim - inicio:.4f} segundos")


        print(np.array_equal(X_old, X_fast))



        """Salvando os dados processados"""
        save_path = os.path.join(
            output_path,
            f"{subject}_{session}_inner_bands_motifs.npy"
        )

        #np.save(save_path, X_bands_motifs)
        np.save(save_path, X_fast)


        print(f"Salvo em: {save_path}")
        #print(f"Shape: {X_bands_motifs.shape}")
        print(f"Shape: {X_fast.shape}")



