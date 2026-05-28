from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import mne

from pipeline.config.channel_mapping import INDICES_CHANNELS_MAPPING

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

experiment_name = "experiment_1"

base_path = (
    project_root
    / "results"
    / current_file.parents[0].name
    / "run_stage_90_in_loop"
    / experiment_name
)

output_path = (
    project_root
    / "results"
    / current_file.parents[0].name
    / "generate_topographic_plot"
    / experiment_name
)

output_path.mkdir(parents=True, exist_ok=True)


subjects = [f"sub-{i:02d}" for i in range(1, 11)]

k_features_list = [20, 40, 60, 80, 100]

max_type_name = "max_1_2_3"   # lags = [1, 2, 3]


def compute_electrode_frequency(k_features):
    """
    Conta a frequência relativa com que cada eletrodo aparece
    entre as top-k features selecionadas nos folds de CV.
    """

    n_channels = len(INDICES_CHANNELS_MAPPING)
    electrode_counts = np.zeros(n_channels)

    total_selected_electrodes = 0

    for subject in subjects:

        file_path = base_path / f"{subject}_maximum_matrices_svm_results_k_{k_features}.pkl"

        if not file_path.exists():
            print(f"Arquivo não encontrado: {file_path}")
            continue

        with open(file_path, "rb") as f:
            results = pickle.load(f)

        selected_features = results[subject][max_type_name]["selected_features"]

        for fold_features in selected_features:
            for feature in fold_features:
                channel_idx = feature["channel_index"]

                electrode_counts[channel_idx] += 1
                total_selected_electrodes += 1

    electrode_frequency = electrode_counts / total_selected_electrodes

    return electrode_frequency


def create_info():
    """
    Cria um objeto MNE Info com os nomes dos canais.
    """

    channel_names = [
        INDICES_CHANNELS_MAPPING[i]
        for i in range(len(INDICES_CHANNELS_MAPPING))
    ]

    info = mne.create_info(# O MNE passa a saber que você temos 128 canais EEG amostrados a 256 Hz
        ch_names=channel_names,
        sfreq=256,
        ch_types="eeg"
    )

    montage = mne.channels.make_standard_montage("biosemi128")
    info.set_montage(montage, on_missing="ignore")

    return info


def plot_topographic():

    info = create_info()

    fig, axes = plt.subplots(
        1,
        len(k_features_list),
        figsize=(18, 4)
    )

    all_frequencies = []

    for k in k_features_list:
        freq = compute_electrode_frequency(k)
        all_frequencies.append(freq)

    all_frequencies = np.array(all_frequencies)

    vmin = 0
    vmax = all_frequencies.max()

    for ax, k, freq in zip(axes, k_features_list, all_frequencies):

        im, _ = mne.viz.plot_topomap(
            freq,
            info,
            axes=ax,
            show=False,
            contours=6,
            cmap="Reds",
            vlim=(vmin, vmax),
            sensors=True
        )

        ax.set_title(f"top-{k}", fontsize=10)

    cbar = fig.colorbar(
        im,
        ax=axes,
        shrink=0.75,
        pad=0.02
    )

    cbar.set_label("Relative electrode frequency")

    fig.suptitle(
        "Electrode selection frequency across top-K CV-selected features\n"
        "Delayed motif synchronization (lags = [1, 2, 3])",
        fontsize=12
    )

    plt.tight_layout()

    save_path = output_path / "figure_8_electrode_selection_frequency.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figura salva em: {save_path}")


if __name__ == "__main__":
    plot_topographic()