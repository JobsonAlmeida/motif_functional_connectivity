from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne

from pipeline.config.channel_mapping import INDICES_CHANNELS_MAPPING

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

subjects = [f"sub-{i:02d}" for i in range(1, 11)]

max_type_name = "max_1_2_3"   # lags = [1, 2, 3]


def compute_electrode_frequency(k_features, base_path):

    n_bands = 11
    n_channels = len(INDICES_CHANNELS_MAPPING)
    n_measures = 2

    electrode_counts = np.zeros(n_channels, dtype=float)
    total_selected_electrodes = 0

    for subject in subjects:

        file_path = base_path / f"{subject}_ranked_features_svm_results.pkl"

        if not file_path.exists():
            print(f"Arquivo não encontrado: {file_path}")
            continue

        with open(file_path, "rb") as f:
            results = pickle.load(f)

        ranked_feature_indices = results[subject][max_type_name]["ranked_feature_indices"]
        topk_indices = ranked_feature_indices[:k_features]

        for feature_idx in topk_indices:

            band_idx, channel_idx, measure_idx = np.unravel_index(
                feature_idx,
                (n_bands, n_channels, n_measures)
            )

            electrode_counts[channel_idx] += 1
            total_selected_electrodes += 1

    if total_selected_electrodes == 0:
        raise ValueError("Nenhuma feature foi encontrada para calcular a frequência.")

    return electrode_counts / total_selected_electrodes


def create_info():

    channel_names = [
        INDICES_CHANNELS_MAPPING[i]
        for i in range(len(INDICES_CHANNELS_MAPPING))
    ]

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=256,
        ch_types="eeg"
    )

    montage = mne.channels.make_standard_montage("biosemi128")
    info.set_montage(montage, on_missing="ignore")

    return info


def plot_topographic_original_flow(
        k_features=[20, 40, 60, 80, 100]
):

    base_path = (
        project_root
        / "processed_data"
        / "stage_90_maximum_matrix_based_SVM"
    )

    output_path = (
        project_root
        / "results"
        / current_file.parents[0].name
        / "generate_topographic_plot"
    )

    output_path.mkdir(parents=True, exist_ok=True)

    k_features_list = list(k_features)

    info = create_info()

    all_frequencies = []

    for k in k_features_list:
        freq = compute_electrode_frequency(k, base_path)
        all_frequencies.append(freq)

    all_frequencies = np.array(all_frequencies)

    vmin = 0
    vmax = all_frequencies.max()

    plt.close("all")

    fig = plt.figure(figsize=(22, 4))

    # Região usada apenas pelos topomaps
    gs = gridspec.GridSpec(
        1,
        len(k_features_list),
        left=0.03,
        right=0.84,     # os topomaps terminam aqui
        bottom=0.10,
        top=0.78,
        wspace=0.35
    )

    axes = [
        fig.add_subplot(gs[0, i])
        for i in range(len(k_features_list))
    ]

    # Eixo exclusivo da colorbar
    # [left, bottom, width, height]
    cax = fig.add_axes([0.87, 0.22, 0.012, 0.50])

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

        ax.set_title(f"Top-{k}", fontsize=10)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Relative electrode frequency")

    fig.suptitle(
        "Electrode selection frequency across top-K ranked features\n"
        "Delayed motif synchronization (lags = [1, 2, 3])",
        fontsize=12,
        y=0.95
    )

    save_path = output_path / "figure_8_electrode_selection_frequency.png"

    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Figura salva em: {save_path}")


if __name__ == "__main__":
    plot_topographic_original_flow()