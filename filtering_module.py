import os
import mne
import numpy as np
import pickle

base_path = "dataset_Thinking_outloud/derivatives"
output_path = "filtered_bands"

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]

bands = [
    (12, 15),
    (15, 18),
    (18, 21),
    (21, 24),
    (24, 27),
    (27, 30),
    (30, 33),
    (33, 36),
    (36, 39),
    (39, 42),
    (42, 45),
]

for subject in subjects:
    for session in sessions:

        file_path = os.path.join(
            base_path,
            subject,
            session,
            f"{subject}_{session}_eeg-epo.fif"
        )

        events_path = os.path.join(
            base_path,
            subject,
            session,
            f"{subject}_{session}_events.dat"
        )

        if not os.path.exists(file_path):
            print(f"Arquivo não encontrado: {file_path}")
            continue

        if not os.path.exists(events_path):
            print(f"Arquivo não encontrado: {events_path}")
            continue

        print(f"\nProcessando {subject} {session}...")
        epochs = mne.read_epochs(file_path, preload=True, verbose=False)

        with open(events_path, "rb") as f:
            events = pickle.load(f)

        mask_inner = events[:, 2] == 1
        epochs_inner = epochs[mask_inner]

        band_arrays = []

        for l_freq, h_freq in bands:
            epochs_band = epochs_inner.copy().filter(
                l_freq=l_freq,
                h_freq=h_freq,
                picks="eeg",
                method="fir",
                phase="zero",
                fir_design="firwin",
                verbose=False
            )

            band_arrays.append(epochs_band.get_data())  # [ banda1, banda2, banda3 ] com cada banda no formato (trials × canais × tempo)

        X_bands = np.stack(band_arrays, axis=1) #(época × bandas × canais × tempo)

        save_path = os.path.join(
            output_path,
            f"{subject}_{session}_inner_bands.npy"
        )

        np.save(save_path, X_bands)

        print(f"Salvo em: {save_path}")
        print(f"Shape: {X_bands.shape}")