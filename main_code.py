import mne

file_path = "dataset_Thinking_outloud/derivatives/sub-01/ses-01/sub-01_ses-01_eeg-epo.fif" 

import os

print(os.path.exists(file_path))

# # carregar epochs
# epochs = mne.read_epochs(file_path, preload=True)

# print(epochs)