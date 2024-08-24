#Import the necessary libraries
#%%
import numpy as np
from scipy.signal import butter, filtfilt
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from sklearn.decomposition import PCA
import sys
path = "../src"
if path not in sys.path:
    sys.path.append(path)
from ged_eeg_analysis.raw_generator import mneObjectGenerator


#Load the data
folder_path = "../data"

Ground_Truth = np.load(f'{folder_path}/example_GT.npy')
Artifactual_EO = np.load(f'{folder_path}/example_Artifactual_EO.npy')
imu_data = np.load(f'{folder_path}/example_motion.npy')
calibration_data = np.load(f'{folder_path}/example_calib.npy')
channel_location_path = folder_path + "/chanlocs.mat"

#Step1: EMG for muscle artifact
def generate_emg(Sampling, SignalLength, FreqRange):
    Sigma = 250
    b, a = butter(4, np.array(FreqRange) / (0.5 * Sampling), btype="bandpass")
    noise = np.random.randn(SignalLength) * Sigma
    noise = filtfilt(b, a, noise)
    return noise

# Step2: eyeblink artifact

def generate_eyeblink(raw, sfreq):

    ica_obj = mne.preprocessing.ICA(
        n_components=0.99,
        method="infomax",
        max_iter="auto",
        random_state=1,
        fit_params=dict(extended=True),
    ).fit(raw)

    ic_labels = label_components(raw, ica_obj, method="iclabel")
    labels = ic_labels["labels"]

    eyeblink_idx = np.where(np.array(labels) == "eye blink")[0]

    ica_sources = ica_obj.get_sources(raw)
    eyeblink_signal = np.sum(ica_sources.get_data(picks=eyeblink_idx), axis=0)

    return eyeblink_signal


# Step3: Motion artifact
def generate_motion(imudata):

    pca = PCA(n_components=3)
    imu_pca = pca.fit_transform(imudata.T).T
    return imu_pca[0, :]


# Generate artifactual data
# find a random number for channels to add artifact (more than 5)
nch = np.random.randint(5, Ground_Truth.shape[0])
chosen_channels = np.random.choice(Ground_Truth.shape[0], nch, replace=False)
coefficients = np.random.uniform(0.1, 0.5, nch)

sfreq = 500
Artifactual_sim = np.copy(Ground_Truth)
noise = generate_emg(sfreq, Ground_Truth.shape[1], FreqRange=[20, 80])

for i, channel in enumerate(chosen_channels):
    artifact = coefficients[i] * noise
    Artifactual_sim[channel, :] += artifact

mneObjectGenerator(Artifactual_sim / 1e6, sfreq, channel_location_path).plot(title="EMG added")

nch = np.random.randint(5, Ground_Truth.shape[0])
chosen_channels = np.random.choice(Ground_Truth.shape[0], nch, replace=False)
coefficients = np.random.uniform(0.1, 0.5, nch)

noise = generate_eyeblink(
    mneObjectGenerator(Artifactual_EO, sfreq, channel_location_path), sfreq
)
for i, channel in enumerate(chosen_channels):
    artifact = coefficients[i] * noise
    Artifactual_sim[channel, :] += artifact
mneObjectGenerator(Artifactual_sim / 1e6, sfreq, channel_location_path).plot(title="EMG added, Eye blink added")



nch = np.random.randint(5, Ground_Truth.shape[0])
chosen_channels = np.random.choice(Ground_Truth.shape[0], nch, replace=False)
coefficients = np.random.uniform(0.1, 0.5, nch)

for i, channel in enumerate(chosen_channels):
    sfreq = 500
    num_samples = 90000
    noise = generate_motion(imu_data)
    artifact = coefficients[i] * noise
    Artifactual_sim[channel, :] += artifact
mneObjectGenerator(Artifactual_sim / 1e6, 500, channel_location_path).plot(title="EMG added, Eye blink added, motion added", scalings = {'eeg': 6e-5})


#GED
from ged_eeg_analysis.contrast_cleaning import ArtifactReconstruct

artifact_reconstruct = ArtifactReconstruct(
    use_covariance=True, type="GED", normalize=False, vis_results= True, verbose=False
)


alpha = 1
results, Maps, clean = artifact_reconstruct.perform_analysis(
    Artifactual_sim, calibration_data, alpha
)
print(clean.shape)
mneObjectGenerator(clean / 1e6, 500, channel_location_path).plot(title="Cleaned data", scalings = {'eeg': 6e-5})


# %%
