# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:05:55 2023

@author: Sahar
create mne raw object from array of data 
"""

import mne
import scipy.io
import numpy as np
from pathlib import Path


def mneObjectGenerator(data, fs):

    # Load the .mat file
    Path_to_chan = "Example Data/chanlocs.mat"
    mat_data = scipy.io.loadmat(Path_to_chan)
    channels = mat_data["chanlocs"]

    # Extract channel names
    ch_names = [ch[0] for ch in channels["labels"][0][:-3]]
    ch_types = ["eeg"] * 32

    # Extract x, y, and z coordinates
    x_coords = [coord[0][0] for coord in channels["X"][0][:-3]]
    y_coords = [coord[0][0] for coord in channels["Y"][0][:-3]]
    z_coords = [coord[0][0] for coord in channels["Z"][0][:-3]]

    # Rotate about the z-axis by 90 degrees counter-clockwise
    rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]])

    # Adjust the coordinates
    rotated_coords = np.dot(
        np.vstack((x_coords, y_coords, z_coords)).T, rotation_matrix.T
    )
    x_coords_rotated = rotated_coords[:, 0]
    y_coords_rotated = rotated_coords[:, 1]
    # z_coords remains the same

    scaling_factor = 0.1
    x_coords_rotated *= scaling_factor
    y_coords_rotated *= scaling_factor
    z_coords = [z * scaling_factor for z in z_coords]

    # Create a 3D montage from the rotated coordinates
    montage = mne.channels.make_dig_montage(
        ch_pos=dict(zip(ch_names, zip(x_coords_rotated, y_coords_rotated, z_coords))),
        coord_frame="head",
    )

    # Create an info object
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)

    raw = mne.io.RawArray(data, info)
    raw.set_montage(montage)

    return raw
