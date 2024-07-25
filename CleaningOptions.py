# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:59:42 2024

@author: Sahar

cPCA for noise based on correlation or covariance

relative correlation analysis for noise 


"""
import pandas as pd
import numpy as np
from scipy import linalg
from EEGfilters import highpass_filter, lowpass_filter
import matplotlib.pyplot as plt

def perform_analysis(input_fg, input_bg, alpha=1, use_covariance=True, use_correlation=True, normalize=True):
    """
    Perform contrastive PCA and relative correlation analysis on input data.

    Parameters:
    input_fg: DataFrame or array-like
        Foreground data.
    input_bg: DataFrame or array-like
        Background data.
    alpha: float, optional
        Weighting factor for background data subtraction (default is 1).
    use_covariance: bool, optional
        If True, perform analysis using covariance matrices (default is True).
    use_correlation: bool, optional
        If True, perform analysis using correlation matrices (default is True).
    normalize: bool, optional
        If True, normalize the data (default is True).

    Returns:
    dict
        A dictionary containing sorted eigenvectors for each analysis type.
    """

    def check_dimensions(data):
        return data if data.shape[0] < data.shape[1] else data.T

    def center_and_normalize(data):
        centered = data - np.mean(data, axis=1, keepdims=True)
        return centered / np.std(centered, axis=1, keepdims=True) if normalize else centered

    # Check dimensions
    fg = check_dimensions(input_fg)
    bg = check_dimensions(input_bg)

    # Center and optionally normalize data
    fg_centered = center_and_normalize(fg)
    bg_centered = center_and_normalize(bg)

    results = {}

    # Covariance/Correlation matrices
    if use_covariance or use_correlation:
        fg_centered = pd.DataFrame(fg_centered.T)
        bg_centered = pd.DataFrame(bg_centered.T)
        cov_fg, cov_bg = fg_centered.cov(), bg_centered.cov() 
        cor_fg, cor_bg = fg_centered.corr(), bg_centered.corr()

        # Contrastive PCA using covariance matrix
        if use_covariance:
            sigma_cov = cov_fg - alpha * cov_bg
            w_cov, v_cov = linalg.eig(sigma_cov)
            sorted_indices_cov = np.argsort(w_cov)[::-1]
            results['covariance'] = v_cov[:, sorted_indices_cov]

        # Contrastive PCA using correlation matrix
        if use_correlation:
            sigma_cor = cor_fg - alpha * cor_bg
            w_cor, v_cor = linalg.eig(sigma_cor)
            sorted_indices_cor = np.argsort(w_cor)[::-1]
            results['correlation'] = v_cor[:, sorted_indices_cor]

    # Relative correlation analysis
    try:
        if use_covariance:
        
            #lambda_ = 1e-15  # Regularization parameter
            gamma = 0.01
            
            # Regularization of covariance matrices
            cov_fg_regularized = cov_fg
            
            # eigenvalues, _ = np.linalg.eig(cov_fg)
            # cov_fg_regularized = cov_fg * (1-gamma) + gamma * np.mean(eigenvalues) * np.eye(cov_fg.shape[0])
            
            
            
            eigenvalues, _ = np.linalg.eig(cov_bg)
            cov_bg_regularized = cov_bg * (1-gamma) + gamma * np.mean(eigenvalues) * np.eye(cov_bg.shape[0])
            
            
            cov_bg_inv = np.linalg.inv(cov_bg_regularized)
            
            sigma_rel_cov = cov_bg_inv.dot(cov_fg_regularized)
            
            # cov_bg_inv = np.linalg.inv(cov_bg)
            
            # sigma_rel_cov = cov_bg_inv.dot(cov_fg)
            
            
            
            w_rel_cov, v_rel_cov = linalg.eig(sigma_rel_cov)
            sorted_indices_rel_cov = np.argsort(w_rel_cov)[::-1]
            
            
            sidx  = np.argsort(w_rel_cov)[::-1]
            evals = w_rel_cov[sidx]
            # tolerance = 1e-10
            # if np.all(np.abs(np.imag(w_rel_cov)) < tolerance):
            #     w_rel_cov = np.real(w_rel_cov)
            
            # if np.all(np.abs(np.imag(v_rel_cov)) < tolerance):
            #     v_rel_cov = np.real(v_rel_cov)
            results['relative_covariance'] = v_rel_cov[:, sorted_indices_rel_cov]
            
            # Normalize each column vector to have an amplitude between 0 and 1
            for i in range(results['relative_covariance'].shape[1]):
                norm = np.linalg.norm(results['relative_covariance'][:, i])
                if norm != 0:
                    results['relative_covariance'][:, i] /= norm
            
            
            #eeg_tensor_FW_filt[subj, Iter] = lowpass_filter(highpass_filter(eeg_tensor_FW[subj, Iter], 1, fs, notch_freq=60), 50, fs)
            #filtered = pd.DataFrame(lowpass_filter(highpass_filter(fg_centered, 12, 500), 50, 500).T)
            #cov_fg2= filtered.T.cov()
            Maps = np.dot( results['relative_covariance'].T, cov_fg)
            
            plt.figure()
            plt.plot(evals/np.max(evals),'s-',markersize=15,markerfacecolor='k')
            
            
        if use_correlation:
            cor_bg_inv = np.linalg.inv(cor_bg)
            sigma_rel_cor = cor_bg_inv.dot(cor_fg)
            w_rel_cor, v_rel_cor = linalg.eig(sigma_rel_cor)
            sorted_indices_rel_cor = np.argsort(w_rel_cor)[::-1]
            results['relative_correlation'] = v_rel_cor[:, sorted_indices_rel_cor]

    except np.linalg.LinAlgError:
        print("Matrix is singular and cannot be inverted.")

    return results, Maps












