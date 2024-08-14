import pandas as pd
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


class ArtifactReconstruct:

    def __init__(
        self,
        use_covariance: bool = True,
        type: str = "GED",
        normalize: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            use_covariance (bool, optional): Defaults to True.
            type (str, optional): GED for Generalized eigen decomposition (Cohen 2022) or cpca for Contrastive PCA (Abid et al, 2018)
            normalize (bool, optional): Center the data if False or normalize if True. Defaults to False.
        """

        self.use_covariance = use_covariance
        self.type = type
        self.normalize = normalize
        self.verbose = verbose

    def _check_dimensions(self, data: np.ndarray):
        return data if data.shape[0] < data.shape[1] else data.T

    def _center_and_normalize(self, data: np.ndarray):
        centered = data - np.mean(data, axis=1, keepdims=True)
        return (
            centered / np.std(centered, axis=1, keepdims=True)
            if self.normalize
            else centered
        )

    def _covariance_analysis(self, fg: np.ndarray, bg: np.ndarray, alpha: int):

        cov_fg, cov_bg = np.cov(fg, rowvar=True), np.cov(bg, rowvar=True)
        sigma_cov = cov_fg - alpha * cov_bg
        w_cov, v_cov = linalg.eig(sigma_cov)
        sorted_indices_cov = np.argsort(w_cov)[::-1]
        return v_cov[:, sorted_indices_cov]

    def _relative_covariance_analysis(self, fg: np.ndarray, bg: np.ndarray):
        cov_fg, cov_bg = np.cov(fg, rowvar=True), np.cov(bg, rowvar=True)
        gamma = 0.01
        eigenvalues, _ = np.linalg.eig(cov_bg)
        cov_bg_reg = cov_bg * (1 - gamma) + gamma * np.mean(eigenvalues) * np.eye(
            cov_bg.shape[0]
        )

        try:
            evals, evecs = linalg.eig(cov_fg, cov_bg_reg)
            sidx = np.argsort(evals)[::-1]
            evals = evals[sidx]
            evecs = evecs[:, sidx]

        except np.linalg.LinAlgError:
            print("Matrix is singular and cannot be inverted.")

        # # plot the eigenspectrum
        # plt.figure()
        # plt.plot(evals / np.max(evals), "s-", markersize=15, markerfacecolor="k")
        # plt.xlim([-0.5, 20.5])
        # plt.title("GED eigenvalues")
        # plt.xlabel("Component number")
        # plt.ylabel("Power ratio (norm-$\lambda$)")

        # # Suppress y-axis tick labels but keep the y-axis label
        # plt.gca().yaxis.set_ticks([])
        # plt.gca().xaxis.set_ticks([])

        # plt.show()

        return evecs, evals

    def _generate_maps(self, evecs: np.ndarray, fg: np.ndarray):

        covS = np.cov(fg)
        if evecs.ndim != 2:
            raise ValueError("evecs should be a 2D array")

        Maps = np.zeros_like(evecs)

        for counter in range(evecs.shape[1]):
            Maps[:, counter] = -1 * evecs[:, counter].T @ covS

        return Maps

    def perform_analysis(self, input_fg: np.ndarray, input_bg: np.ndarray, alpha: int):
        fg = self._check_dimensions(input_fg)
        bg = self._check_dimensions(input_bg)
        fg_centered = self._center_and_normalize(fg)
        bg_centered = self._center_and_normalize(bg)

        results = {}
        Maps = None

        if self.type == "cpca":
            results["contrastive pca"] = self._covariance_analysis(
                fg_centered, bg_centered, alpha
            )
            Maps = self.generate_maps(
                results["cpca"], fg_centered
            )  # TODO should it be fg_centered?

        if self.type == "GED":

            results["ged"] = self._relative_covariance_analysis(
                fg_centered, bg_centered
            )
            evecs_f, _ = results["ged"]

            Maps = self._generate_maps(np.array(evecs_f), fg_centered)

        return results, Maps
