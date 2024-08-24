# EEG Motion Artifact Analysis

This project is dedicated to improving the quality of EEG recordings by identifying and mitigating motion artifacts. These artifacts, often caused by subject movement or environmental factors, can distort EEG data and lead to inaccurate interpretations. Our goal is to develop robust algorithms that can detect and reduce the impact of these artifacts, enhancing the reliability of EEG analysis.

## Table of Contents
- [Introduction](#introduction)
- [Running the Example](#running-the-example)
- [Results from the Figs Directory](#results-from-the-figs-directory)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Motion artifacts in EEG recordings, whether from subject movement, electrode displacement, or environmental interference, introduce noise that can obscure meaningful brain activity. This project focuses on developing advanced methods for detecting and mitigating these artifacts, thereby improving the clarity and usability of EEG data.

## Running the Example

To run the example code, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone git@github.com:SaharSattari/ged_eeg_analysis.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd ged_eeg_analysis
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Execute the example script:**

    ```bash
    python3 example/ged_analysis_simulated_data.py
    ```

    The script will run the analysis and display the results in your console.

## Results from the Figs Directory
Below are some key figures illustrating the example results from simulated data:

- **Simulated Data:**
  ![Simulated data](figs/Artifactual.png)

- **Post-GED Analysis:**
  ![Post-GED simulated data](figs/PostGED.png)

- **GED Components:**
  ![GED Components](figs/GED_Components.png)

## Reference
[A tutorial on generalized eigendecomposition for denoising, contrast enhancement, and dimension reduction in multichannel electrophysiology](https://www.sciencedirect.com/science/article/pii/S1053811921010806)

[Exploring patterns enriched in a dataset with contrastive principal component analysis](https://www.nature.com/articles/s41467-018-04608-8)

## License
This project is licensed under the [MIT License](LICENSE). For more details, refer to the LICENSE file.