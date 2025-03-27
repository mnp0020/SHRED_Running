# SHRED_Running
This repository contains the code used for the research presented in Evaluating Sparse Inertial Measurement Unit Configurations for Inferring Treadmill Running Motion, available in Sensors

# Shallow Recurring Decoder Networks (SHRED) for Human Motion Inference

This repository contains the code used for the research presented in *Evaluating Sparse Inertial Measurement Unit Configurations for Inferring Treadmill Running Motion*, available in *Sensors*.

## Table of Contents
1. [Introduction](#introduction)
2. [Setup and Configuration](#setup-and-configuration)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [License and Contact](#license-and-contact)

## Introduction
The purpose of this code is to serve as an instructional pipeline to reconstruct biomechanics data using Shallow Recurrent Decoder Networks (SHRED). SHRED features an LSTM and a shallow decoder to reconstruct a dense set of signals from sparse sensor input. Example data containing IMU data is provided, and this pipeline trains and tests a subject-specific SHRED model to reconstruct the dense dataset from specified input signals. The primary audience is researchers interested in machine learning applications in biomechanics. For more information on SHRED and its applications, see the [original paper].

## Setup and Configuration
### Step 1: Create Directories/Folders
Manually set up folders to correspond to the following example paths. These can vary based on your dataset:
- **Notebooks Path:** `SHRED/Notebooks`
- **.mat File Path:** `SHRED/Datasets/Data/filename.mat` (location for subject#.mat files)
- **DataFrame Results Path:** `SHRED/Datasets/Dataframes/AC13/RightAnkle/3acc_Training`
- **Optional Save Paths:** `SHRED/Datasets/Figures` and `SHRED/Datasets/Models`

### Step 2: Download Necessary Files
- **Python Notebooks & .mat File:** Download `SHRED_Running_main.ipynb` and a suitable `.mat` file.
- **Supporting Scripts:** Download `functions.py`, `models.py`, and `processdata.py`.
- Store the `.ipynb` and `.py` files in the `SHRED/Notebooks` directory and the `.mat` file in the designated data path.

### Step 3: Install Required Packages
If running locally, ensure the following are installed:
- Python 3.8+
- Jupyter Notebook
- numpy
- pandas
- scipy
- pytorch (for SHRED model)
- altair (for visualization)

### Step 4: Update Code
- Open `SHRED_Running_main.ipynb` in an environment (e.g., VS Code, Google Colab).
- Update the code block *“Obtain subject’s experimental data”* with the subject number, trial length (1-6), and frequency (up to 128).
- Verify the dataset path and dataframe path in the block *“Import Matlab file structure”*.
- Customize settings under *“Set up sensors”* and *“Visualize IMU data”* as needed.

## Usage
To use your own subject data, replace the directory `.../Data/Subject#.mat` with your data and adjust paths in the code accordingly.
- **Example Input:** `Subject01.mat`
- **Example Output:** SHRED reconstruction of Left Ankle Acceleration vs. True Data, including RMSE calculation.

## Contributing
We welcome contributions:
- **Bug Fixes & Enhancements:** Submit a pull request for improvements.
- **Documentation:** Refine explanations, add examples, or correct typos.
- **Testing & Validation:** Share your findings with new datasets.

**To contribute:**
1. Fork the repository.
2. Create a new branch (`feature-name`).
3. Commit changes with descriptive messages.
4. Submit a pull request for review.

For major contributions or inquiries, contact the authors.

## License and Contact
**Contributors:**
- Mackenzie Pitts, MS ([mnp0020@uw.edu](mailto:mnp0020@uw.edu)) — Contact
- Megan Ebers, PhD ([mebers@uw.edu](mailto:mebers@uw.edu))
- David Green ([dgreen74@uw.edu](mailto:dgreen74@uw.edu))
- Jan Williams, MS — Original SHRED Development

*For license details, refer to the LICENSE file in the repository.*
