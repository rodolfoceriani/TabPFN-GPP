# SCOPE GPP Model Inference and Training Dashboard

This repository provides code and pre-trained models to perform GPP (Gross Primary Productivity) modeling and wavelength importance analysis using a unified **TabPFN** and PLS (Partial Least Squares) regression pipeline. 

This project is built and distributed under the terms of the TabPFN License.

***

## 🚀 Built with PriorLabs-TabPFN
This project utilizes the **TabPFN** model weights and inference library from Prior Labs. For licensing compliance (Section 10 of the Prior Labs License), all derivative works, interfaces, and documentations prominently attribute Prior Labs.

***

## 📁 Repository Structure

```
├── .gitattributes          # LFS tracking configuration for large model files
├── .gitignore              # Standard Python and output Git exclusions
├── LICENSE                 # Copy of the Prior Labs License (Apache 2.0 + Attribution Provision)
├── README.md               # This documentation file
├── requirements.txt        # Python dependency list
│
├── Inference/
│   ├── gpp_gui.py           # PyQt6 Desktop GUI dashboard for running model inference
│   ├── TabPFN_GPP.joblib    # [LFS] Pre-trained unified TabPFN-GPP model weights (122.6 MB)
│   └── example_inference.csv# Sample dataset for validation and inference runs
│
└── SCOPE/
    ├── PLS_TabPFN.py              # Script to train TabPFN with PLS features
    ├── wavelenght_importance.py   # Script to evaluate TabPFN wavelength importance via permutation
    ├── plot_importance.py         # Visualizes relative importance of spectral bands
    └── dataset_GPP_Ta_Rin_ea_reflectance_EMIT.csv  # Full model training dataset
```

---

## 🛠️ Installation & Setup

Before running the scripts or dashboard, ensure you set up your Python environment and install the required dependencies.

### 1. Activating your Virtual Environment
If you are using a virtual environment (e.g. at `/home/musashi/venv` or locally), activate it first:

```bash
# On Linux / macOS:
source /home/musashi/venv/bin/activate
```

### 2. Installing Dependencies
Install all package requirements using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Git LFS (Large File Storage) for Model Weights
The pre-trained model file `Inference/TabPFN_GPP.joblib` is **122.6 MB**, which exceeds GitHub's standard 100 MB file limit. To clone or push this repository without losing the model weights, make sure Git LFS is installed and initialized on your system:

```bash
# Install Git LFS on your OS (Debian/Ubuntu example):
sudo apt-get install git-lfs

# Initialize Git LFS inside the repository directory:
git lfs install
```

When you push or pull, Git LFS will automatically track and manage `*.joblib` files based on the `.gitattributes` configuration.

---

## 💻 How to Run

### A. GPP Inference GUI Dashboard
The GUI dashboard (`gpp_gui.py`) is designed to run out-of-the-box. It provides a premium PyQt6 desktop application to load datasets, select models, run predictions, save tabular results, and view a live scatterplot preview.

To launch the dashboard, run:
```bash
python Inference/gpp_gui.py
```

### B. SCOPE PLS-TabPFN Model Training
To train the PLS-TabPFN pipeline on the provided dataset and save the trained model artifacts, run:
```bash
python SCOPE/PLS_TabPFN.py
```

### C. Wavelength Importance Analysis
To compute permutation-based wavelength importances (TabPFN-aware) and save the importance mappings:
```bash
python SCOPE/wavelenght_importance.py
```

To visualize the computed spectral band importances and flag the water/noisy bands:
```bash
python SCOPE/plot_importance.py
```

---

## ⚖️ Licensing
This repository is licensed under the **Prior Labs License Version 1.2** (Apache 2.0 with additional provision). A copy of the license is included in the `LICENSE` file. Please read the license to understand your redistribution and attribution obligations.
