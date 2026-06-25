# TabPFN GPP Model

This repository provides code and pre-trained models to perform GPP (Gross Primary Productivity) modeling and wavelength importance analysis using a unified **TabPFN** regression pipeline. 

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
├── Model/
│   ├── TabPFN_GPP.joblib    # [LFS] Pre-trained unified TabPFN-GPP model weights (122.6 MB)
│
└── SCOPE/
    ├── PLS_TabPFN.py              # Script to train TabPFN with PLS dimensionality reduction
    ├── wavelenght_importance.py   # Script to evaluate TabPFN wavelength importance via permutation
    ├── plot_importance.py         # Visualizes relative importance of spectral bands
    └── dataset_GPP_Ta_Rin_ea_reflectance_EMIT.csv  # Full training set from SCOPE RTM model
└── Validation/
    ├── plot.py              # Script to plot validation results
    └── validation.csv  # Test set from ICOS/AmeriFlux networks, ERA5 meteorological variables, EMIT real acquisitions.
```

---
## Note
We designed the entire pipeline in CUDA environment. Even if the codes can run on CPU, we suggest to adopt an NVIDIA GPU with at least 8 GB of VRAM. If you can't get access to a discrete GPU, you can find the key datasets already available.

## 🛠️ Installation & Setup

Before running the scripts or dashboard, ensure you set up your Python environment and install the required dependencies.


### 1. Installing Dependencies
Install all package requirements using `pip`:

```bash
pip install -r requirements.txt
```

### 2. Git LFS (Large File Storage) for Model Weights
The pre-trained model file `Model/TabPFN_GPP.joblib` is **122.6 MB**, which exceeds GitHub's standard 100 MB file limit. To clone or push this repository without losing the model weights, make sure Git LFS is installed and initialized on your system:

```bash
# Install Git LFS on your OS (Debian/Ubuntu example):
sudo apt-get install git-lfs

# Initialize Git LFS inside the repository directory:
git lfs install
```


---

## 💻 How to Run


### A. SCOPE PLS-TabPFN Model Training
To train the PLS-TabPFN pipeline on the provided dataset and save the trained model, run:
```bash
python SCOPE/PLS_TabPFN.py
```

### B. Wavelength Importance Analysis
To compute permutation-based wavelength importances (TabPFN-aware) and save the importance mappings:
```bash
python SCOPE/wavelenght_importance.py
```

To visualize the computed spectral band importances:
```bash
python SCOPE/plot_importance.py
```

### C. Validation Results Plotting
To evaluate model predictions against the independent validation dataset (from ICOS/AmeriFlux networks) and generate performance scatter plots:
```bash
python Validation/plot.py
```
This saves the validation performance plots (`validation_landcover.png` and `validation_dataset.png`) under the `Validation/` directory.

---

## ⚖️ Licensing
This repository is licensed under the **Prior Labs License Version 1.2** (Apache 2.0 with additional provision). A copy of the license is included in the `LICENSE` file. Please read the license to understand your redistribution and attribution obligations.
