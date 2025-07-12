# ArielML: Exoplanet Data Analysis Pipeline

This project provides a comprehensive, high-performance Python pipeline for analyzing simulated data from the ESA Ariel Space Telescope, as part of the "Ariel Data Challenge 2025". The primary goal is to process raw time-series image data to recover the atmospheric transmission spectra of exoplanets.

## ✨ Features

* **Modular Library (`arielml`):** All core logic is encapsulated in a clean, scalable, and installable Python library.
* **High-Performance Backend:** A backend-agnostic design allows all numerical operations to run seamlessly on either a **CPU (NumPy)** or a **GPU (CuPy)** for maximum performance.
* **End-to-End Data Reduction:** Implements the full pipeline from raw data to clean light curves:
    * Full instrument calibration (ADC, masking, linearity, dark, flat, CDS).
    * Robust aperture photometry with cosmic ray rejection.
* **Interactive Visualization Tool:** A powerful GUI built with PyQt6 for interactively exploring the data, visualizing the impact of each processing step, and comparing CPU vs. GPU performance.

## 📂 Project Structure

The project is organized into a clean and logical directory structure:

```plaintext
.
├── arielml/            # The core, installable Python library
│   ├── backend.py      # Handles CPU/GPU backend switching
│   ├── config.py       # Central configuration for paths and parameters
│   ├── data/           # Data processing modules
│   │   ├── calibration.py
│   │   ├── loaders.py
│   │   ├── observation.py
│   │   └── photometry.py
│   └── ...
├── dataset/            # Target directory for raw competition data
│   └── .gitkeep
├── notebooks/          # Jupyter notebooks for experimentation
├── output/             # Target directory for generated files (models, submissions)
│   └── .gitkeep
├── tools/              # Standalone GUI applications
│   └── data_inspector.py
├── .gitignore          # Specifies files and directories to be ignored by Git
├── README.md           # This file
└── requirements.txt    # Project dependencies
```

## 🚀 Setup and Installation

Follow these steps to set up the project environment.

**1. Clone the Repository:**

```bash
git clone git@github.com:jalalirs/arielml.git
cd arielml
```

**2. Create a Virtual Environment:**
It is highly recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

**3. Install Dependencies:**
Install all required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

*Note: For GPU support, you must have the NVIDIA CUDA Toolkit installed and then install the appropriate version of `cupy`.*

**4. Download the Data:**
Download the competition data and place it inside the `dataset/` directory. The structure should look like:

```
dataset/
├── train/
├── test/
├── train.csv
└── ...
```

## 🛠️ Usage

### Running the Data Inspector Tool

The primary tool for visualization and debugging is the `Data Inspector`. To run it, execute the following command from the project's root directory:

```bash
python tools/data_inspector.py
```

This will launch the GUI, allowing you to load planet data, apply calibration steps, and visualize the results on either the CPU or GPU.

## 🔮 Next Steps

With the data processing foundation now complete, the project will focus on the following machine learning and analysis tasks:

1.  **Detrending:** Implement algorithms to remove systematic noise from the 1D light curves to isolate the exoplanet transit signal.
2.  **Model Building:** Develop machine learning models to predict the transit depth (the spectrum) from the clean light curves.
3.  **Pipeline Integration:** Create end-to-end training and prediction pipelines to generate the final competition submission.
