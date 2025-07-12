Project Summary: Ariel Exoplanet Data Challenge
This document outlines the problem statement, our current progress, and the architecture of our solution for the "Ariel Data Challenge 2025" competition.

1. The Problem Statement
The core objective is to analyze simulated data from the European Space Agency's (ESA) Ariel space telescope to recover the atmospheric transmission spectra of exoplanets.

Goal: For each exoplanet, we must predict its spectrum (the amount of starlight its atmosphere blocks at different wavelengths) and our uncertainty for that prediction.

Input Data: We are given raw, time-series image data from two different instruments:

AIRS-CH0: A spectrometer that spreads starlight into its constituent colors (wavelengths).

FGS1: A photometer that captures the total brightness of the star in a single spot.

Core Challenge: The faint signal from the exoplanet's atmosphere is buried in significant noise from the telescope's electronics, cosmic rays, and the natural variability of the host star. Our task is to build a robust pipeline to clean this data and isolate the true signal.

2. Our Approach & Architecture
We have adopted a professional, modular approach to solve this problem, focusing on scalability, performance, and interpretability.

Modular Library (arielml): All core logic is being built into a custom, installable Python library. This keeps our experimental notebooks clean and ensures code reusability.

Object-Oriented Pipeline: We created a central DataObservation class. An instance of this class represents a single observation (e.g., one planet, one instrument, one visit) and encapsulates all the data and processing logic for it.

High-Performance Backend: The library features a backend-agnostic design that can run all numerical operations on either the CPU (using NumPy) or a GPU (using CuPy). This allows for rapid prototyping on any machine and high-speed processing on GPU-enabled systems.

Interactive Visualization Tool: We built a sophisticated GUI application, the Data Inspector, using PyQt6. This tool is critical for debugging our pipeline, visualizing the effect of each processing step in real-time, and gaining intuition about the data.

3. Current Progress & Implemented Features
We have successfully completed the entire data reduction phase, from raw images to clean, 1D light curves.

Full Calibration Pipeline: We have implemented a complete, GPU-accelerated calibration pipeline that performs:

ADC Conversion

Hot/Dead Pixel Masking

Vectorized Linearity Correction

Dark Current Subtraction

Correlated Double Sampling (CDS)

Flat Field Correction

Robust Photometry Pipeline: We have implemented an aperture photometry function that:

Handles both instrument types correctly (spectrometer vs. photometer).

Uses sigma clipping to reject outliers both spatially and temporally (cosmic rays).

Extracts clean, background-subtracted 1D light curves for each wavelength.

Data Inspector Tool: Our visualization tool is complete and feature-rich, allowing us to:

Load any observation for any planet.

Select a CPU or GPU backend for processing.

Interactively toggle calibration steps.

Visualize the 2D detector images with photometry apertures overlaid.

Plot the final 1D light curves.

View star/planet metadata and performance logs.

4. Project File Structure
Our project is organized into the following structure:

.
├── arielml/
│   ├── backend.py
│   ├── config.py
│   ├── data/
│   │   ├── calibration.py
│   │   ├── detrending.py
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── observation.py
│   │   └── photometry.py
│   ├── evaluation/
│   │   └── ...
│   ├── __init__.py
│   ├── models/
│   │   └── ...
│   └── pipelines/
│       └── ...
├── dataset/
│   └── (Raw competition data)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_prototyping.ipynb
│   └── submission.ipynb
├── output/
│   └── ...
├── tools/
│   └── data_inspector.py
└── ... (setup files)

5. Next Steps
With the data cleaning and extraction phase complete, our immediate next steps are to move into the final stages of the analysis:

Detrending: Implement algorithms in arielml/data/detrending.py to remove the remaining systematic noise from our 1D light curves and isolate the exoplanet transit signal.

Model Building: Begin implementing machine learning models in arielml/models/ to predict the transit depth (the spectrum) from the detrended light curves and associated metadata.