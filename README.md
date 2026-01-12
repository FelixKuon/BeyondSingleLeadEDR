# BeyondSingleLeadEDR
# 3D EASI Vectorcardiogram EDR Pipeline for Respiratory Frequency Estimation

## Overview
This repository contains the complete Python pipeline for processing mobile EASI-lead ECG data into 3D Frank-XYZ vectorcardiograms (VCG) and deriving beat-to-beat respiration surrogate signals. The methods enable accurate estimation of slow-paced breathing frequencies (< 0.1 Hz), outperforming traditional single-lead EDR approaches based on RR-intervals or R-amplitude.

Key features:
- EASI-to-Frank-XYZ transformation for 3D heart vector reconstruction [file:16]
- Robust 3D kinematic R-peak detection (adaptive thresholding, topology-based)
- Multivariate Mahalanobis-distance artifact correction across multiple features
- HeartMovement signal: Rotation-based EDR from complex angular projection
- Continuous Wavelet Transform (CWT) ridge analysis for non-stationary frequency tracking
- No restrictive bandpass filtering – preserves ultra-low frequencies

The pipeline was developed for the paper "Beyond Single-Lead EDR Respiration Estimation from 3D EASI Vectorcardiograms for Accurate Tracking of Slow-Paced Breathing" and validated on paced-breathing data from healthy adults.

## Usage
1. Clone the repo: `git clone https://github.com/YOURUSERNAME/REPONAME.git`
2. Install dependencies: `pip install -r requirements.txt` (numpy, scipy, matplotlib, etc.)
3. Place your EASI ECG files (.csv or .edf) in `/data/`
4. Run: `python main_pipeline.py --input data/your_file.csv --output results/`

Example Jupyter notebooks:
- `demo_edr_analysis.ipynb`: Full walkthrough on sample data
- `3d_vcg_visualization.ipynb`: Interactive VCG plotting

## Results Summary
- Superior SNR, frequency stability, and accuracy vs. RR/Amplitude-EDR [file:16]
- Handles 6–18 bpm (0.1–0.3 Hz) reliably
- Fully reproducible with provided code and configs

## License
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This code is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.

- **Allowed**: Non-commercial research, education, personal use, modifications, distribution (with attribution).
- **Not allowed**: Commercial use (e.g., in products sold for profit).
- Derivatives must use the same license.

See [LICENSE](LICENSE) for full terms. For commercial inquiries, contact [].

## Citation
If you use this code in your work, please cite the paper:



**Note**: Original ECG data is not included due to privacy restrictions. Contact the study PI for access.
