# fMRI MVPA scripts in Tseng, 2025 (Ch 2 in hal.science/tel-04988192)

---

# Cross-Modal Decoding

This script performs **cross-modal decoding** (classification) of fMRI data, specifically using a linear SVM classifier to train on one modality (e.g., movement) and test on another (e.g., perception), within a specified region of interest (individual 10mm sphere ROI).

## overview

- **Modality 1 (e.g., Movement)**: Used to train a classifier
- **Modality 2 (e.g., Perception)**: Used for testing
- **Region of Interest**: Individual 10mm spheres (in the precentral gyrus as in indivROI_10sphere.py)
- **Preprocessing**: To minimize univariate differences between the two modalities and the two scan sequences (Tseng, 2025), each t-map is demeaned and standardized for each modality and group (Rezk et al., 2020).
- **Classification**: LinearSVC with `LeaveOneGroupOut` cross-validation
- **Output**: Decoding accuracy within and across modalities

---

## install required packages with

```bash
pip install nilearn nibabel scikit-learn matplotlib
```

---

# Individual 10mm sphere ROI

