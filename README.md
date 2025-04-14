## fMRI MVPA scripts

- Here I share my scripts for fMRI MVPA performed in my doctoral thesis (Tseng, 2025): Sensorimotor representations for native and non-native phoneme perception (Chpater 2 in hal.science/tel-04988192).
- These scripts are based on <a href="https://nilearn.github.io/stable/index.html">Nilearn</a> for cross-modal decoding and <a href="https://rsatoolbox.readthedocs.io/en/stable/">rsatoolbox</a> for representational similarity analysis in Python.
- Nilearn weekly <a href="https://nilearn.github.io/stable/development.html#how-to-get-help">drop-in hours</a> helped me a lot during my PhD study to discuss my questions directly with developers.

---

### Cross-Modal Decoding

This script performs **cross-modal decoding** (classification) of fMRI data, specifically using a linear SVM classifier to train on one modality (e.g., movement) and test on another (e.g., perception), within a specified region of interest (individual 10mm sphere ROI).

- **Modality 1 (e.g., movement)**: Used to train a classifier
- **Modality 2 (e.g., perception)**: Used for testing using the same classifier trained for modality 1
- **Region of Interest**: Individual 10mm spheres (in the precentral gyrus as in indivROI_10sphere.py)
- **Preprocessing**: To minimize univariate differences between the two modalities and the two scan sequences, each t-map is demeaned and standardized for each modality and group (Rezk et al. Curr. Biol. 2020).
- **Classification**: LinearSVC with `LeaveOneGroupOut` cross-validation
- **Output**: Decoding accuracy within and across modalities
- install required packages with
```bash
pip install nilearn nibabel scikit-learn matplotlib
```

---

### Individual 10mm sphere ROI

... under construction ...
