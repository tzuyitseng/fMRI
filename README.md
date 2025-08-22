## fMRI MVPA scripts

- Here I share my scripts for fMRI MVPA performed in my doctoral thesis: <a href="https://hal.science/tel-04988192">Sensorimotor representations for native and non-native phoneme perception</a> (Chpater 2 in Tseng, 2025).
- These scripts are based on <a href="https://nilearn.github.io/stable/index.html">Nilearn</a>/<a href="https://scikit-learn.org/stable/">scikit-learn</a> for cross-modal decoding and <a href="https://rsatoolbox.readthedocs.io/en/stable/">rsatoolbox</a> for representational similarity analysis in Python.
- Nilearn weekly <a href="https://nilearn.github.io/stable/development.html#how-to-get-help">drop-in hours</a> helped me a lot during my PhD study to discuss my questions directly with developers.

---

### Cross-Modal Decoding

This script performs **cross-modal decoding** (classification) on fMRI data, specifically using a linear SVM classifier to train on one modality (e.g., movement) and test on another (e.g., perception), within a specified region of interest (individual 10mm sphere ROI).

- Modality 1 (e.g., movement): Used to train a classifier
- Modality 2 (e.g., perception): Used for testing using the same classifier trained for modality 1
- Region of Interest: Individual 10mm spheres in the precentral gyrus (see indivROI_10sphere.py)
- Preprocessing: To minimize univariate differences between the two modalities and the two scan sequences, each t-map is demeaned and standardized for each modality and group (Rezk et al. Curr. Biol. 2020).
- Classification: LinearSVC with `LeaveOneGroupOut` cross-validation
- Output: Decoding accuracy within and across modalities
- Packages required: 
```bash
pip install nilearn nibabel scikit-learn matplotlib
```

---

### Individual 10mm sphere ROI

... under construction ...

---

### searchlight RSA rsatoolbox

This script performs **searchlight RSA** on fMRI data, using an open-source Python toolbox (rsatoolbox.readthedocs.io) to compute the correlation between predicted and observed neural similarity patterns across experimental conditions.

- At each searchlight center (a spherical region systematically moved across voxels within the ROI), a neural representational dissimilarity matrix (RDM) was generated and compared to a predefined theoretical RDM.
- Neural RDMs were constructed from the pairwise correlations between condition-specific neural patterns, using cosine distance as the metric.
- For each examined voxel, the similarity between neural and theoretical RDMs was calculated with Spearman’s rank correlation (ρ).
- A stronger correlation indicates better alignment between the theoretical model and the empirical data, providing evidence that the modeled features are encoded in neural activity.
- To assess voxel-level significance, a non-parametric permutation test (n = 10,000) was further conducted using Nilearn.
