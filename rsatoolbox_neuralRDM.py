""" 
representational similarity analysis (RSA): gernaration of neural representational (dis)similarity matrices (RDMs)

Correlate the neural patterns of the associated conditions within each ROI.
The current script

22 August 2025 first uploaded 
by Tzuyi TSENG
tzuyitseng.neuroling@gmail.com
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn.maskers import NiftiMasker
from nilearn import image
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

# Paths (change '...' as your paths)
data_path = '...'
mask_path = '...'
tmap_path = '...'
save_dir = '...'

sub_list = ['sub1', 'sub2']
conditions = ['condition1', 'condition2', ..., 'condition16']
rois = ['PreCG','STG']

similarity_matrices = []
for roi in rois:
    for selected_hemisphere in ['LH', 'RH']:
        
        for idx, subject in enumerate(sub_list):
            # Initialize an empty list to collect t-maps for the current subject
            subject_tmaps = []

            mask_pattern = f'{roi}_mask_{selected_hemisphere}.nii.gz'
            mask_img = image.load_img(os.path.join(mask_path, mask_pattern))
            nifti_masker = NiftiMasker(mask_img=mask_img, standardize=False) #'zscore_sample'

            for condition in conditions:
                tmap_file = os.path.join(tmap_path, f'{subject}_task-phono_desc-{condition}-BL_stat.nii.gz')

                if os.path.exists(tmap_file):
                    masked_tmap = nifti_masker.fit_transform(tmap_file)
                    subject_tmaps.append(masked_tmap)

            # Convert list to numpy array
            subject_tmaps = np.array(subject_tmaps).reshape((len(conditions), -1))

            # Compute correlation dissimilarities for the current subject
            n_conditions = len(conditions)
            correlation_dissimilarities = np.zeros((n_conditions, n_conditions))

            for i in range(n_conditions):
                for j in range(n_conditions):
                    if i != j:
                        corr, _ = pearsonr(subject_tmaps[i], subject_tmaps[j])
                        correlation_dissimilarities[i, j] = 1 - corr
                    else:
                        correlation_dissimilarities[i, j] = 0

            df = pd.DataFrame(correlation_dissimilarities, index=conditions, columns=conditions)
            df.to_csv(os.path.join(save_dir, f'{subject}_{roi}_{selected_hemisphere}.csv'))
            
            similarity_matrices.append(correlation_dissimilarities)

        # Compute the average similarity matrix across subjects
        average_similarity_matrix = np.mean(similarity_matrices, axis=0)

        # Create a DataFrame for the averaged similarity matrix
        rdm_df = pd.DataFrame(average_similarity_matrix, index=conditions, columns=conditions)
        rdm_df.to_csv(os.path.join(save_dir, f'{roi}_{selected_hemisphere}_averaged.csv'))