""" 
searchlight representational similarity analysis (RSA)

A 10 mm spherical radius moved across each voxel in the combined ROI, with the voxel serving as the center of the searchlight sphere.
At each searchlight center, a neural RDM was generated and correlated to a (self-defined) theoretical RDM using an open-source Python toolbox (rsatoolbox.readthedocs.io). 
Each neural RDM was calculated from the pairwise correlation between neural patterns of each condition (the original script included n = 16) using cosine distance, resulting in a 16 x 16 pairwise matrix. 
The correlation between each of the theoretical RDMs (predicted similarity) and the neural RDMs (observed similarity) was calculated using Spearman's rho (ρ).

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
tRDM_path = '...'
save_dir = '...'

sub_list = ['sub1', 'sub2']
conditions = ['condition1', 'condition2', ..., 'condition16']
models = ['theoreticalRDM1','theoreticalRDM2']

all_eval_scores = {}

for ml in models:
    eval_scores_per_subject = []
    print(f'====={ml}=====')

    for subject in sub_list:
        # load the ROI mask; convert the mask as a binary image; get the volume dimensions
        mask_img = nib.load(data_path + 'ROIs_mask.nii.gz') 
        mask = mask_img.get_fdata() > 0
        x, y, z = mask_img.get_fdata().shape

        image_paths = []
        for condition in conditions:
            # load individual tmap of each condition
            tmap_paths = os.path.join(tmap_path, f'{subject}_task-phono_desc-{condition}-BL_stat.nii.gz') # change the file name format as yours
            image_paths.append(tmap_paths)
        print(subject, len(image_paths))

        data = np.zeros((len(image_paths), x, y, z))
        for x, im in enumerate(image_paths):
            data[x] = nib.load(im).get_fdata()

        """rsatoolbox function get_volume_searchlight: define the searchlight range"""
        centers, neighbors = get_volume_searchlight(mask, radius=4, threshold=0.5)  # radius=4: 10mm radius in voxel units

        """rsatoolbox function get_searchlight_RDMs: generate neural RDM"""
        data_2d = data.reshape([data.shape[0], -1])
        data_2d = np.nan_to_num(data_2d)
        image_value = np.arange(len(image_paths))
        neural_rdms = get_searchlight_RDMs(data_2d, centers, neighbors, image_value, method='correlation')

        """rsatoolbox function RDMs: generate theoretical RDM prior defined in a csv file; ModelFixed: define generated theoretical RDM as the fixed model that will be fit to the neural RDM"""
        concept_rdm_array = pd.read_csv(os.path.join(tRDM_path, f'{ml}.csv'), index_col=0).values
        concept_rdm_vector = concept_rdm_array[np.triu_indices_from(concept_rdm_array, k=1)]
        concept_rdm_array = np.array([concept_rdm_vector])
        concept_rdm = RDMs(dissimilarities=concept_rdm_array, descriptors={'conditions': conditions})
        model = ModelFixed('Conceptual Model', concept_rdm)

        """rsatoolbox function evaluate_models_searchlight: correlate neural RDM and theoretical RDM using spearman"""
        eval_results = evaluate_models_searchlight(neural_rdms, model, eval_fixed, method='spearman')
        eval_score = [float(e.evaluations) for e in eval_results]

        # Store the eval_score for this subject
        eval_scores_per_subject.append(eval_score)

        # Create a 3D array, with the size of the mask
        x, y, z = mask.shape
        RDM_brain = np.zeros([x * y * z])
        RDM_brain[list(neural_rdms.rdm_descriptors['voxel_index'])] = eval_score
        RDM_brain = RDM_brain.reshape([x, y, z])

        subject_rsa_img = new_img_like(mask_img, RDM_brain)
        result_path = os.path.join(save_dir, 'searchlightRSA', f'{subject}_seachlightRSA_{ml}.nii.gz') #change "searchlightRSA" as your filename
        nib.save(subject_rsa_img, result_path)

    all_eval_scores[ml] = eval_scores_per_subject

# Save the eval scores as CSV
for ml, scores in all_eval_scores.items():
    df = pd.DataFrame(scores, index=sub_list)
    csv_path = os.path.join(save_dir, 'searchlightRSA', 'eval_scores', f'{ml}_eval_scores.csv')
    df.to_csv(csv_path)
