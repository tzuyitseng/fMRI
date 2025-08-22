""" 
searchlight representational similarity analysis (RSA)
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

all_eval_scores = {}

for ml in models:
    eval_scores_per_subject = []
    print(f'====={ml}=====')

    for subject in sub_list:
        mask_img = nib.load(data_path + 'mvpa/searchlight/7ROIs_mask.nii.gz')
        mask = mask_img.get_fdata() > 0
        x, y, z = mask_img.get_fdata().shape

        image_paths = []
        for condition in conditions:
            tmap_paths = os.path.join(tmap_path, f'{subject}_task-phono_desc-{condition}-BL_stat.nii.gz')
            image_paths.append(tmap_paths)
        print(subject, len(image_paths))

        data = np.zeros((len(image_paths), x, y, z))
        for x, im in enumerate(image_paths):
            data[x] = nib.load(im).get_fdata()

        centers, neighbors = get_volume_searchlight(mask, radius=4, threshold=0.5)  # 10mm radius in voxel units

        data_2d = data.reshape([data.shape[0], -1])
        data_2d = np.nan_to_num(data_2d)
        image_value = np.arange(len(image_paths))
        neural_rdms = get_searchlight_RDMs(data_2d, centers, neighbors, image_value, method='correlation')

        concept_rdm_array = pd.read_csv(os.path.join(save_dir, 'conceptualRDMs', 'phono_feature', f'{ml}.csv'), index_col=0).values
        concept_rdm_vector = concept_rdm_array[np.triu_indices_from(concept_rdm_array, k=1)]
        concept_rdm_array = np.array([concept_rdm_vector])
        concept_rdm = RDMs(dissimilarities=concept_rdm_array, descriptors={'conditions': conditions})
        model = ModelFixed('Conceptual Model', concept_rdm)

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
        result_path = os.path.join(save_dir, 'searchlightRSA_all_lang', f'{subject}_seachlightRSA_all_{ml}.nii.gz')
        nib.save(subject_rsa_img, result_path)

    all_eval_scores[ml] = eval_scores_per_subject

# Save the eval scores as CSV
for ml, scores in all_eval_scores.items():
    df = pd.DataFrame(scores, index=sub_list)
    csv_path = os.path.join(save_dir, 'searchlightRSA_all_lang', 'eval_scores', f'{ml}_eval_scores.csv')
    df.to_csv(csv_path)
