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
from rsatoolbox.inference import eval_fixed
from rsatoolbox.model import ModelFixed
from rsatoolbox.rdm import RDMs
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight
from scipy.stats import spearmanr
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

"""searchlight RSA for each individual for each theoretical RDM"""
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



"""group-level analysis using permutation test using Nilearn"""
import matplotlib.pyplot as plt
from nilearn import image, maskers, plotting, datasets, masking
from nilearn.image import threshold_img, math_img, load_img, resample_to_img, coord_transform, mean_img, get_data, index_img
from nilearn.plotting import view_img, plot_stat_map, plot_glass_brain, plot_img, plot_design_matrix
from nilearn.glm.thresholding import threshold_stats_img
from nilearn.reporting import get_clusters_table
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference

for ml in models:
    print(ml)
    # load individual searchlight RSA images
    imgs = []
    for i, subject in enumerate(sub_list):
        img_sub = os.path.join(save_dir, 'searchlightRSA', f'{subject}_seachlightRSA_{ml}.nii.gz')
        imgs.append(img_sub)

    # second-level design matrix
    second_level_input = imgs
    design_matrix = pd.DataFrame(
        [1] * len(second_level_input),
        columns=["intercept"],
    )
    
    ## permutation test
    out_dict_voxel = non_parametric_inference(
        second_level_input,
        design_matrix=design_matrix,
        model_intercept=True,
        mask=mask_img,
        n_perm=10000,
        two_sided_test=True,
        n_jobs=-1,
        threshold=0.05
    )
    
    # this gives an oupt image with negative log10 family-wise error rate-corrected p-values corrected threshold (voxel-level error control)
    #for other threshold see: https://nilearn.github.io/dev/modules/generated/nilearn.glm.second_level.non_parametric_inference.html
    result_img = out_dict_voxel['logp_max_t']
    result_path = os.path.join(data_path, f'perm_seachlightRSA_FWE_{ml}.nii.gz')
    nib.save(result_img, result_path)
