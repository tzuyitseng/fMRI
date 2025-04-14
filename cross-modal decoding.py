""" 
Cross-modal decoding
Train one modality (e.g., movement) of fMRI data and test on another modality (e.g. perception) 
within an ROI (e.g. individual 10mm sphere in the precentral gyrus).
To minimize the univariate differences between the two modalities and the two scan sequences (in Tseng T., 2025), 
each t-map in each modality of each participant was demeanalized (demeaning and z scored standardization; Rezk et al., 2020).
The classification was conducted with a LeaveOneGroupOut cross-validation scheme.

March 28 2025 first uploaded 
by Tzuyi TSENG
tzuyitseng.neuroling@gmail.com
"""

import os
import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiMasker
from nilearn.decoding import Decoder
from nilearn.image import load_img, mean_img, math_img, index_img
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneGroupOut
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from nilearn.plotting import view_img, plot_stat_map, show


def load_apply_mask_and_demean(subject, conditions, groups, path, task_name, mask_img, demean_path):
    """ optional: define a function to load, demean, apply individual ROI mask and save image """
    img_paths = []
    masked_images = []
    demeaned_images = []
    group_masked = {group: [] for group in groups} # each 'group' corresponds to a scan run (e.g., run 1 to 6), as indicated in the filenames

    # load all t-maps and apply individual ROI mask
    for condition in conditions:
        for group in groups:
            img_path = os.path.join(path, f'sub-{subject}', f'sub-{subject}_task-{task_name}_desc-{condition}-{group}_stat.nii.gz') # task_name: two modalities
            img_paths.append(img_path)
            nifti_masker = NiftiMasker(mask_img=mask_img) # load mask, see indivROI_10sphere.py
            masked_img_np = nifti_masker.fit_transform(img_path) # fit the mask and transform the image as numpy array
            masked_img_4d = nifti_masker.inverse_transform(masked_img_np) # transform numpy array back to 4D image
            masked_img = index_img(masked_img_4d, 0) # index 4D image as 3D image for later use
            masked_images.append(masked_img)
            group_masked[group].append(masked_img)

    mean_img_group = {group: mean_img(group_masked[group]) for group in groups} # get the mean image of two modalities within each group, which are the objects to train and test

    # demean the data respective to two modalities within each group (Rezk et al., 2020)
    for idx, masked_img in enumerate(masked_images):
        group_index = idx % len(groups)
        group = groups[group_index]
        mean_image = mean_img_group[group]
        demeaned_img = math_img('img - mean_image', img=masked_img, mean_image=mean_image)
        demeaned_images.append(demeaned_img)      
        demeaned_path = os.path.join(demean_path, f'sub-{subject}_task-{task_name}_desc-{conditions[idx // len(groups)]}-{group}_demeaned.nii.gz') # name and save the demeaned images
        nib.save(demeaned_img, demeaned_path)

    return img_paths, mean_img_group, demeaned_images

def load_demeaned_images(subject, condition, groups, path, task_name):
    """ load the saved and demeaned images """
    images = []
    labels = []
    for group in groups:
        img_path = os.path.join(path, f'sub-{subject}_task-{task_name}_desc-{condition}-{group}_demeaned.nii.gz')
        images.append(img_path)
        labels.append(condition)
    return images, labels


# Paths (change '...' as your paths)
data_path = '...' # first-level imgs (e.g. condition vs baseline img with motion correction)
motor_path = '...' # img of one modality e.g. movement
percep_path = '...' # img of another modality e.g. perception
mask_path = '...' # individual 10mm sphere mask if needed
motor_demean_path = '...' # demeaned img of one modality
percep_demean_path = '...' # demeaed img of another modality

sub_list = ['sub1', 'sub2']

# loop over hemispheres
for selected_hemisphere in ['RH', 'LH']:
    # Update the mask pattern based on the selected hemisphere
    mask_pattern = '{}_ROI10_{}.nii.gz'.format('{}', selected_hemisphere) # mask file name as e.g. sub1_ROI10_RH
    print('*' * 100)
    print('[', selected_hemisphere, ']')
    print('*' * 100)

    # define perception conditions
    test_conditions_percep_pairs = [
        ['01-labial-BL', '02-dental-BL'],
        ['03-intact-labial-BL', '04-intact-dental-BL'], # examples of my image data conditions
    ]

    # loop over subjects
    for subject in sub_list:
        mask_img = load_img(os.path.join(mask_path, mask_pattern.format(subject)))

        # define motor conditions
        all_train_images_motor = []
        all_train_labels_motor = []
        all_train_groups_motor = []
        train_conditions_motor = ['Lip-BL', 'Tongue-BL']
        img_paths_motor, mean_img_group_motor, demeaned_images_motor = load_apply_mask_and_demean(subject, train_conditions_motor, np.arange(1, 7), motor_path, 'loc', mask_img, motor_demean_path)

        # Load and process motor conditions
        for condition in train_conditions_motor:
            images_motor, labels_motor = load_demeaned_images(subject, condition, np.arange(1, 7), motor_demean_path, 'loc')
            nifti_masker_motor = NiftiMasker(mask_img=mask_img)
            images_motors = nifti_masker_motor.fit_transform(images_motor)
            all_train_images_motor.extend(images_motors)
            all_train_labels_motor.extend(['Lips' if label == 'Lip-BL' else 'Tongue' for label in labels_motor]) # define class/category labels to be classified based on condition names.
            all_train_groups_motor.extend([f'Group-{i}' for i in np.arange(1, 7)])

        all_accuracy_fold_percep_pair = [] 
        avg_accuracy_fold_motor = []
        # loop over perception condition pairs
        for pair_idx, perception_pair in enumerate(test_conditions_percep_pairs):
            test_conditions_percep = perception_pair
            img_paths_percep, mean_img_group_percep, demeaned_images_percep = load_apply_mask_and_demean(subject, test_conditions_percep, np.arange(1, 7), percep_path, 'phono', mask_img, percep_demean_path)

            # load and process perception conditions
            all_test_images_percep = []
            all_test_labels_percep = []
            all_test_groups_percep = []
            for condition in perception_pair:
                images_percep, labels_percep = load_demeaned_images(subject, condition, np.arange(1, 7), percep_demean_path, 'phono')
                nifti_masker_percep = NiftiMasker(mask_img=mask_img)
                images_perceps = nifti_masker_percep.fit_transform(images_percep)
                all_test_images_percep.extend(images_perceps)
                all_test_labels_percep.extend(['Lips' if 'labial' in label else 'Tongue' for label in labels_percep])
                all_test_groups_percep.extend([f'Group-{i}' for i in np.arange(1, 7)])

            # define cross-validation
            cv_motor = LeaveOneGroupOut()
            # optional: define classifier with pipeline
            decoder_motor = make_pipeline(StandardScaler(with_mean=False), LinearSVC()) # standardize the data without demeaning again
            ## decoder_motor = Decoder(estimator="svc", mask=mask_img, standardize="zscore_sample") # simply use nilearn.decoding.Decoder() if no concerns for univariate differences

            all_accuracy_fold_motor = []
            all_accuracy_fold_percep = []
            folds = []
            # perform cross-validation for motor
            # split condition defined here: the same group of the two labels are splitted into either train or test set
            for fold_motor, (train_idx_motor, test_idx_motor) in enumerate(cv_motor.split(all_train_images_motor, all_train_labels_motor, groups=all_train_groups_motor), 1):
                X_train_motor, X_test_motor = np.array(all_train_images_motor)[train_idx_motor], np.array(all_train_images_motor)[test_idx_motor]
                y_train_motor, y_test_motor = np.array(all_train_labels_motor)[train_idx_motor], np.array(all_train_labels_motor)[test_idx_motor]
                folds.append(fold_motor)
                train_groups_motor = np.array(all_train_groups_motor)[train_idx_motor]

                decoder_motor.fit(X_train_motor, y_train_motor)
                coef_shapes = decoder_motor.named_steps['linearsvc'].coef_.shape[1] # if use nilearn.decoding.Decoder, simply get the info from decoder_motor.coef_
                y_pred_motor = decoder_motor.predict(X_test_motor)
                y_true_motor = y_test_motor
                accuracy_fold_motor = accuracy_score(y_true_motor, y_pred_motor)
                all_accuracy_fold_motor.append(accuracy_fold_motor)
                # optional: print info to check if needed
                print(f'decoding: {subject} {selected_hemisphere} fold: {fold_motor}')
                print(f'accuracy_fold_motor:{accuracy_fold_motor}','\n')

                # test on perception images using the same motor decoder
                y_pred_percep = decoder_motor.predict(all_test_images_percep)
                y_true_percep = all_test_labels_percep
                accuracy_fold_percep = accuracy_score(y_true_percep, y_pred_percep)
                all_accuracy_fold_percep.append(accuracy_fold_percep)
                # optional: print info to check if needed
                print(f'y_true_percep: {y_true_percep}')
                print(f'y_pred_percep: {y_pred_percep}')
                print(f'accuracy_fold_percep: {perception_pair} {accuracy_fold_percep}')
                print('-' * 50)
