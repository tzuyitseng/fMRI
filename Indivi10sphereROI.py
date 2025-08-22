""" 
individual 10mm sphere ROI mask

22 August 2025 first uploaded 
by Tzuyi TSENG
tzuyitseng.neuroling@gmail.com
"""

# ======================================================================
import numpy as np
import os
import nibabel as nib
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn import neighbors
from joblib import Memory
from scipy import sparse
from nibabel import Nifti1Image
from nilearn.image.resampling import coord_transform
from nilearn._utils.niimg_conversions import _safe_get_data
from nilearn._utils import CacheMixin, logger
from nilearn._utils.niimg import img_data_dtype
from nilearn._utils.niimg_conversions import check_niimg_4d, check_niimg_3d
from nilearn._utils.class_inspect import get_params
from nilearn import image, maskers, plotting, datasets, masking
from nilearn.image import threshold_img, math_img, resample_img, resample_to_img
from nilearn.maskers import nifti_spheres_masker, NiftiSpheresMasker
from nilearn.masking import _unmask_3d
from nilearn.plotting import view_img
from nilearn.glm.thresholding import threshold_stats_img
from nilearn.reporting import get_clusters_table
from atlasreader import create_output
# ======================================================================
def _apply_mask_and_get_affinity(seeds, niimg, radius, allow_overlap,
                                 mask_img=None):
    '''Utility function to get only the rows which are occupied by sphere at
    given seed locations and the provided radius. Rows are in target_affine and
    target_shape space.

    Parameters
    ----------
    seeds : List of triplets of coordinates in native space
        Seed definitions. List of coordinates of the seeds in the same space
        as target_affine.

    niimg : 3D/4D Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Images to process. It must boil down to a 4D image with scans
        number as last dimension.

    radius : float
        Indicates, in millimeters, the radius for the sphere around the seed.

    allow_overlap: boolean
        If False, a ValueError is raised if VOIs overlap

    mask_img : Niimg-like object, optional
        Mask to apply to regions before extracting signals. If niimg is None,
        mask_img is used as a reference space in which the spheres 'indices are
        placed.

    Returns
    -------
    X : 2D numpy.ndarray
        Signal for each brain voxel in the (masked) niimgs.
        shape: (number of scans, number of voxels)

    A : scipy.sparse.lil_matrix
        Contains the boolean indices for each sphere.
        shape: (number of seeds, number of voxels)

    '''
    seeds = list(seeds)

    # Compute world coordinates of all in-mask voxels.
    if niimg is None:
        mask, affine = masking._load_mask_img(mask_img)
        # Get coordinate for alle voxels inside of mask
        mask_coords = np.asarray(np.nonzero(mask)).T.tolist()
        X = None

    elif mask_img is not None:
        affine = niimg.affine
        mask_img = check_niimg_3d(mask_img)
        mask_img = image.resample_img(mask_img, target_affine=affine,
                                      target_shape=niimg.shape[:3],
                                      interpolation='nearest')
        mask, _ = masking._load_mask_img(mask_img)
        mask_coords = list(zip(*np.where(mask != 0)))

        X = masking._apply_mask_fmri(niimg, mask_img)
    elif niimg is not None:
        affine = niimg.affine
        if np.isnan(np.sum(_safe_get_data(niimg))):
            warnings.warn('The imgs you have fed into fit_transform() contains'
                          ' NaN values which will be converted to zeroes ')
            X = _safe_get_data(niimg, True).reshape([-1, niimg.shape[3]]).T
        else:
            X = _safe_get_data(niimg).reshape([-1, niimg.shape[3]]).T
        mask_coords = list(np.ndindex(niimg.shape[:3]))
    else:
        raise ValueError("Either a niimg or a mask_img must be provided.")

    # For each seed, get coordinates of nearest voxel
    nearests = []
    for sx, sy, sz in seeds:
        nearest = np.round(coord_transform(sx, sy, sz, np.linalg.inv(affine)))
        nearest = nearest.astype(int)
        nearest = (nearest[0], nearest[1], nearest[2])
        try:
            nearests.append(mask_coords.index(nearest))
        except ValueError:
            nearests.append(None)

    mask_coords = np.asarray(list(zip(*mask_coords)))
    mask_coords = coord_transform(mask_coords[0], mask_coords[1],
                                  mask_coords[2], affine)
    mask_coords = np.asarray(mask_coords).T

    clf = neighbors.NearestNeighbors(radius=radius)
    A = clf.fit(mask_coords).radius_neighbors_graph(seeds)
    A = A.tolil()
    for i, nearest in enumerate(nearests):
        if nearest is None:
            continue
        A[i, nearest] = True

    # Include the voxel containing the seed itself if not masked
    mask_coords = mask_coords.astype(int).tolist()
    for i, seed in enumerate(seeds):
        try:
            A[i, mask_coords.index(list(map(int, seed)))] = True
        except ValueError:
            # seed is not in the mask
            pass

    sphere_sizes = np.asarray(A.tocsr().sum(axis=1)).ravel()
    empty_spheres = np.nonzero(sphere_sizes == 0)[0]
    if len(empty_spheres) != 0:
        raise ValueError("These spheres are empty: {}".format(empty_spheres))

    if not allow_overlap:
        if np.any(A.sum(axis=0) >= 2):
            raise ValueError('Overlap detected between spheres')

    return X, A
# =====================================================================
data_path = '...'
mask_path = '...'
sub_list = [...]

threshold = 0
for i, subject in enumerate(sub_list):
    print(subject)
    moto_LTF = os.path.join(data_path, f'sub-{subject}_task-loc_desc-13-ROI_stat.nii.gz')
    _, threshold = threshold_stats_img(
        moto_LTF, alpha=0.001,
        height_control='bonferroni',
        cluster_threshold=20,
        two_sided=False)
    table = get_clusters_table(moto_LTF, threshold, cluster_threshold=20)
    print(table)
    if subject == 'AJ032':
        peak_table = table[
            ((table['Cluster ID'].astype(str).str.isdigit()) &
             table['Cluster ID'].astype(str).isin(['1', '3']))]
    else:
        peak_table = table[
            ((table['Cluster ID'].astype(str).str.isdigit()) &
             table['Cluster ID'].astype(str).isin(['1', '2']))]
    print(peak_table)
    for index, row in peak_table.iterrows():
        cluster_id = str(row['Cluster ID'])
        x_coord, y_coord, z_coord = row['X'], row['Y'], row['Z']  
        if x_coord > 0:
            coords_RH = [x_coord, y_coord, z_coord]
        if x_coord < 0:
            coords_LH = [x_coord, y_coord, z_coord]
    brain_mask = datasets.load_mni152_brain_mask()
    moto_LTF = os.path.join(data_path, f'sub-{subject}_task-loc_desc-13-ROI_stat.nii.gz')
    space_defining_image = masking.compute_brain_mask(moto_LTF)
    for seed in [coords_RH, coords_LH]:
        if seed == coords_RH:
            H = 'RH'
        else:
            H = 'LH'
        _, A = nifti_spheres_masker._apply_mask_and_get_affinity(
            seeds=[seed],
            niimg=None,
            radius=10,
            allow_overlap=False, 
            mask_img=brain_mask)
        sphere = _unmask_3d(
            X=A.toarray().flatten(), 
            mask=brain_mask.get_fdata().astype(bool))
        sphere_img = Nifti1Image(sphere, brain_mask.affine)
        binary_mask = math_img('(np.abs(img) > 0)', img=sphere_img)
        sphere_mask = resample_to_img(binary_mask, space_defining_image)
        nib.save(sphere_mask, mask_path + f'{subject}_MNI-LTF-FWE10-ROI_{H}.nii.gz')
        plotting.plot_stat_map(moto_LTF, threshold=3, cut_coords=seed, title = f'Lip & Tongue vs Fingers localizer activation_sub-{subject}_{H}')
        plt.savefig(mask_path + f'{subject}_LTF-activation-FWE001_{H}.png')
        plotting.plot_roi(sphere_mask, cut_coords=seed, title = f'MNI152-LTF-FWE10-ROI_sub-{subject}_{H}')
        plt.savefig(mask_path + f'{subject}_MNI-LTF-FWE10-ROI_{H}.png')