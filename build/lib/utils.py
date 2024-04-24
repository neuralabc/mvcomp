import nibabel as nb
import numpy as np
import os
import glob
import re
import time
# from numba import jit
import tempfile
import subprocess as subproc


def mrconvert(dwi_mif, dwi_nii):
    ''''
    Convenience function converts mif files to nifti with mrtrix3
    Mrtrix3 must be installed and available on the command line
    Args:
        dwi_mif (string): address of mif file.
        dwi_nii (string): address of nifti file to be created.

    Returns:
        This function creates a nifti file from the imported mif file and save it in the nifti file address (dwi_nii)

    '''

    subproc.call(['mrconvert', dwi_mif, dwi_nii])


def gen_fname_list(in_dir, glob_search_string):
    """
    Convenience function to generated alphanumerically sorted filename list
    Returns alphanumeric sorted list, uses glob
    """
    return mysort(glob.glob(os.path.join(in_dir, glob_search_string)))


def mah_dist_mat(feature_mat1, feature_mat2, pinv_s):
    """
    Compute Mahalanobis distance between two feature matrices.

    Args:
        feature_mat1, feature_mat2 (numpy.ndarray): 2D feature matrix in the shape of (number of voxels) x (number of features)
        pinv_s (numpy.ndarray): pseudo-inverse of the covariance matrix of size (number of features) x (number of features)

    Returns:
        dist (numpy array): an array of distances between two feature matrices. The distance calculation is voxelwise and all the voxels in the first feature matrix are compared to their corresponding ones in the second feature matrix. So the result is and 1D array of size (number of voxels).


    """
    diff_mat = (feature_mat1 - feature_mat2)
    num_els = diff_mat.shape[0]
    dist = np.zeros(num_els)
    for idx in range(num_els):
        dist_temp = np.dot((diff_mat[idx, :]), pinv_s)
        dist[idx] = np.dot(dist_temp, (diff_mat[idx, :]))
    return dist


def mah_dist_mat_2_roi_legacy(feature_mat, roi_feature_vec, pinv_s):
    """
    Calculate distance from all elements of feature_mat to roi_feature_vec
    """

    num_els = feature_mat.shape[0]
    dist = np.zeros(num_els)
    for idx in range(num_els):
        vox1 = feature_mat[idx, :]
        dist_temp = np.dot((vox1 - roi_feature_vec).T, pinv_s)
        dist[idx] = np.dot(dist_temp, (vox1 - roi_feature_vec))

    return dist


def mah_dist_mat_connectivity(feature_mat, pinv_s):
    """
    Calculate distance matrix from all elements of feature_mat. For space efficiency and to prevent memory over loading,
    all args and the result are in "Float16" format

    Args:
        feature_mat (numpay.ndarray): 2D feature matrix in the shape of (number of voxels) x (number of features)
        pinv_s (numpy array): pseudo-inverse of the covariance matrix.

    Returns:
        dist_mem (mem-map): 2D distance matrix of size (number of voxels) x (number of voxels)

    """
    import dask.array as da
    #     compressor = Blosc(cname='zlib', clevel=3, shuffle=Blosc.BITSHUFFLE)
    #     dist_zarr = zarr.zeros((feature_mat.shape[0], feature_mat.shape[0]), chunks=(20000, 20000), compressor=compressor, dtype='float16')

    st1 = time.time()
    tf = tempfile.NamedTemporaryFile()

    dist_mem = np.memmap(tf, dtype='float16', mode='w+',
                         shape=(feature_mat.shape[0], feature_mat.shape[0]))

    # dist = np.zeros((1000, feature_mat.shape[0]), dtype='float16')

    n = feature_mat.shape[0]
    f = np.array([], dtype='float16')
    f_mat = np.array([], dtype='float16')

    #     upper = np.triu_indices_from(dist_mem,k=1)

    for i in range(feature_mat.shape[0]):

        st = time.time()

        if i % 100 == 0:
            f = np.tile(feature_mat[i, :], (feature_mat.shape[0], 1)).copy()
            f_mat = feature_mat.copy()
        else:
            f = np.concatenate(
                (f, np.tile(feature_mat[i, :], (feature_mat.shape[0], 1))), axis=0)
            f_mat = np.concatenate((f_mat, feature_mat), axis=0)

        if i == feature_mat.shape[0] - 1:
            f_sub = da.from_array(f - f_mat)
            pinv_s_dask = da.from_array(pinv_s)
            dist_temp = da.dot(f_sub, pinv_s_dask)
            matrix_multi = dist_temp * f_sub
            matrix_multi = matrix_multi.astype('float16')

            dist_1 = da.dot(matrix_multi, da.ones(
                (feature_mat.shape[1], 1), dtype='float16'))

            dist_mem[i - (feature_mat.shape[0] - 1) %
                     100:feature_mat.shape[0]] = np.reshape(dist_1, (-1, n))

        if i % 100 == 99:
            f_sub = da.from_array(f - f_mat)
            pinv_s_dask = da.from_array(pinv_s)
            dist_temp = da.dot(f_sub, pinv_s_dask)
            matrix_multi = dist_temp * f_sub
            matrix_multi = matrix_multi.astype('float16')

            dist_1 = da.dot(matrix_multi, da.ones(
                (feature_mat.shape[1], 1), dtype='float16'))

            dist_mem[i - 99:i + 1] = np.reshape(dist_1, (-1, n))

            f = np.array([], dtype='float16')
            f_mat = np.array([], dtype='float16')

    return dist_mem


def connectivity_map(
        feature_in_dir,
        model_dir,
        suffix_name="warped_multcon_feature.nii.gz",
        mask_f=None,
        mask_img=None,
        mask_threshold=0.9,
        subject_ids=None,
        feat_sub=[]):
    ''''

    Args:
        feature_in_dir (String): The working directory that contains all the subjects and the model features directory
        model_dir (String): The directory address of the model that contains all the features of it
        suffix_name (String): The suffix of the features (model and individuals have the same naming format).
        mask_f (String): full pathname to the mask
        mask_img (nibabel): A nibabel object of the mask
        mask_threshold (float): A number in range 0-1 that determines the treshold for the mask (certainty of the mask)
        subject_ids (List of Strings): A list of strings containing the IDs of the subjects we want to work with them. if empty, a list of all the subjects available in the working directory will be created.
        feat_sub (List of strings): The name of the features we don't want to involve in the distance calculation.

    Returns:
        all_dist (numpy.ndarray): 3D array of size (number of voxels) x (number of voxels) x (number of subjects) that contains distance matrix of all the subjects.
        all_mask (numpy.ndarray): 2D array of size (number of voxels) x (number of subjects) that is all one except in the locations of nan\\inf.

    '''

    # Create subject_ids list from input directory if it is not in the input
    # args
    if subject_ids is None:
        subject_ids = subject_list(feature_in_dir)

    # Create a list of the features from the model. htis list contains the
    # location address of the features.
    model_feature_image_fname_list, model_f_list = feature_list(
        model_dir, suffix_name, feat_sub)

    # create feature matrix from the model
    m_f_mat, mask_img, mat_mask = feature_gen(
        model_feature_image_fname_list, mask_image_fname=mask_f, mask_image=mask_img, mask_threshold=0.9)

    # compute the covariance and invert it, since we need to compute only once
    s, pinv_s = norm_covar_inv(m_f_mat, mat_mask)

    # loop over individuals, compute Mahalanobis d
    all_feat = np.zeros((m_f_mat.shape + (len(subject_ids),)))
    all_dist = np.zeros(
        (m_f_mat.shape[0],
         m_f_mat.shape[0],
         len(subject_ids)),
        dtype="float16")
    all_mask = np.zeros((m_f_mat.shape[0], len(subject_ids)))

    for idx, subject_id in enumerate(subject_ids):

        st = time.time()
        comp_image_fname_list = []
        comp_f_list = []
        for f_name in model_f_list:
            if os.path.isfile(
                feature_in_dir +
                subject_id +
                "/" +
                f_name +
                    suffix_name):
                comp_image_fname_list.append(
                    feature_in_dir + subject_id + "/" + f_name + suffix_name)
                comp_f_list.append(f_name)

        # now we check to see if our metrics are going to be in the same order
        if not (model_f_list == comp_f_list):
            print(
                ">>You do not have exactly the same metric names for your model and comparison images (stopping)<<\n\tmodel:\t{}\n\tcomp:\t{}".format(
                    model_f_list,
                    comp_f_list))
            break
        else:  # everything is OK! lets do the comparison!
            c_f_mat, _, sub_mat_mask = feature_gen(
                comp_image_fname_list, mask_image=mask_img, mask_threshold=0.9)  # extract features from each individual

            all_feat[..., idx] = c_f_mat
            all_mask[..., idx] = sub_mat_mask
            print("subject ", subject_id,
                  "feature matrix creation in ", time.time() - st, "s")

    num_subs = all_feat.shape[-1]
    sub_mask = np.ones(num_subs).astype(bool)
    for idx in range(num_subs):  # for each subject
        st = time.time()
        all_feat = all_feat.astype('float16')
        #     pinv_s = pinv_s.astype('float16')
        # do MHD
        all_dist[..., idx] = mah_dist_mat_connectivity(
            all_feat[..., idx], pinv_s)
        print("Distance matrix calculation in {}s".format(time.time() - st))

        return all_dist, all_mask


def mah_dist_2_ROI(feature_mat, ROI_feature_vec, pinv_s, loc):
    """
    Deprecated: compute distance between ROI_feature_vec and feature_mat at specified loc (voxel indices x,y,z)
    """
    # loc determines the location of the voxel we want to calculate distance
    x = loc[0]
    y = loc[1]
    z = loc[2]
    vox1 = np.ravel(np.expand_dims(feature_mat[x, y, z], axis=1))
    vox2 = ROI_feature_vec
    dist_temp = np.dot((vox1 - vox2).T, pinv_s)
    dist = np.dot(dist_temp, (vox1 - vox2))
    return dist


def gen_roi_feature_vec_mat(
        feature_image_fname_list,
        region_roi_mask,
        summary_metric='mean'):
    """
    Extract feature vector from ROI mask (filename)

    """
    feature_mat, mask_img, mask = feature_gen(feature_image_fname_list, feature_in_dir=None,
                                              mask_image_fname=region_roi_mask, mask_image=None)
    if summary_metric == 'mean':
        roi_feature_vec = np.mean(feature_mat, axis=0)
    elif summary_metric == 'median':
        roi_feature_vec = np.median(feature_mat, axis=0)
    return roi_feature_vec


def correlation_from_covariance(cov_mat):
    """
    Compute the correlation matrix from the covariance matrix.
    from: https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    Args:
        cov_mat (numpy.ndarray): covariance matrix of size feature x feature

    Returns:
        corr_mat (numpy.ndarray): correlation matrix of size feature x feature
    """
    v = np.sqrt(np.diag(cov_mat))
    outer_v = np.outer(v, v)
    corr_mat = cov_mat / outer_v
    corr_mat[cov_mat == 0] = 0
    return corr_mat


def compute_average_simplified(
        model_feature_images_fname_list,
        out_dir,
        model_feature_list=None,
        verbose=0):
    """
    Refactored simplified version to work with lists only
    """

    num_features = len(model_feature_images_fname_list[0])

    model_feature_average_images_fname_list = []

    if model_feature_list is None:
        zfill_num = len(str(num_features)) + 1
        model_feature_list = np.arange(num_features).astype(
            str)  # if not provided, we just have values
        feature_names = [_ff.zfill(zfill_num) for _ff in model_feature_list]
    else:
        feature_names = model_feature_list
    print(f"features are {feature_names}")
    print("================================")

    for _idx in range(num_features):  # iterate over features
        if verbose > 0:
            print(f"Feature {_idx}")

        # get all the filenames of the first feature
        fnames = [_ff[_idx] for _ff in model_feature_images_fname_list]
        if verbose > 0:
            print(
                f"Found {len(fnames)} {feature_names[_idx]} files, concatenating")

        _imgs = nb.concat_images(fnames)
        _imgs_data = _imgs.get_fdata()
        print(f"shape of concatenated images is {_imgs_data.shape}")
        if verbose > 0:
            print(
                f"computing mean on the 4th axis, for {_imgs_data.shape[3]} subjects")

        _model_data = np.mean(_imgs_data, axis=-1)
        if verbose > 0:
            print(f"shape of the average image is {_model_data.shape}")

        _model_img = nb.Nifti2Image(_model_data, _imgs.affine, _imgs.header)

        if verbose > 0:
            print(
                f"saving to {out_dir}/{feature_names[_idx]}_{_imgs_data.shape[3]}_average.nii.gz")
        full_fname = f"{out_dir}/{feature_names[_idx]}_{_imgs_data.shape[3]}_average.nii.gz"
        nb.save(_model_img, full_fname)
        model_feature_average_images_fname_list.append(full_fname)
        print("------------------------------")
    print(f"Averages saved to {out_dir}")
    return model_feature_average_images_fname_list


def precomputed_spatial_mvcomp(
        subjects_matrix,
        model_feature_vector,
        pinv_s,
        return_raw=False):
    '''
    Convenience function for computing D2 on 2D input matrix consisting subjects and single metric when model feature vector
    and pseudo-inverse of the covariance matrix are already computed (e.g., mean values extracted from a set of tracts).

    Args:
        subjects_matrix (subjects X tracts/ROIs): matrix of a feature to compute D2 on
        model_feature_vector: vector from the model for comparison to subjects_matrix
        pinv_s: inverse of the covariance matrix
        return_raw: whether to return the distance between every subject's tract and the mean tract values, accouting for covariance between tracts

    Returns:
        all_dist (subjects X 1): the mahalanobis distnce for each subject
        raw_dist (subject x tract): if return_raw=True, the distance between every subject's tract and the mean tract values, accouting for covariance between tracts
    '''
    m_f_mat = model_feature_vector
    diff_mat = (subjects_matrix - m_f_mat[..., :])
    all_dist = (np.dot(diff_mat, pinv_s) * diff_mat)
    if return_raw:
        results = {'all_dist': np.array(all_dist).sum(
            axis=1), 'raw_dist': all_dist}
    else:
        results = {'all_dist': np.array(all_dist).sum(axis=1)}
    return results
