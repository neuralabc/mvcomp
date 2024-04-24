import math
import os
import glob
import re
import time

import nibabel as nb
import numpy as np

import matplotlib.pyplot as plt
import nilearn.plotting as nip


def mysort(l):
    """
    Sort the given iterable alphanumerically, in the way that humans expect.
    based on: https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python

    """
    def convert(text): return int(text) if text.isdigit() else text
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def compute_average(ids, in_dir, out_dir, features=[], feature_suffix=".nii.gz", verbose=0):
    """
    Computes averages (to be used as reference) for each feature based on selected subject IDs. 

    Args:
        ids (list): desired participants IDs
        in_dir (string): directory where subjects subdirectories are
        out_dir (string): output directory to save average images in
        features (list of strings): list of features names (e.g., FA)
        feature_suffix (string): suffix of feature files. features
            and feature_suffix should create the file names for the features wanted.
        verbose (int): level of verbosity. 
            0 = only important steps, 1 = more detailed.
    """

    feature_fnames = features
    print(f"features are {feature_fnames}")
    print("================================")

    for _idx, feature_fname in enumerate(feature_fnames):
        if verbose > 0:
            print(f"working on {feature_fname}")

        fnames = [os.path.join(in_dir, str(
            _id), feature_fname + feature_suffix) for _id in ids]
        if verbose > 0:
            print(f"Found {len(fnames)} {feature_fname} files, concatenating")

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
              f"saving to {out_dir}{os.sep}{feature_fname}_{_imgs_data.shape[3]}average.nii.gz")
        nb.save(
          _model_img, f"{out_dir}{os.sep}{feature_fname}_{_imgs_data.shape[3]}average.nii.gz")

        print("------------------------------")
    print(f"averages saved to {out_dir}")


def compute_average_simplified(model_feature_images_fname_list,
                            out_dir,
                            model_feature_list=None,
                            verbose=0):
    """
    Refactored simplified version to work with lists only
        and return a list of the created averages.
    Each feature must be in the same order for each participant
        and in the same order as the model.

    Computes averages (to be used as reference) for each feature based on selected subject IDs.

    Args:
        model_feature_images_fname_list (list): list of lists, 
            where the first dimension (outer) is subject and the second (inner) is feature
            all features must be in the same order for each individual
        model_feature_list (list): list of feature names for input features, 
            when None (default) will generate a list of indices starting at 00
            must be in the same order as those for the individual, 
            or files will be incorrectly labeled
        out_dir (string): output directory to save average images in
        model_feature_list (list of strings): list of features names (e.g., FA)
        verbose (int): level of verbosity. 0 = only important steps, 1 = more detailed.

    Returns:
        model_feature_average_images_fname_list (list): 
            full paths to feature average images, in same order as input
            of form <out_dir>_<feature_name>_numFeatures_average.nii.gz
    """

    num_features = len(model_feature_images_fname_list[0])

    model_feature_average_images_fname_list = []

    if model_feature_list is None:
        zfill_num = len(str(num_features))+1
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
              f"saving to {out_dir}{os.sep}{feature_names[_idx]}_{_imgs_data.shape[3]}_average.nii.gz")
        full_fname = f"{out_dir}{os.sep}{feature_names[_idx]}_{_imgs_data.shape[3]}_average.nii.gz"
        nb.save(_model_img, full_fname)
        model_feature_average_images_fname_list.append(full_fname)
        print("------------------------------")
    print(f"Averages saved to {out_dir}")
    return model_feature_average_images_fname_list


def feature_list(feature_in_dir, suffix_name, remove_list=[]):
    """
    This function is to create a list of the features names
    (and of their full paths) from the reference (model)

    Args:
        1. feature_in_dir (String): defines the directory where the features are located
        2. suffix_name (String): defines the suffix that comes after the features name 
            (e.g. in "T1_divided_T2_mean.nii.gz" the suffix is "_mean.nii.gz"
        3. remove_list (List of strings): A list of the features names 
            we want to exclude from calculation

    Returns:
        1. feature_image_fname_list (List of strings): 
            A list of the full pathnames of the features.
        2. f_list (List of strings): A list of features names.

    """
    feature_image_fname_list = mysort(
        glob.glob(os.path.join(feature_in_dir, "*" + suffix_name)))
    f_list = [os.path.basename(mod).replace(suffix_name, "")
              for mod in feature_image_fname_list]

    if len(remove_list)>0:
        for i, rem in enumerate(remove_list):
            feature_image_fname_list.remove(
              feature_image_fname_list[f_list.index(rem)])
            f_list.remove(rem)

    print("Features are : ", f_list)

    return feature_image_fname_list, f_list


def feature_gen(feature_image_fname_list,
                feature_in_dir=None,
                mask_image_fname=None,
                mask_image=None,
                verbosity=0,
                mask_threshold=0):
    """
    Creates a 2D feature matrix of size (number of voxels) x (number of features) 
        from a set of images (often used on reference images)
    Args:
        feature_image_fname_list (list of strings):
            a list of full pathnames of the reference images 
        feature_in_dir (string): path of directory that contains all the reference images
            (This could likely be removed since first arg is required 
            and already contains full file paths)
        mask_image_fname (string): Full pathname of the mask used for analysis. 
            mask_image takes precedence over this.
        mask_image (nibabel object): nibabel object of the mask
        verbosity (int): if not zero, it prints additional information 
        mask_threshold (float): a number in the range of 0-1 that determines 
            the threshold to apply on non-binarized mask.  

    Returns:
        feature_mat (numpay.ndarray): 2D feature matrix 
            in the shape of (number of voxels) x (number of features)
        mask_img (nibabel object): In the case that 
            we have mask_image as input it is the same as that, 
            otherwise, it is the nibabel object of mask_image_fname.
        feature_mat_vec_mask (numpy boolian array): 
            Lookup vector of size (number of voxels) 
            that is zero(False) where there are nans or infs. 

    """
    feature_dict = {}
    if feature_in_dir is None:
        feature_in_dir = ""
    for idx, feature_image_fname in enumerate(feature_image_fname_list):
        feature_img = nb.load(os.path.join(
            feature_in_dir, feature_image_fname))
        if (mask_image_fname is not None) and (mask_image is None):
            mask_img = nb.load(mask_image_fname)
            aff = mask_img.affine
            header = mask_img.header

            mask = mask_img.get_fdata() > mask_threshold
        elif mask_image is not None:
            mask_img = mask_image
            mask = mask_img.get_fdata() > mask_threshold
        else:  # generate a mask where voxels == 0
            if (idx == 0):
                if verbosity > 0:
                    print(
                        "Generating a mask from the first input feature image,\
                            where values == 0")
                mask = feature_img.get_fdata() != 0  # assume only zeros will become mask
        # overwrite the mask image, since we may have a new threshold
        mask_img = nb.Nifti1Image(mask, affine=feature_img.affine,
                                  header=feature_img.header)  
        feature_data = feature_img.get_fdata()[mask]
        feature_dict[feature_image_fname] = feature_data
        if idx == 0:
            # a mask for out of bounds data (nan and inf)
            feature_mat_vec_mask = np.zeros(mask.sum())

    # construct the feature matrix
    feature_mat = np.zeros((feature_data.shape[0], len(feature_dict.keys())))
    for idx, feature in enumerate(feature_dict):
        if verbosity > 0:
            print("{}".format(feature))
        feature_mat[:, idx] = feature_dict[feature]

    feature_mat_vec_mask += np.isnan(feature_mat).sum(axis=-1)
    # this will NOT be binary
    feature_mat_vec_mask += np.isinf(feature_mat).sum(axis=-1)
    # now we have zeros where the nan and infs are
    feature_mat_vec_mask = np.logical_not(feature_mat_vec_mask.astype(bool))

    if verbosity > 0:
        print("")
    # returns original or computed mask img and a vector lookup for good data (non nan/inf)
    return feature_mat, mask_img, feature_mat_vec_mask


def norm_covar_inv(feature_mat, mask=None):
    """
    Computes the covariance of the model and returns its pseudo-inverse (Moore-Penrose)
    Args:
        feature_mat (numpay.ndarray): 2D feature matrix 
            in the shape of (number of voxels) x (number of features)
        mask (numpy array): A vector that works as a mask (nan/inf = 0 otherwise = 1). 
            If not provided, the pseudo-inverse will be computed on the entire feature matrix

    Returns:
        s (numpy array): covariance matrix of size (number of features) x (number of features)
        pinv_s (numpy array): pseudo-inverse of the covariance matrix.

    """

    # get all the voxels inside the mask and compute covariance matrix
    if mask is not None:
        feature_masked = feature_mat[mask, :]
    else:
        feature_masked = feature_mat
    # compute covariance matrix across dimensions, within mask
    s = np.cov(feature_masked.T)
    pinv_s = np.linalg.pinv(s)
    return s, pinv_s  # return pinv of covariance matrix and cov mat


def correlation_fig(s, f_list):
    """
    Plot the correlation table from the covariance matrix.

    Args: 
        s (numpy.ndarray): covariance matrix of size feature x feature
        f_list (List of strings): A list of the names of the features 
            that should be in the same order of the covarinca matrix.
    """

    r = np.zeros(s.shape)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            r[i, j] = s[i, j] / (math.sqrt(s[i, i] * s[j, j]))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(r, cmap='RdBu_r', vmin=-1, vmax=1)
    title = ax.set_title('Metric-Metric Correlation Matrix')
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(s.shape[1]))
    ax.set_yticks(np.arange(s.shape[0]))
    xtl = ax.set_xticklabels(f_list)
    # Let the horizontal axes labeling appear on top.
    ytl = ax.set_yticklabels(f_list)
    _ = ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    _ = plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(f_list)):
        for j in range(len(f_list)):
            if i == j:
                pass
            else:
                val = r[i, j]
                if abs(val) > 0.6:
                    cc = 'w'
                else:
                    cc = 'k'
                text = ax.text(j, i, "{:.2f}".format(val),
                               ha="center", va="center", color=cc)


def mah_dist_feat_mat(feature_mat1, feature_mat2, pinv_s, return_raw=False):
    """
    Computes the Mahalanobis distance (D2) between two feature matrices 
        with the option of returning raw distances.
    Args:
        feature_mat1, feature_mat2 (numpy.ndarray): 
            2D feature matrix in the shape of (number of voxels) x (number of features)
        pinv_s (numpy.ndarray): 
            pseudo-inverse of the covariance matrix of size 
            (number of features) x (number of features)
        return_raw (boolean): If it is false the function returns D2, 
            otherwise, it returns raw distances (one distance value for each feature). 

    Returns: 
        results (numpy array): A 1D array of size (number of voxels) 
            containing D2 values between two feature matrices.
        Or
        raw_dist (numpy.ndarray): 2D array of size (number of voxels) x (number of features) 
            that contains raw distances for each feature.

    """

    diff_mat = (feature_mat1 - feature_mat2)

    dist = (np.dot(diff_mat, pinv_s) * diff_mat)

    if return_raw:
        raw_dist = dist
        return raw_dist
    else:
        results = dist.sum(axis=1)
        return results


def subject_list(root_dir, ex_subjects=[]):
    """
    This function is to create a list of all the subjects in the root directory.

    Args:
        root_dir (string): Address of the working directory that contains 
            all the subjects (The subjects folder names should be all number e.g. "001020").
        ex_subjects (List of strings): A list of subjects to be excluded from the subjects list.

    Returns:
        subject_ids (List of strings): A list of all the subjects 
            inside the root_dir, except the ones specified in ex_subjects.

    """

    subject_ids = os.listdir(root_dir)
    ext = []  # to exclude the non-subject folders
    for idx, subj in enumerate(subject_ids):
        if not subj.isdigit():
            ext.append(subj)

    for idx, subj in enumerate(ext):
        subject_ids.remove(subj)

    if len(ex_subjects) != 0:
        for i in ex_subjects:
            subject_ids.remove(i)

    return subject_ids


def model_comp(feature_in_dir, 
                model_dir=None,
                suffix_name_comp=".nii.gz",
                exclude_comp_from_mean_cov=True,
               suffix_name_model=".nii.gz",
               mask_f=None,
               mask_img=None,
               verbosity=1,
               mask_threshold=0.9,
               subject_ids=[],
               exclude_subject_ids=[],
               feat_sub=[],
               return_raw=False):
    
    '''
    This function is desiged to loop over a list of subjects and
    return a D2 array of size (number of voxels) x (number of subjects) 
    with the option of returning raw distances. 

    Args:
        feature_in_dir (String): The working directory that contains all the subjects' subdirectories 
        model_dir (String): The directory containing the reference images (feature averages). 
            Not needed if exclude_comp_from_mean_cov is set to True
        suffix_name_comp (String): The suffix of the subjects' files 
        suffix_name_model (String): The suffix of the reference files
        exclude_comp_from_mean_cov (bool): If True (default), 
            does not include the subject for which D2 is being calculated in the reference
        mask_f (String): full pathname of the mask
        mask_img (nibabel): A nibabel object of the mask
        mask_threshold (float): A number in range 0-1 that determines the threshold of the mask
        subject_ids (List of Strings): A list of strings containing the IDs of the subjects 
            we want to calculate D2 for. If empty, a list of IDs will be created 
            from all the subdirectories in feature_in_dir.
        exclude_subject_ids (list of str): List of subject IDs (str) to exclude from analysis
        feat_sub (List of strings): The names of the features we don't want to include in D2 calculation.
        return_raw (bool): If True, also returns raw distances

    Returns:
        dict with the following
        {'all_dist' (numpy.ndarray): 2D array of size (number of voxels) x (number of subjects) 
            that contains voxelwise D2 for all subjects.
        'all_mask' (numpy.ndarray): 2D array of size (number of voxels) x (number of subjects) 
            that is all ones except in the locations of nan\inf.
        'subject_ids': subject IDs
        'feature_names': feature names
        'raw_dist' (numpy.ndarray): if return_raw=True. 
            3D array of size (number of voxels) x (number of features) x (number of subjects)
            that contains the voxel-wise raw distances for each feature}
    '''

    # If there is no reference directory (model_dir), then we should have set the exclude_comp_from_mean_cov to True
    if model_dir is None:
        if not (exclude_comp_from_mean_cov):
            print("You must either set a model directory (model_dir)\
            or iteratively compute leave one out models on the fly \
            with set exclude_comp_from_mean_cov=True")
            print("Exiting")
            return 0

    # Create subject_ids list from input directory if it is not in the input args
    if len(subject_ids) == 0:
        if exclude_subject_ids is None:
            subject_ids = subject_list(feature_in_dir)
        else:
            subject_ids = subject_list(
                feature_in_dir, ex_subjects=exclude_subject_ids)

    print(subject_ids)
    # if a model is to be used
    # Create a list of the features from the model. 
    # This list contains the location address of the features.
    if model_dir is not None:
        model_feature_image_fname_list, model_feature_list = feature_list(
            model_dir, suffix_name_model, feat_sub)
        # We don't care that our comparison is within the mean, 
        # then we can compute this one time
        m_f_mat, mask_img, mat_mask = feature_gen(model_feature_image_fname_list,
                                                  mask_image_fname=mask_f,
                                                  mask_image=mask_img,
                                                  mask_threshold=mask_threshold)

        # compute the covariance and invert it, since we need to compute only once
        s, pinv_s = norm_covar_inv(m_f_mat, mat_mask)

        # prep output matrices
        if return_raw:

            all_feat = np.zeros((m_f_mat.shape + (len(subject_ids),)))
            raw_dist = np.zeros(
                (m_f_mat.shape[0], m_f_mat.shape[1], len(subject_ids)))
            all_mask = np.zeros((m_f_mat.shape[0], len(subject_ids)))

        else:
            all_feat = np.zeros((m_f_mat.shape + (len(subject_ids),)))
            raw_dist = np.zeros((m_f_mat.shape[0], len(subject_ids)))
            all_mask = np.zeros((m_f_mat.shape[0], len(subject_ids)))

    # if there's no model, just grab feature names from the first subject
    else:
        model_feature_image_fname_list, model_feature_list = feature_list(
          f"{feature_in_dir}{os.sep}{subject_ids[0]}{os.sep}", suffix_name_comp, feat_sub)


    # create feature matrix from the model

    # loop over individuals, compute D2
    for idx, subject_id in enumerate(subject_ids):

        st = time.time()

        comp_image_fname_list = []
        comp_f_list = []
        for feature_name in model_feature_list:
            if verbosity >= 2:
                print(f"Feature: {feature_name}")
            # try to be flexible for identifying the individual comparison file, 
            # this is not ideal for all cases

            full_comp_path_fname = os.path.join(os.path.join(
                feature_in_dir, subject_id), "*" + feature_name + "*" + suffix_name_comp)
            if verbosity >= 2:
                print(
                    f'-- Full comp path name identified as: {full_comp_path_fname}')
            full_comp_path_fname = glob.glob(full_comp_path_fname)

            if len(full_comp_path_fname) == 1:
                full_comp_path_fname = full_comp_path_fname[0]
            else:
                print(f"File does not exist:\n{full_comp_path_fname}")
                break

            comp_image_fname_list.append(full_comp_path_fname)
            comp_f_list.append(feature_name)

            # comp_image_fname_list has all path to all subjects features, comp_f_list has the names of features

        # now we check to see if our metrics are going to be in the same order
        if not (model_feature_list == comp_f_list):
            print(
                ">>You do not have exactly the same metric names for your model\
                    and comparison images (stopping)<<\n\tmodel:\t{}\n\tcomp:\t{}".format(
                    model_feature_list, comp_f_list))
            break

        else:  # everything is OK! lets do the comparison!
            c_f_mat, _, sub_mat_mask = feature_gen(comp_image_fname_list,
                                                mask_image_fname=mask_f,
                                                mask_image=mask_img,
                                                mask_threshold=mask_threshold)  # extract features from model
            if idx == 0:
                # we have not defined the output matrices yet in this case, so define here
                if exclude_comp_from_mean_cov:  
                    all_feat = np.zeros((c_f_mat.shape[0], len(
                        comp_image_fname_list), len(subject_ids)))
                    all_mask = np.zeros((c_f_mat.shape[0], len(subject_ids)))

                    if return_raw:
                        raw_dist = np.zeros(
                            (c_f_mat.shape[0], c_f_mat.shape[1], len(subject_ids)))
                    else:
                        raw_dist = np.zeros(
                            (c_f_mat.shape[0], len(subject_ids)))

            all_feat[..., idx] = c_f_mat
            all_mask[..., idx] = sub_mat_mask
            if verbosity >= 1:
                print("subject {} feature matrix creation in {:.3} s".format(
                    subject_id, time.time()-st))

    num_subs = len(subject_ids)
    st = time.time()
    for idx in range(num_subs):  # for each subject
        # we remove the subject that is going to be compared from the mean (reference)
        #  and pinv_s calculation so that they are independent
        if exclude_comp_from_mean_cov:
            m_f_mat = np.mean(np.delete(all_feat, idx, axis=-1), axis=-1)
            # mask specific to voxels from ALL subjects
            s, pinv_s = norm_covar_inv(
                m_f_mat[:,], np.sum(all_mask, axis=-1) == num_subs)

        # compute D2
        raw_dist[..., idx] = mah_dist_feat_mat(
            all_feat[..., idx], m_f_mat, pinv_s, return_raw=return_raw)

    print("Total time for mahalanobis distance calculation on {}\
        subjects with {} voxels: {:.3}s".format(
        num_subs, sub_mat_mask.shape[0], time.time() - st))

    # if we set return_raw=True then we still need to compute D2 
    # by summing across the features of the 3d array returned by mah_dist_feat_mat
    # if we did not, raw_dist only contains the distance (2d)
    if return_raw:
        all_dist = raw_dist.sum(axis=1)
        results = {'all_dist': all_dist, 'all_mask': all_mask, 'subject_ids': subject_ids,
                   'feature_names': model_feature_list, "raw_dist": raw_dist}
    else:
        all_dist = raw_dist
        results = {'all_dist': all_dist, 'all_mask': all_mask,
                   'subject_ids': subject_ids, 'feature_names': model_feature_list}

    return results


def dist_plot(all_dist,
                all_mask,
                subject_ids,
                feat_sub=[],
                save_results=True,
                out_dir=None,
                mask_f=None,
                mask_img=None,
                coordinate=(-10, -50, 10),
                vmin=None,
                vmax=5,
                hist_tr=100,
                nobin=100):
    '''
    Plots the mean of all subjects' D2 maps and the histogram of D2 values. 
     It also saves all the subjects' D2 maps alongside the mean D2 map to the result directory. 
     The naming format of the folder it creates in the result directory 
     depends on the features that were excluded during calculation 
     and on the number of subjects used. 
     e.g. if folder's name is "results_without_MD_18", 
     it means we had 18 subjects and we didn't use MD in the D2 calculation.

     Args:
        all_dist (numpy.ndarray): 
            2D array of size (number of voxels) x (number of subjects)
            that contains voxelwise D2 of all subjects.
        all_mask (numpy.ndarray): 
            2D array of size (number of voxels) x (number of subjects)
            that is all one except in the locations of nan\inf.
        subject_ids (List of Strings): A list of strings containing the IDs of the subjects.
        feat_sub (List of strings): 
            The name of the features that were not involved in D2 calculation.
        save_results (Boolian): If True, results will be saved.
        out_dir (String): Directory where we want to save the results.
        mask_f (String): full pathname of the mask
        mask_img (nibabel): A nibabel object of the mask
        coordinate (Tuple): Location of the crosshair at which the plot will be centered.
        vmin (Float): Lower limit of intensity
        vmax (Float): Upper limit of intensity
        hist_tr (Float): Maximum D2 value we want to be shown in the histogram.
        nobin (int): Number of bins for the histogram. 
     '''
    # Load mask
    if mask_f is not None:
        mask_img = nb.load(mask_f)

    mat_out = np.zeros(mask_img.shape)
    mask_out = np.zeros(mask_img.shape)
    # create the mask of nan and inf values
    oob_mask = np.prod(all_mask, axis=-1) == 0

    d_out = np.mean(all_dist, axis=-1)
    d_out[oob_mask] = 0  # just zero the values that are fringe

    print("number of not NAN voxels: ", np.count_nonzero(d_out))

    m_out = np.mean(all_mask, axis=-1)
    m_out[oob_mask] = 0

    mat_out[mask_img.get_fdata().astype(bool)] = d_out

    mask_out[mask_img.get_fdata().astype(bool)] = m_out

    img_out = nb.Nifti1Image(
        mat_out, affine=mask_img.affine, header=mask_img.header)
    img_out.update_header()
    allmask_out = nb.Nifti1Image(
        mask_out, affine=mask_img.affine, header=mask_img.header)
    allmask_out.update_header()

    nip.plot_img(img_out,
                 cut_coords=coordinate,
                 display_mode='ortho',
                 vmin=vmin, vmax=vmax,
                 colorbar=True, cmap='viridis')
    plt.figure()
    hh = plt.hist(d_out[d_out < hist_tr], bins=nobin)

    nip.plot_img(allmask_out,
                 cut_coords=coordinate,
                 display_mode='ortho',
                 vmin=vmin, vmax=vmax,
                 colorbar=True, cmap='viridis')

    if save_results:
        if out_dir is None:
            result_dir = f'..{os.sep}'
        else:
            result_dir = out_dir

        feat_str = ""
        for idx, feat in enumerate(feat_sub):
            feat_str = feat_str + "_" + feat

        if len(feat_sub) != 0:
            if not os.path.isdir(result_dir + "results_without"
                                + feat_str + "_" + str(len(subject_ids))):
                os.makedirs(result_dir + "results_without" +
                            feat_str + "_" + str(len(subject_ids)))
            result_dir = result_dir + "results_without" + \
                feat_str + "_" + str(len(subject_ids))
        else:
            if not os.path.isdir(result_dir + "results_with_allfeatures" 
                                + "_" + str(len(subject_ids))):
                os.makedirs(result_dir + "results_with_allfeatures" +
                            "_" + str(len(subject_ids)))
            result_dir += "results_with_allfeatures" + \
                "_" + str(len(subject_ids))

        nb.save(img_out, result_dir + os.sep + "mean" + 
                str(len(subject_ids)) + "subjects.nii.gz")

        for idx, subject_id in enumerate(subject_ids):
            d_out = all_dist[:, idx]
            # just zero the values that are fringe
            d_out[all_mask[:, idx] == 0] = 0

            mat_out[mask_img.get_fdata().astype(bool)] = d_out

            img_out = nb.Nifti1Image(
                mat_out, affine=mask_img.affine, header=mask_img.header)
            img_out.update_header()
            nb.save(img_out, result_dir + os.sep + "xxx_" + subject_id + ".nii.gz")
        print('data has been saved to output directory: {}'.format(result_dir))


def model_comp_simplified(comp_images_fname_list,
                            subject_ids=None,
                            model_feature_list=None,
                            model_feature_image_fname_list=None,
                            return_raw=False,
                            mask=None,
                            mask_threshold=0,
                            verbosity=1):
    """
    Simplified version of model_comp to work with list of lists as input. 
    Each feature must be in the same order 
    for each participant and in the same order as the model.

    When model_feature_image_fname_list == None, 
        model mean and covariance is generated by leaving out all but the current subject being compared.
    When model_feature_image_fname_list is provided, 
        mean and covariance are based on the specified files. 

    Args:
        comp_images_fname_list (list): list of lists, 
            where the first dimension (outer) is subject and second (inner) 
            is feature. All features must be in the same order for each individual
        subject_ids (list, optional): List of subject IDs in the same order as in comp_images_fname_list. If none, 
            indices are generated. IDs are carried through to the output dictionary.
            Defaults to None.
        model_feature_list (list): 
            list of feature names for input features, 
            when None (default) will generate a list of indices starting at 00
            must be in the same order as those for the individual, or files will be incorrectly labeled
        model_feature_image_fname_list (list, optional): 
            List of feature average images for use as model. Defaults to None.
        return_raw (bool, optional): If True, also returns raw distances. Defaults to False.
        mask (nibabel.image, optional): 
            A nibabel object of the mask or path to nibabel convertable object. 
            Defaults to None. If None, mask is generated based on first input feature 
            for each subject (likely not ideal).
        mask_threshold (float, optional): Threshold cutoff for mask. Defaults to 0.
        verbosity (int, optional): Controls how much output is printed to standard out. Defaults to 1.

    Returns:
        dict with the following
        {
            'all_dist' (numpy.ndarray): 
                2D array of size (number of voxels) x (number of subjects) 
                that contains voxelwise D2 for all subjects.
            'all_mask' (numpy.ndarray): 
                2D array of size (number of voxels) x (number of subjects)
                that is all ones except in the locations of nan\inf.
            'subject_ids': subject IDs
            'feature_names': feature names
            'raw_dist' (numpy.ndarray): 
                if return_raw=True. 3D array of size (number of voxels) x (number of features) x (number of subjects)
                 that contains the voxel-wise raw distances for each feature
        }
    """
    num_subjects = len(comp_images_fname_list)
    num_features = len(comp_images_fname_list[0])

    # if values were not provided for subject IDs or feature names, then we fill with None
    # TODO: make smarter based on regex
    if subject_ids is None:
        subject_ids = [None]*num_subjects
    if model_feature_list is None:
        model_feature_list = [None]*num_features

    if isinstance(mask, str):
        mask_f = mask
        mask_img = None
    else:  # TODO either check if this is a nibabel object or catch an exception
        mask_img = mask
        mask_f = None

    if model_feature_image_fname_list is not None:
        exclude_comp_from_mean_cov = False
        print("You provided a list of features to serve as the model comparison")
        print("\t- All computed D2 values will be relative to the provided model")
    else:
        exclude_comp_from_mean_cov = True
        print("No 'model_feature_image_list' was provided")
        print("\t- Model features will be iteratively computed as the mean of all other subjects (leave one out)")

    # create feature matrix from the model
    # if we don't care that our comparison is within the mean, 
    # then we can compute this one time
    if not exclude_comp_from_mean_cov:  
        m_f_mat, mask_img, mat_mask = feature_gen(model_feature_image_fname_list,
                                                  mask_image_fname=mask_f,
                                                  mask_image=mask_img,
                                                  mask_threshold=mask_threshold)

        # compute the covariance and invert it, since we need to compute only once
        # this will be used below
        s, pinv_s = norm_covar_inv(m_f_mat, mat_mask)

        # prep output matrices
        if return_raw:

            all_feat = np.zeros((m_f_mat.shape + (num_subjects,)))
            raw_dist = np.zeros(
                (m_f_mat.shape[0], m_f_mat.shape[1], num_subjects))
            all_mask = np.zeros((m_f_mat.shape[0], num_subjects))

        else:
            all_feat = np.zeros((m_f_mat.shape + (num_subjects,)))
            # actually will just contain the distnaces, not the raw weightings
            raw_dist = np.zeros((m_f_mat.shape[0], num_subjects))
            all_mask = np.zeros((m_f_mat.shape[0], num_subjects))

    # loop over individuals, compute Mahalanobis d
    for idx, comp_image_fname_list in enumerate(comp_images_fname_list):
        st = time.time()
        # extract features for this individual
        c_f_mat, _, sub_mat_mask = feature_gen(comp_image_fname_list,
                                                mask_image_fname=mask_f,
                                                mask_image=mask_img,
                                                mask_threshold=mask_threshold)  

        if idx == 0:
            # we have not defined the output matrices yet in this case, so define here
            if exclude_comp_from_mean_cov:  
                if return_raw:
                    all_feat = np.zeros(
                        (sub_mat_mask.shape[0], num_features, num_subjects))
                    raw_dist = np.zeros(
                        (sub_mat_mask.shape[0], sub_mat_mask.shape[1], num_subjects))
                    all_mask = np.zeros((sub_mat_mask.shape[0], num_subjects))
                else:
                    all_feat = np.zeros(
                        (sub_mat_mask.shape[0], num_features, num_subjects))
                    # actually will just contain the distnaces, not the raw weightings
                    raw_dist = np.zeros((sub_mat_mask.shape[0], num_subjects))
                    all_mask = np.zeros((sub_mat_mask.shape[0], num_subjects))
        all_feat[..., idx] = c_f_mat
        all_mask[..., idx] = sub_mat_mask
        if verbosity >= 1:
            print("subject {} feature matrix creation in {:.3} s".format(
                subject_ids[idx], time.time()-st))

    st = time.time()
    for idx in range(num_subjects):  # for each subject
        # we remove the subject that is going to be compared 
        # from the mean and pinv_s calculation so that they are independent
        if exclude_comp_from_mean_cov:  
            m_f_mat = np.mean(np.delete(all_feat, idx, axis=-1), axis=-1)
            # mask specific to voxels from ALL subjects
            s, pinv_s = norm_covar_inv(m_f_mat[:,], np.sum(
                all_mask, axis=-1) == num_subjects)

        # compute D2
        raw_dist[..., idx] = mah_dist_feat_mat(
            all_feat[..., idx], m_f_mat, pinv_s, return_raw=return_raw)

    print("Total time for mahalanobis distance calculation on {}\
        subjects with {} voxels: {:.3}s".format(
        num_subjects, sub_mat_mask.shape[0], time.time() - st))

    # if we set return_raw=True then we still need to compute D2 
    # by summing across the features of the 3d array returned by mah_dist_feat_mat
    # if we did not, raw_dist only contains the distance (2d)
    if return_raw:
        all_dist = raw_dist.sum(axis=1)
        results = {'all_dist': all_dist, 'all_mask': all_mask, 'subject_ids': subject_ids,
                   'feature_names': model_feature_list, "raw_dist": raw_dist}
    else:
        all_dist = raw_dist
        results = {'all_dist': all_dist, 'all_mask': all_mask,
                   'subject_ids': subject_ids, 'feature_names': model_feature_list}

    return results


def spatial_mvcomp(subjects_matrix,
                    return_raw=False,
                    exclude_comp_from_mean_cov=False):
    '''
    To compute D2 between subjects and a reference, where dimensions combined are spatial dimensions 
    (e.g., WM tracts or ROIs).
    Args:
        subjects_matrix (subjects X tracts/ROIs): 
            2D matrix of size (number of subjects) x (number of tracts)
            (e.g., mean FA values in a set of WM tracts).
        return_raw (bool): whether to return the raw distances
        exclude_comp_from_mean_cov: 
            If True, the subject being compared is excluded from the mean (reference) and covariance computation
    Returns:
        all_dist (subjects X 1): D2 value for each subject
        raw_dist (subject x tract): if return_raw=True, the raw distances for each tract
    '''
    if exclude_comp_from_mean_cov == False:
        # compute tract-wise covariance
        s, pinv_s = norm_covar_inv(subjects_matrix)
        # compute subject-wise mean for each and every tract
        m_f_mat = np.mean(subjects_matrix, axis=0)
        diff_mat = (subjects_matrix - m_f_mat)
        all_dist = (np.dot(diff_mat, pinv_s) * diff_mat)

    else:
        # all_dist = np.zeros(subjects_matrix.shape[0])
        all_dist = np.zeros(subjects_matrix.shape)
        for subject_idx in range(subjects_matrix.shape[0]):
            _dropped_matrix = np.delete(
                subjects_matrix, subject_idx, axis=0)  # drop the subject
            # compute the mean along all other subject, for each tract
            m_f_mat = np.mean(_dropped_matrix, axis=0)
            # compute tract-wise covariance
            s, pinv_s = norm_covar_inv(_dropped_matrix)

            diff_mat = (subjects_matrix[subject_idx, :] - m_f_mat)
            dist = (np.dot(diff_mat, pinv_s) * diff_mat)
            all_dist[subject_idx, :] = dist
    if return_raw:
        results = {'all_dist': np.array(all_dist).sum(
            axis=1), "raw_dist": all_dist}
    else:
        results = {'all_dist': np.array(all_dist).sum(axis=1)}
    return results


def mah_dist_mat_2_roi(feature_mat, roi_feature_vec, pinv_s, return_raw=False):
    """
    Calculates voxelwise D2 values between feature_mat and roi_feature_vec 
    with the option of returning raw distances.
    Args:
        feature_mat (numpy.ndarray): 
            2D array of size (number of voxels) X (number of features)
            that we want to compute D2 over.
        roi_feature_vec (numpy.ndarray): 
            1D array of size (number of features) containing ROI-averaged feature values (reference). 
        pinv_s (numpy.ndarray): 
            pseudo-inverse of the covariance matrix of size 
            (number of features) x (number of features)
        return_raw (boolean): If it is false the function returns D2, 
            otherwise, it returns raw distances (one distance value for each feature).
    Returns:
        results (dict): 
            if return_raw is True: 
                        all_dist (numpy array): 
                            A 1D array of size (number of voxels) 
                            containing D2 values between two feature matrices.
                        and 
                        raw_dist (numpy.ndarray): 
                            2D array of size (number of voxels) x (number of features) 
                            that contains raw distances for each feature.
            otherwise:
                    all_dist (numpy array): A 1D array of size (number of voxels) containing D2 values between two feature matrices.
    """

    diff_mat = (feature_mat-roi_feature_vec[..., :])
    raw_dist = np.dot(diff_mat, pinv_s) * diff_mat

    if return_raw:
        all_dist = raw_dist.sum(axis=1)
        results = {'all_dist': all_dist, "raw_dist": raw_dist}
    else:
        all_dist = raw_dist
        results = {'all_dist': all_dist}
    return results


def voxel2voxel_dist(subdir,
                    suffix_name_comp=".nii.gz",
                    mask_f=None,
                    mask_img=None,
                    mask_threshold=None,
                    feat_sub=[],
                    return_raw=False):
    '''
    Computes D2 between each voxel and all other voxels in a mask, within a single subject.

    Args:
        subdir (string): subject directory path that contains their feature images
        suffix_name_comp (str): The suffix of the features image files
        mask_f (String): full pathname of the mask
        mask_img (nibabel): A nibabel object of the mask
        mask_threshold (float): A number in range 0-1 that determines the threshold of the mask 
        feat_sub (List of strings): The names of the features we don't want to include in D2 calculation.

    Returns:
        all_dist (numpy array): A symmetrical D2 array of D2 values of the shape (Number of voxels) X (Number of voxels)        
    '''

    my_subject_features_path, my_subject_features = feature_list(
        subdir, suffix_name_comp, feat_sub)
    comp_image_fname_list = []
    comp_f_list = []

    _submat, _submask, _matmask = feature_gen(
        feature_image_fname_list=my_subject_features_path, mask_image_fname=mask_f, mask_image=mask_img, mask_threshold=0.9)

    s, pinv_s = norm_covar_inv(_submat, _matmask)

    all_dist = np.zeros((_submat.shape[0], _submat.shape[0]))

    for _vox in range(_submat.shape[0]):
        _dict = mah_dist_mat_2_roi(
            _submat, _submat[_vox], pinv_s, return_raw=return_raw)
        all_dist[_vox, :] = _dict["all_dist"].sum(axis=1)

    return all_dist
