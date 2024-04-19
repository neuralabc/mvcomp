Comparisons between subject(s) and a reference: Combining MRI metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here the reference would be defined as the voxel-wise group average for each MRI measure and D2 is computed by comparing the feature values in each voxel of an individual to the corresponding voxel in the reference. If two groups are being analyzed (e.g., patients vs controls), the control group could be used as the reference and D2 values computed between each patient and the reference would represent voxel-wise multivariate distance from controls. 

To ensure that each subject’s data will not bias their D2 values in single sample designs (i.e., where the entire sample is used as a reference) and to allow the evaluation of controls in two-sample designs, a leave-one-subject-out approach is possible. In this way, the subject under evaluation is excluded from the group mean (reference) and covariance matrix prior to calculating D2. To implement the leave-one-subject-out approach, skip to the `model_comp` function and set the `exclude_comp_from_mean_cov` option to True.

For details on how to organize data see: https://mvcomp.readthedocs.io/en/latest/FirstLevel/index.html#data-organization

- **compute_average**: To compute the group average maps of each metric (will serve as the reference). 
    Args:
        - `ids` (list): desired participants IDs (numbers only, e.g., 001, 002, etc.)
        - `in_dir` (string): directory where subjects subdirectories are
        - `out_dir` (string): output directory to save average images in
        - `features` (list of strings): list of features names (e.g., "FA")
        - `feature_suffix` (string): suffix of feature files (e.g., "_warped_to_group.nii.gz"). features and feature_suffix should create the file names for the features wanted (e.g., "FA_warped_to_group.nii.gz").
        - `verbose` (int): level of verbosity. 0 = only important steps, 1 = more detailed.

- **feature_gen**: Apply to the reference group average maps to extract the feature matrix (m_f_mat of shape n voxels x n features), a mask vector (mat_mask of shape n voxels) and a nibabel object of the mask (mask_img).
    Args:
        - `feature_image_fname_list` (list of strings): a list of full pathnames of the reference images 
        - `feature_in_dir` (string): path of directory that contains all the reference images (This could likely be removed since first arg is required and already contains full file paths)
        - `mask_image_fname` (string): Full pathname of the mask used for analysis. mask_image takes precedence over this.
        - `mask_image` (nibabel object): nibabel object of the mask
        - `verbosity` (int): if not zero, it prints additional information 
        - `mask_threshold` (float): a number in the range of 0-1 that determines the threshold to apply on non-binarized mask.  
    
    Returns:
        - `feature_mat` (numpy.ndarray): 2D feature matrix in the shape of (number of voxels) x (number of features)
        - `mask_img` (nibabel object): In the case that we have `mask_image` as input it is the same as that, otherwise, it is the nibabel object of `mask_image_fname`.
        - `feature_mat_vec_mask` (numpy boolean array): Lookup vector of size (number of voxels) that is zero (`False`) where there are nans or infs. 

- **norm_covar_inv**: To compute the covariance matrix (s) and its pseudoinverse (pinv_s) from the reference feature and mask matrices (m_f_mat and mat_mask).
    Args:
        - `feature_mat` (numpy.ndarray): 2D feature matrix in the shape of (number of voxels) x (number of features)
        - `mask` (numpy array): A vector that works as a mask (nan/inf = 0 otherwise = 1). If not provided, the pseudo-inverse will be computed on the entire feature matrix
                            
    Returns:
        - `s` (numpy array): covariance matrix of size (number of features) x (number of features)
        - `pinv_s` (numpy array): pseudo-inverse of the covariance matrix.

- **correlation_fig**: To generate a correlation matrix figure from the covariance matrix (s). For visualization.
    Args: 
        - `s` (numpy.ndarray): covariance matrix of size feature x feature
        - `f_list` (List of strings): A list of the names of the features that should be in the same order as the covariance matrix.

- **model_comp**: To calculate voxel-wise D2 between each subject contained in the provided subject_ids list and the reference (group average). Yields a D2 matrix of size number of voxels x number of subjects.
    \* For leave-one-out approach, set the `exclude_comp_from_mean_cov` option to True (the previous steps can be skipped in this case since a new covariance matrix is computed for each subject, within the `model_comp` function).
    
    \* `model_comp` with  `return_raw` set to True: To extract features contribution to D2 in a region of interest. When `return_raw` is set to True, the function returns a 3D array of size (number of voxels) x (number of metrics) x (number of subjects). This information can then be summarized to obtain the % contribution of each metric for a group of subjects.

    Args:
        - `feature_in_dir` (String): The working directory that contains all the subjects' subdirectories 
        - `model_dir` (String): The directory containing the reference images (feature averages). Not needed if `exclude_comp_from_mean_cov` is set to True
        - `suffix_name_comp` (String): The suffix of the subjects' files 
        - `suffix_name_model` (String): The suffix of the reference files
        - `exclude_comp_from_mean_cov` (bool): If True (default), does not include the subject for which D2 is being calculated in the reference
        - `mask_f` (String): full pathname of the mask
        - `mask_img` (nibabel): A nibabel object of the mask
        - `mask_threshold` (float): A number in range 0-1 that determines the threshold of the mask
        - `subject_ids` (List of Strings): A list of strings containing the IDs of the subjects we want to calculate D2 for. If empty, a list of IDs will be created from all the subdirectories in `feature_in_dir`.
        - `exclude_subject_ids` (list of str): List of subject IDs (str) to exclude from analysis
        - `feat_sub` (List of strings): The names of the features we don't want to include in D2 calculation.
        - `return_raw` (bool): If True, also returns raw distances

    Returns:
        - dict with the following
        {'all_dist' (numpy.ndarray): 2D array of size (number of voxels) x (number of subjects) that contains voxelwise D2 for all subjects.
        'all_mask' (numpy.ndarray): 2D array of size (number of voxels) x (number of subjects) that is all ones except in the locations of nan\inf.
        'subject_ids': subject IDs
        'feature_names': feature names
        'raw_dist' (numpy.ndarray): if return_raw=True. 3D array of size (number of voxels) x (number of features) x (number of subjects) that contains the voxel-wise raw distances for each feature}
    
- **dist_plot**: To produce D2 maps for every subject from the D2 matrix generated by `model_comp`.
    Args:
        - `all_dist` (numpy.ndarray): 2D array of size (number of voxels) x (number of subjects) that contains voxelwise D2 of all subjects.
        - `all_mask` (numpy.ndarray): 2D array of size (number of voxels) x (number of subjects) that is all one except in the locations of nan\inf.
        - `subject_ids` (List of Strings): A list of strings containing the IDs of the subjects.
        - `feat_sub` (List of strings): The name of the features that were not involved in D2 calculation.
        - `save_results` (Boolian): If True, results will be saved.
        - `out_dir` (String): Directory where we want to save the results.
        - `mask_f` (String): full pathname of the mask
        - `mask_img` (nibabel): A nibabel object of the mask
        - `coordinate` (Tuple): Location of the crosshair at which the plot will be centered.
        - `vmin` (Float): Lower limit of intensity
        - `vmax` (Float): Upper limit of intensity
        - `hist_tr` (Float): Maximum D2 value we want to be shown in the histogram.
        - `nobin` (int): Number of bins for the histogram.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
