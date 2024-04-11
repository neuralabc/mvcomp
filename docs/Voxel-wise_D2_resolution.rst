Comparisons within a single subject – Voxel-wise D2 resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **feature_gen**: Provide the path of the images (i.e., one image per metric) and the reference ROI mask to this function to extract the feature matrix (m_f_mat of shape n voxels x n features), a mask vector (mat_mask of shape n voxels) and a nibabel object of the mask (mask_img). This function can also be used to extract the data inside the ROI of voxels to be evaluated.
- **norm_covar_inv**: To compute the covariance matrix (s) and its pseudoinverse (pinv_s) from the reference feature and mask matrices (m_f_mat and mat_mask).
- **correlation_fig**: To generate a correlation matrix figure from the covariance matrix (s). For visualization.
- **mah_dist_mat_2_roi**: To compute voxel-wise D2 between all voxels within a mask and a specific reference ROI. The user will need to provide a vector of data for the reference ROI (i.e., mean across voxels in the ROI for each metric), along with the feature matrix containing the data for the voxels to be evaluated.
    \*`mah_dist_mat_2_roi` with `return_raw` set to True : To extract features’ contributions. The output will be of shape (number of voxels) x (number of metrics).
    Args:
        - `feature_mat` (numpy.ndarray): 2D array of size (number of voxels) X (number of features) that we want to compute D2 over.
        - `roi_feature_vec` (numpy.ndarray): 1D array of size (number of features) containing ROI-averaged feature values (reference). 
        - `pinv_s` (numpy.ndarray): pseudo-inverse of the covariance matrix of size (number of features) x (number of features)
        - `return_raw` (boolean): If it is false the function returns D2, otherwise, it returns raw distances (one distance value for each feature).
    Returns:
        - `results` (dict): 
            if return_raw is True: 
                        all_dist (numpy array): A 1D array of size (number of voxels) containing D2 values between two feature matrices.
                        and 
                        raw_dist (numpy.ndarray): 2D array of size (number of voxels) x (number of features) that contains raw distances for each feature.
            otherwise:
                    all_dist (numpy array): A 1D array of size (number of voxels) containing D2 values between two feature matrices.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~