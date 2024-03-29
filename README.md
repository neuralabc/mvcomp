# Multivariate Comparisons (MVComp)

**Authors:** Stefanie Tremblay @stremblay18, Zaki Alasmar @zklsmr, Amir Pirhadi @amirpirhadi, Felix Carbonell, Yasser Iturria-Medina, Claudine Gauthier, Christopher Steele @steelec

A set of functions to compute Mahalanobis D (D**2) on multimodal MRI images in a group of individuals in comparison to a reference group or within a single individual.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10713027.svg)](https://doi.org/10.5281/zenodo.10713027)

[Biorxiv link](https://www.biorxiv.org/content/10.1101/2024.02.27.582381v1)

Reference:
```
Tremblay, SA, Alasmar, Z, Pirhadi, A, Carbonell, F, Iturria-Medina, Y, Gauthier, C, Steele, CJ, (under review) MVComp toolbox: MultiVariate Comparisons of brain MRI features accounting for common information across metrics (submitted)
```

# Mahalanobis d

The Mahalanobis distance, or D2, is defined as the multivariate distance between a point and a distribution in which covariance between features (e.g., imaging metrics) is accounted for (Mahalanobis, 1936). D2 is thus similar to a z-score, but applicable to multivariate data because when computing D2, the shape of the distribution is taken into account such that the values that are more probable have lower distance values (Taylor et al., 2020). In other words, D2 tells us how improbable a certain combination of features is. For example, because height and weight are highly correlated, very tall individuals that have a very low weight would appear outside of the distribution and would have a high D2 value. Relationships between variables are thus accounted for in the D2 framework by dividing the Euclidean distance by the covariance matrix, which also scales the variables to have unit variance.

The Mahalanobis distance approach has been used extensively in outlier detection, cluster analysis, and classification applications (e.g., Kritzman & Li, 2010; Xiang et al., 2008; Ghorbani, 2019). D2 has also been used in neuroimaging, mainly in the study of neurological disorders, to detect lesions (Gyebnár et al., 2019; Lindemer et al., 2015), or to evaluate the degree of abnormality in the brains of patients relative to controls (Dean et al., 2017; Owen et al., 2021; Taylor et al., 2020), but also to study healthy WM development (Kulikova et al., 2015). Despite promising implementations and its high versatility, D2 has not yet been widely adopted. To facilitate its use, we present here an open-source python-based tool for computing D2 relative to a reference group or within a single individual: the MultiVariate Comparison (MVComp) toolbox.

## Usage

Depending on the application, a different set of functions should be used. See corresponding section below.

**Comparisons between subject(s) and a reference – Combining MRI metrics**

- `compute_average` : To compute the group average maps of each metric (will serve as the reference).
- `feature_gen` : Apply to the reference group average maps to extract the feature matrix (m_f_mat of shape n voxels x n features), a mask vector (mat_mask of shape n voxels) and a nibabel object of the mask (mask_img).
- `norm_covar_inv` : To compute the covariance matrix (s) and its pseudoinverse (pinv_s) from the reference feature and mask matrices (m_f_mat and mat_mask).
- `correlation_fig` : To generate a correlation matrix figure from the covariance matrix (s). For visualization.
- `model_comp` : To calculate voxel-wise D2 between each subject contained in the provided subject_ids list and the reference (group average). Yields a D2 matrix of size number of voxels x number of subjects.
*For leave-one-out approach, set the `exclude_comp_from_mean_cov` option to True (the previous steps can be skipped in this case since a new covariance matrix is computed for each subject, within the `model_comp` function).
- `dist_plot` : To produce D2 maps for every subject from the D2 matrix generated by `model_comp`.
- `model_comp` with  `return_raw` set to True : To extract features contribution to D2 in a region of interest. When `return_raw` is set to True, the function returns a 3D array of size (number of voxels) x (number of metrics) x (number of subjects). This information can then be summarized to obtain the % contribution of each metric for a group of subjects.

**Comparisons between subject(s) and a reference – Combining spatial dimensions**

- `spatial_mvcomp` : To compute a D2 score between each subject and the reference from a matrix containing the data (e.g., mean FA in each WM tract) of all subjects (n subjects x n tracts). Returns a vector with a single D2 value per subject.
*For leave-one-out approach, set the `exclude_comp_from_mean_cov` option to True.
- `spatial_mvcomp` with `return_raw` set to True : To extract features contribution to D2. If set to True, a 2D array of size (number of subjects) x (number of tracts) is returned. This information can then be summarized to obtain the relative importance of each tract to D2.

**Comparisons within a single subject – Voxel-wise D2 resolution**

- `feature_gen` : Provide the path of the images (i.e., one image per metric) and the reference ROI mask to this function to extract the feature matrix (m_f_mat of shape n voxels x n features), a mask vector (mat_mask of shape n voxels) and a nibabel object of the mask (mask_img). This function can also be used to extract the data inside the ROI of voxels to be evaluated.
- `norm_covar_inv` : To compute the covariance matrix (s) and its pseudoinverse (pinv_s) from the reference feature and mask matrices (m_f_mat and mat_mask).
- `correlation_fig` : To generate a correlation matrix figure from the covariance matrix (s). For visualization.
- `mah_dist_mat_2_roi` : To compute voxel-wise D2 between all voxels within a mask and a specific reference ROI. The user will need to provide a vector of data for the reference ROI (i.e., mean across voxels in the ROI for each metric), along with the feature matrix containing the data for the voxels to be evaluated.
- `mah_dist_mat_2_roi` with `return_raw` set to True : To extract features’ contributions. The output will be of shape (number of voxels) x (number of metrics).

**Comparisons within a single subject – Voxel-voxel matrix D2 resolution**

- `voxel2voxel_dist` : To compute D2 between each voxel and all other voxels in a mask. Yields a symmetric 2-D matrix of size n voxels x n voxels containing D2 values between each pair of voxels.

## Dependencies

The `mvcomp` core functionality requires only `numpy`, `matplotlib`, `nibabel`, and `nilearn`.

# Extras

## Simplified calling based only on lists of lists of filenames

While the core functionality is based around a filename and data storage structure that _should_ help to reduce human error when using these functions, additional versions of two of the main functions (`compute_average` and `model_comp`) have also been implemented in a simplified form that is based on lists of lists (subject X feature). These functions have _little to no error-checking_ and have no way to ensure that the correct data has been input in the correct order - this is the responsibility of the user. These functions are more flexible and can be used for data stored in any accessible location and with any filename convention (while still relying on nibabel readable objects) and have fewer required inputs to simplify the workflow for users who are familiar with the approach, but the user should take care to ensure that the **ordering of inputs is correct and consistent across calls**.
- `compute_average_simplified` : takes a list (subjects) of lists (features), otherswise works identically to `compute_average`. Feature names will be auto-generated as indices if not provided.
- `model_comp_simplified` : takes a list (subjects) of lists (features), otherswise core functionality works identically to `model_comp`. Leave-one-out D2 computations is supported by not specifying a list of model features for input. Subject IDs will be auto-generated as indices if not provided.

## Code examples
- jupyter notebooks in `./examples/*.ipynb`

## Plotting
- plotting code snippets in `./plotting.py`

## Utilities
- random potentially useful utility functions in `./utils.py`
