Comparisons between subject(s) and a reference – Combining spatial dimensions

A single MRI measure can be used and combined across multiple ROIs (e.g., mean FA in pre-defined WM tracts). The reference is defined as the group mean of each tract and a single D2 value is computed for each subject. In this case, D2 represents a measure of how different a subject's WM microstructure is relative to a reference, across multiple tracts.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **spatial_mvcomp**: To compute a D2 score between each subject and the reference from a matrix containing the data (e.g., mean FA in each WM tract) of all subjects (n subjects x n tracts). Returns a vector with a single D2 value per subject.
    \* For leave-one-out approach, set the `exclude_comp_from_mean_cov` option to True.
    
    \* `spatial_mvcomp` with `return_raw` set to True: To extract features contribution to D2. If set to True, a 2D array of size (number of subjects) x (number of tracts) is returned. This information can then be summarized to obtain the relative importance of each tract to D2.
    
    **Args**:
        - `subjects_matrix` (subjects X tracts/ROIs): 2D matrix of size (number of subjects) x (number of tracts) (e.g., mean FA values in a set of WM tracts).
        - `return_raw` (bool): whether to return the raw distances
        - `exclude_comp_from_mean_cov`: If True, the subject being compared is excluded from the mean (reference) and covariance computation
    Returns:
        - `all_dist` (subjects X 1): D2 value for each subject
        - `raw_dist` (subject x tract): if return_raw=True, the raw distances for each tract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
