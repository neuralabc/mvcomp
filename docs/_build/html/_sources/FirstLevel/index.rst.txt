**Multivariate Comparisons (MVComp)**
=======================================

.. module:: mvcomp
    :synopsis: A set of functions to compute Mahalanobis D (D**2) on multimodal MRI images in a group of individuals in comparison to a reference group or within a single individual.



A set of functions to compute the Mahalanobis Distance (D\ :sup:`2`) 
on MRI images in a group of individuals in comparison to a reference group or within a single individual.

Tha Mahalanobis Distance (D\ :sup:`2`) is a measure of the distance between a point P (i.e. MRI features in a subject or an ROI) 
and a distribution D (i.e. the distribution of the same MRI features in a reference group or another ROI). 
The Mahalanobis Distance is a generalization of the Euclidean Distance that takes into account the covariance of the data, which is important given 
the high correlation between MRI-derived microstructural measures. 

The framework can also be used to integrate spatial information using a single MRI measure (e.g., FA across several white matter tracts). 

For a detailed description of how D\ :sup:`2` can be applied to neuroimaging data, please refer to our paper:

    .. code-block:: latex 

        Tremblay, SA, Alasmar, Z, Pirhadi, A, Carbonell, F, Iturria-Medina, Y, Gauthier, C, Steele, CJ, (2024). MVComp toolbox: MultiVariate Comparisons of brain MRI features accounting for common information across metrics.
        BioRxiv, https://doi.org/10.1101/2024.02.27.582381.

You will also find detailed steps different use cases here: https://mvcomp.readthedocs.io/en/latest/UserGuide/index.html 

=======================================
----------------------------
Installation
----------------------------
MVComp can be installed directly using pip:

.. code-block:: bash

    pip install mvcomp

Alternatively, you can install MVComp from the source code by clonning the repository from GitHub 

.. code-block:: bash

    git clone https://github.com/neuralabc/mvcomp.git
    cd mvcomp
    pip intsall .

After installation, you can import the package in your python script

.. code-block:: python

    import mvcomp as mvc




After installation, please run the three examples in the `./examples/` subdirectory to confirm that your installation is functioning correctly. 
As necessary, please open an `issue on github <https://github.com/neuralabc/mvcomp/issues>`_ to report any issues that may arise.



Quick Start
----------------------------

Here's an example of how to use MVComp if your data is ready and you wish to compute D2 voxel-wise between a subject and a reference group:

.. code-block:: python

    from mvcomp import model_comp

    result = model_comp(feature_in_dir, model_dir=None, suffix_name_comp=".nii.gz", exclude_comp_from_mean_cov=True,
                        suffix_name_model=".nii.gz", mask_f=None, mask_img=None, verbosity=1,
                        mask_threshold=0.9, subject_ids=None, exclude_subject_ids=None, feat_sub=[], return_raw=False)

    print(result)

Here, `feature_in_dir` is the path to the directory containing the MRI feature maps of all subjects and `model_dir` is the path to the directory containing the reference maps (to which each subject will be compared). You will also need to provide a mask to which you can apply a threshold using `mask_threshold`. To implement the leave-one-subject-out approach so that the subject under evaluation is excluded from D2 calculation, set `exclude_comp_from_mean_cov` to True. In this case, `model_dir` is not needed.
The output `result` is a dictionary containing a D2 matrix (`all_dist`) of size number of voxels x number of subjects.

Check out some example notebooks here:
    https://github.com/neuralabc/mvcomp/tree/main/examples

These use cases currently illustrated as runnable examples:

1. `Comparing a subject to a reference group <https://github.com/neuralabc/mvcomp/blob/main/examples/Example1_Voxel-wise_D2_subj_to_group_MVComp.ipynb>`_
2. `Comparing subjects based on spatial MRI metrics (i.e. tractwise multimodal features) <https://github.com/neuralabc/mvcomp/blob/main/examples/Example2_Spatial_D2_subj_to_group_MVComp.ipynb>`_
3. `Comparing a subject to themselves (i.e. within-subject comparison) <https://github.com/neuralabc/mvcomp/blob/main/examples/Example3_voxelvoxel_D2_within_subj_MVComp.ipynb>`_


Data organization
----------------------------

The data must be organized as such:

Subjects' directories are inside feature_in_dir (e.g., /my_project/processed_maps/) and their folder names consists in numbers only (e.g., 001, 002, etc.). Ensure your feature maps have consistent file names such that the file prefix is the name of the MRI measure (e.g., FA) and the suffix is the same across all features (e.g., suffix_name_comp = "_warped_to_group.nii.gz"). The MRI maps that will be used as reference should be a group average of all subjects (or of subjects of a control group) for each MRI measure. These maps should have the same prefix as the feature maps and they should be contained in `model_dir`.   

Example:

    Feature maps:

    /my_project/processed_maps/001/FA_warped_to_group.nii.gz
    /my_project/processed_maps/001/MD_warped_to_group.nii.gz
    /my_project/processed_maps/002/FA_warped_to_group.nii.gz
    /my_project/processed_maps/002/MD_warped_to_group.nii.gz
    ...

    Reference (average) maps:
    /my_project/average_reference_group/FA_warped_to_group_average.nii.gz
    /my_project/average_reference_group/MD_warped_to_group_average.nii.gz

    Args would thus be:
        - feature_in_dir = "/my_project/processed_maps/"
        - suffix_name_comp = "_warped_to_group.nii.gz"
        - model_dir = "/my_project/average_reference_group/"
        - suffix_name_model = "_warped_to_group_average.nii.gz"
    

For more details on the steps to follow for computing voxel-wise D2 between a subject and a reference group: https://mvcomp.readthedocs.io/en/latest/UserGuide/Combining_MRI_metrics.html 

You will also find detailed steps for other use cases here: https://mvcomp.readthedocs.io/en/latest/UserGuide/index.html  


Reference Us!
---------------------
If you use the MVComp package in your research, please cite the following paper:

    .. code-block:: latex 

        Tremblay, SA, Alasmar, Z, Pirhadi, A, Carbonell, F, Iturria-Medina, Y, Gauthier, C, Steele, CJ, (2024). MVComp toolbox: MultiVariate Comparisons of brain MRI features accounting for common information across metrics.
        BioRxiv, https://doi.org/10.1101/2024.02.27.582381.



License Information
-------------------
    .. small::

    the MVComp toolobox is licensed under the Apache License 2.0. you can use it, distribute it, modify it granted you provide the same license as well. 
    Refer to the LICENSE file for more information.

=====================


